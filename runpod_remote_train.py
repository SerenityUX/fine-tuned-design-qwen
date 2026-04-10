#!/usr/bin/env python3
"""
Run SFT or CPT on a RunPod GPU from your laptop: create pod → upload code+data → train → download adapter → delete pod.

Authentication
  - RUNPOD_API_KEY or RUNPOD_TOKEN (same value: your RunPod API key from the dashboard).
  - SSH: uses your private key (default ~/.ssh/id_ed25519 or id_rsa). Add the matching *public* key in RunPod account settings.

Do you need GitHub?
  - No. This script tars `generate.py`, training scripts, `requirements.txt`, and your data JSONL, then uploads via SFTP.
  - Optional: set --git-url to clone a repo on the pod instead of uploading a bundle (still no GitHub if you use another git host).

Storage (RunPod)
  - Container disk: ephemeral (image + temp); training output should go under /workspace.
  - Volume disk (`--volume-gb`): per-pod volume at /workspace; survives pod restarts but is removed when the pod is terminated.
  - Network volume (`--network-volume-id` / RUNPOD_NETWORK_VOLUME_ID): shared persistent disk at /workspace; survives pod deletion and can be reattached; API replaces the per-pod volume when set. Size/cost are fixed when you create the volume in the console.

Requires: requests, paramiko (see requirements.txt).

SSH modes
  - Default: root@public_ip -p mapped_port — supports SFTP (upload/download).
  - Gateway (--ssh-via-gateway): user@ssh.runpod.io:22 — RunPod docs say SCP/SFTP are not
    supported on the proxy; this script uses exec + stdin/tar streams instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests

ROOT = Path(__file__).resolve().parent
RUNPOD_REST = "https://rest.runpod.io/v1"
RUNPOD_SSH_GATEWAY_HOST = "ssh.runpod.io"
RUNPOD_SSH_GATEWAY_PORT = 22

DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# Files needed to run training (no local venv / no huge model dirs).
BUNDLE_PATHS = [
    "generate.py",
    "cpt_train.py",
    "sft_on_design_qa.py",
    "extract_data_from_books.py",
    "requirements.txt",
]


def _token() -> str:
    for k in ("RUNPOD_API_KEY", "RUNPOD_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    print(
        "Set RUNPOD_API_KEY or RUNPOD_TOKEN to your RunPod API key.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}


def _ssh_key_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    home = Path.home() / ".ssh"
    for name in ("id_ed25519", "id_rsa"):
        p = home / name
        if p.is_file():
            return p
    print("No SSH private key found; pass --ssh-key", file=sys.stderr)
    raise SystemExit(1)


def build_tarball(
    *,
    project_root: Path,
    data_files: list[Path],
    preload_adapter: Optional[Path],
    out_path: Path,
) -> None:
    """Write a .tar.gz with code + data (+ optional LoRA adapter to continue)."""
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in BUNDLE_PATHS:
            p = project_root / rel
            if not p.is_file():
                if rel == "extract_data_from_books.py":
                    continue
                raise FileNotFoundError(f"Missing required file for bundle: {p}")
            tar.add(p, arcname=f"DesignModel/{rel}")
        for df in data_files:
            df = df.resolve()
            if not df.is_file():
                raise FileNotFoundError(df)
            arc = f"DesignModel/data/{df.name}"
            tar.add(df, arcname=arc)
        if preload_adapter is not None:
            pad = preload_adapter.resolve()
            if not pad.is_dir():
                raise NotADirectoryError(pad)
            for fp in pad.rglob("*"):
                if fp.is_file():
                    arc = Path("DesignModel") / "preload_adapter" / fp.relative_to(pad)
                    tar.add(fp, arcname=str(arc))


def create_pod(
    *,
    name: str,
    gpu_type_id: str,
    image: str,
    cloud_type: str,
    container_disk_gb: int,
    volume_gb: int,
    env: Optional[dict[str, str]] = None,
    network_volume_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a pod. If ``network_volume_id`` is set, it attaches that Network Volume at ``/workspace`` and the API omits per-pod ``volumeInGb`` (network volume replaces pod volume)."""
    body: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": container_disk_gb,
        "ports": ["22/tcp"],
        "volumeMountPath": "/workspace",
        "dockerStartCmd": ["/bin/bash", "-lc", "sleep infinity"],
    }
    nv = (network_volume_id or "").strip()
    if nv:
        body["networkVolumeId"] = nv
    else:
        body["volumeInGb"] = volume_gb
    if env:
        body["env"] = env
    if cloud_type == "COMMUNITY":
        body["supportPublicIp"] = True
    r = requests.post(f"{RUNPOD_REST}/pods", headers=_headers(), json=body, timeout=120)
    if r.status_code not in (200, 201):
        print(r.text, file=sys.stderr)
        r.raise_for_status()
    return r.json()


def get_pod(pod_id: str) -> dict[str, Any]:
    r = requests.get(f"{RUNPOD_REST}/pods/{pod_id}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def wait_for_ssh(
    pod_id: str,
    *,
    timeout_s: float = 900,
    poll_s: float = 5.0,
) -> tuple[str, int]:
    """Return (public_ip, ssh_host_port)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = get_pod(pod_id)
        ip = (pod.get("publicIp") or "").strip()
        mappings = pod.get("portMappings") or {}
        # keys may be str or int
        ssh_port: Optional[int] = None
        for k, v in mappings.items():
            if str(k) == "22":
                ssh_port = int(v)
                break
        if ip and ssh_port:
            return ip, ssh_port
        time.sleep(poll_s)
    raise TimeoutError(f"Pod {pod_id} did not get publicIp + SSH port mapping in time.")


def wait_for_pod_running(
    pod_id: str,
    *,
    timeout_s: float = 900,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Poll until desiredStatus is RUNNING (for SSH via ssh.runpod.io; no public IP required)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = get_pod(pod_id)
        status = (pod.get("desiredStatus") or "").strip().upper()
        if status == "RUNNING":
            return pod
        time.sleep(poll_s)
    raise TimeoutError(f"Pod {pod_id} did not reach RUNNING in time.")


def delete_pod(pod_id: str) -> None:
    r = requests.delete(f"{RUNPOD_REST}/pods/{pod_id}", headers=_headers(), timeout=60)
    if r.status_code not in (200, 204):
        print(f"Warning: delete pod {pod_id}: {r.status_code} {r.text}", file=sys.stderr)


def _paramiko_connect(
    host: str,
    port: int,
    key_path: Path,
    *,
    username: str = "root",
    connect_attempts: int = 12,
    delay_s: float = 15.0,
):
    """Connect with retries — RunPod often exposes SSH before sshd accepts connections."""
    import paramiko

    pkey = None
    for loader in (
        paramiko.Ed25519Key,
        paramiko.RSAKey,
        paramiko.ECDSAKey,
    ):
        try:
            pkey = loader.from_private_key_file(str(key_path))
            break
        except Exception:
            continue
    if pkey is None:
        raise RuntimeError(f"Could not load SSH key {key_path}")

    last_exc: Optional[BaseException] = None
    for attempt in range(connect_attempts):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                pkey=pkey,
                timeout=90,
                banner_timeout=120,
                auth_timeout=120,
            )
            return client
        except Exception as e:
            last_exc = e
            try:
                client.close()
            except Exception:
                pass
            if attempt < connect_attempts - 1:
                print(
                    f"  SSH connect attempt {attempt + 1}/{connect_attempts} failed; retry in {delay_s:.0f}s…",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_s)
    raise last_exc  # type: ignore[misc]


def ssh_run(
    host: str,
    port: int,
    key_path: Path,
    command: str,
    *,
    username: str = "root",
    timeout: int = 86400,
    use_shell: bool = False,
) -> tuple[int, str, str]:
    if not use_shell:
        client = _paramiko_connect(host, port, key_path, username=username)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
            return code, out, err
        finally:
            client.close()

    # Some gateways (RunPod's ssh.runpod.io proxy) require a PTY-like shell session.
    # For those, use invoke_shell() and wait for an exit-code marker.
    client = _paramiko_connect(host, port, key_path, username=username)
    chan = None
    try:
        chan = client.invoke_shell(term="xterm", width=120, height=40)
        chan.settimeout(1.0)

        marker = "__RUNPOD_EXIT__"
        wrapped = f"bash -lc {shlex.quote(command)}; echo {marker}:$?; printf '\\n'\n"

        # Drain initial banner/prompt (non-deterministic; ignore content).
        buf = b""
        drain_deadline = time.time() + 8.0
        while time.time() < drain_deadline:
            if chan.recv_ready():
                buf += chan.recv(4096)
            else:
                time.sleep(0.1)

        chan.send(wrapped)

        out_buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if chan.recv_ready():
                out_buf += chan.recv(4096)
                text = out_buf.decode("utf-8", errors="replace")
                m = re.search(rf"{re.escape(marker)}:(\\d+)", text)
                if m:
                    code = int(m.group(1))
                    return code, text, ""
            else:
                time.sleep(0.2)
        raise TimeoutError(f"Timed out waiting for command completion ({timeout}s)")
    finally:
        try:
            if chan is not None:
                chan.close()
        except Exception:
            pass
        client.close()


def _ssh_put_file_via_exec(
    host: str,
    port: int,
    key_path: Path,
    username: str,
    local_path: Path,
    remote_path: str,
) -> None:
    """Stream file over SSH exec (works through ssh.runpod.io; SFTP does not)."""
    import paramiko

    rp = shlex.quote(remote_path)
    parent = shlex.quote(str(Path(remote_path).parent))
    client = _paramiko_connect(host, port, key_path, username=username)
    try:
        stdin, stdout, stderr = client.exec_command(
            f"mkdir -p {parent} && cat > {rp}",
            timeout=86400,
        )
        with local_path.open("rb") as lf:
            while True:
                chunk = lf.read(1024 * 1024)
                if not chunk:
                    break
                stdin.write(chunk)
        stdin.channel.shutdown_write()
        err_b = stderr.read()
        code = stdout.channel.recv_exit_status()
        if code != 0:
            raise RuntimeError(
                f"Remote write failed ({code}): {err_b.decode('utf-8', errors='replace')}"
            )
    finally:
        client.close()


def _ssh_get_dir_via_tar(
    host: str,
    port: int,
    key_path: Path,
    username: str,
    remote_dir: str,
    local_dir: Path,
) -> None:
    """Download directory by streaming tar over SSH (gateway-compatible)."""
    rd = shlex.quote(remote_dir.rstrip("/"))
    client = _paramiko_connect(host, port, key_path, username=username)
    try:
        stdin, stdout, stderr = client.exec_command(
            f"set -e; cd {rd} && tar czf - .",
            timeout=86400,
        )
        stdin.close()
        chunks: list[bytes] = []
        while True:
            b = stdout.read(256 * 1024)
            if not b:
                break
            chunks.append(b)
        err_b = stderr.read()
        code = stdout.channel.recv_exit_status()
        if code != 0:
            raise RuntimeError(
                f"Remote tar failed ({code}): {err_b.decode('utf-8', errors='replace')}"
            )
        data = b"".join(chunks)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
            tf.extractall(local_dir)
    finally:
        client.close()


def put_file_remote(
    host: str,
    port: int,
    key_path: Path,
    local_path: Path,
    remote_path: str,
    *,
    username: str = "root",
    use_sftp: bool = True,
) -> None:
    if use_sftp:
        client = _paramiko_connect(host, port, key_path, username=username)
        try:
            sftp = client.open_sftp()
            try:
                sftp.put(str(local_path), remote_path)
            finally:
                sftp.close()
        finally:
            client.close()
    else:
        _ssh_put_file_via_exec(
            host, port, key_path, username, local_path, remote_path
        )


def get_dir_remote(
    host: str,
    port: int,
    key_path: Path,
    remote_dir: str,
    local_dir: Path,
    *,
    username: str = "root",
    use_sftp: bool = True,
) -> None:
    if use_sftp:
        client = _paramiko_connect(host, port, key_path, username=username)
        try:
            sftp = client.open_sftp()

            def _walk(rpath: str, lpath: Path) -> None:
                lpath.mkdir(parents=True, exist_ok=True)
                for attr in sftp.listdir_attr(rpath):
                    name = attr.filename
                    if name in (".", ".."):
                        continue
                    rp = f"{rpath.rstrip('/')}/{name}"
                    lp = lpath / name
                    if attr.st_mode is not None and (attr.st_mode & 0o40000):
                        _walk(rp, lp)
                    else:
                        sftp.get(rp, str(lp))

            try:
                _walk(remote_dir, local_dir)
            finally:
                sftp.close()
        finally:
            client.close()
    else:
        _ssh_get_dir_via_tar(host, port, key_path, username, remote_dir, local_dir)


def sftp_put_file(
    host: str,
    port: int,
    key_path: Path,
    local_path: Path,
    remote_path: str,
    *,
    username: str = "root",
    use_sftp: bool = True,
) -> None:
    put_file_remote(
        host,
        port,
        key_path,
        local_path,
        remote_path,
        username=username,
        use_sftp=use_sftp,
    )


def sftp_get_dir(
    host: str,
    port: int,
    key_path: Path,
    remote_dir: str,
    local_dir: Path,
    *,
    username: str = "root",
    use_sftp: bool = True,
) -> None:
    """Recursively download remote_dir to local_dir (SFTP or tar-over-SSH)."""
    get_dir_remote(
        host,
        port,
        key_path,
        remote_dir,
        local_dir,
        username=username,
        use_sftp=use_sftp,
    )


def remote_train_sft(
    host: str,
    port: int,
    key_path: Path,
    *,
    ssh_username: str = "root",
    use_shell_exec: bool = False,
    output_name: str,
    steps: int,
    preload: bool,
    precision: str,
    base_model: str,
    preload_remote_path: Optional[str] = None,
    max_length: Optional[int] = None,
    lr: Optional[float] = None,
    grad_clip: Optional[float] = None,
) -> None:
    remote_root = "/workspace/DesignModel"
    preload_block = ""
    if preload_remote_path:
        preload_block = f'--preload-adapter "{preload_remote_path}"'
    elif preload:
        preload_block = f'--preload-adapter "{remote_root}/preload_adapter"'
    extra = []
    if max_length is not None:
        extra.append(f"--max-length {int(max_length)}")
    if lr is not None:
        extra.append(f"--lr {lr}")
    if grad_clip is not None:
        extra.append(f"--grad-clip {grad_clip}")
    extra_s = (" " + " ".join(extra)) if extra else ""
    cmd = f"""set -euo pipefail
cd {remote_root}
python3 -m pip install --upgrade pip -q
python3 -m pip install peft transformers accelerate pypdf huggingface_hub hf_transfer python-dotenv requests
python3 sft_on_design_qa.py --data data/sft_design_qa.jsonl --output {remote_root}/models/{output_name} --base {base_model} --steps {steps} {preload_block} --precision {precision}{extra_s}
"""
    code, out, err = ssh_run(
        host,
        port,
        key_path,
        cmd,
        username=ssh_username,
        timeout=86400,
        use_shell=use_shell_exec,
    )
    print(out)
    if err:
        print(err, file=sys.stderr)
    if code != 0:
        raise RuntimeError(f"Remote training failed with exit code {code}")


def remote_train_cpt(
    host: str,
    port: int,
    key_path: Path,
    *,
    ssh_username: str = "root",
    use_shell_exec: bool = False,
    data_name: str,
    output_name: str,
    steps: int,
    base_model: str,
    max_length: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> None:
    remote_root = "/workspace/DesignModel"
    extra = []
    if max_length is not None:
        extra.append(f"--max-length {int(max_length)}")
    if lr is not None:
        extra.append(f"--lr {lr}")
    if batch_size is not None:
        extra.append(f"--batch-size {int(batch_size)}")
    extra_s = (" " + " ".join(extra)) if extra else ""
    cmd = f"""set -euo pipefail
cd {remote_root}
python3 -m pip install --upgrade pip -q
python3 -m pip install peft transformers accelerate pypdf huggingface_hub hf_transfer python-dotenv requests
python3 cpt_train.py --data data/{data_name} --output {remote_root}/models/{output_name} --steps {steps} --base {base_model}{extra_s}
"""
    code, out, err = ssh_run(
        host,
        port,
        key_path,
        cmd,
        username=ssh_username,
        timeout=86400,
        use_shell=use_shell_exec,
    )
    print(out)
    if err:
        print(err, file=sys.stderr)
    if code != 0:
        raise RuntimeError(f"Remote training failed with exit code {code}")


def _resolve_cpt_dataset(project_root: Path) -> Path:
    for name in ("book_corpus.jsonl", "book_corpus_test.jsonl"):
        p = project_root / "data" / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No data/book_corpus.jsonl or data/book_corpus_test.jsonl under {project_root}")


def _extract_adapter_tar_gz_base64(
    *,
    tar_gz_b64: str,
    dest_dir: Path,
    overwrite: bool,
) -> None:
    import base64 as _base64
    import shutil as _shutil

    raw = _base64.b64decode(tar_gz_b64.encode("ascii"))

    if overwrite and dest_dir.exists():
        _shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(dest_dir)


def run_flash_train_cpt_then_sft_and_return_adapters(
    *,
    project_root: Path,
    local_models_dir: Path,
    endpoint_name: str,
    gpu_type_name: str,
    volume_name: Optional[str],
    volume_id: Optional[str],
    volume_datacenter_name: str,
    volume_size_gb: int,
    cpt_steps: int,
    sft_steps: int,
    cpt_lr: float,
    sft_lr: float,
    cpt_max_length: int,
    sft_max_length: int,
    sft_grad_clip: float,
    sft_precision: str,
    cpt_output_name: str,
    sft_output_name: str,
    overwrite_local: bool,
) -> None:
    """
    Flash implementation (serverless). Runs CPT then SFT on the GPU and returns adapters
    as base64-encoded tar.gz files, which we then extract locally.
    """

    import asyncio

    from runpod_flash import DataCenter, Endpoint, GpuGroup, GpuType, NetworkVolume

    gpu_type_name = (gpu_type_name or "").strip()
    if not gpu_type_name or gpu_type_name.upper() == "ANY":
        gpu = GpuGroup.ANY
    else:
        # Example expected values: "NVIDIA_L40S", "NVIDIA_GEFORCE_RTX_4090"
        gpu = getattr(GpuType, gpu_type_name)

    volume = None
    if volume_id:
        volume = NetworkVolume(id=volume_id)
    elif volume_name:
        dc = getattr(DataCenter, volume_datacenter_name)
        volume = NetworkVolume(name=volume_name, size=volume_size_gb, datacenter=dc)

    deps = [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "hf_transfer",
        "huggingface_hub",
        "python-dotenv",
        "requests",
    ]

    cpt_adapter_dir_name = cpt_output_name
    sft_adapter_dir_name = sft_output_name

    @Endpoint(
        name=endpoint_name,
        gpu=gpu,
        workers=(1, 1),
        idle_timeout=600,
        dependencies=deps,
        volume=volume,
        execution_timeout_ms=0,
    )
    # No `typing.Any` in annotations: Flash execs this on workers without our module imports.
    def train_adapters():
        import os
        import base64 as _base64
        import hashlib as _hashlib
        import shutil as _shutil
        from pathlib import Path
        from io import BytesIO as _BytesIO

        # Ensure Hugging Face cache goes onto the persistent volume (if attached).
        cache_root = Path("/runpod-volume") if Path("/runpod-volume").is_dir() else Path("/tmp")
        hf_home = cache_root / "hf"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_home)

        # Pick datasets from this repo (bundled in Flash build).
        repo_root = ROOT
        data_cpt = _resolve_cpt_dataset(repo_root)
        data_sft = repo_root / "data" / "sft_design_qa.jsonl"
        if not data_sft.is_file():
            raise FileNotFoundError(f"Missing {data_sft}")

        from generate import BASE_MODEL
        import cpt_train
        import sft_on_design_qa

        out_root = cache_root / "adapters"
        cpt_dir = out_root / cpt_adapter_dir_name
        sft_dir = out_root / sft_adapter_dir_name

        if cpt_dir.exists():
            _shutil.rmtree(cpt_dir)
        if sft_dir.exists():
            _shutil.rmtree(sft_dir)
        cpt_dir.mkdir(parents=True, exist_ok=True)
        sft_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: CPT -> produces LoRA adapter in cpt_dir
        cpt_train.continue_pretrain_on_model(
            data_jsonl=data_cpt,
            output_dir=cpt_dir,
            base_model=BASE_MODEL,
            steps=cpt_steps,
            batch_size=1,
            lr=cpt_lr,
            max_length=cpt_max_length,
        )

        # Stage 2: SFT on top of CPT adapter -> produces LoRA adapter in sft_dir
        sft_on_design_qa.run_sft_on_design_qa(
            data_jsonl=data_sft,
            output_dir=sft_dir,
            base_model=BASE_MODEL,
            preload_adapter=cpt_dir,
            steps=sft_steps,
            lr=sft_lr,
            max_length=sft_max_length,
            grad_clip=sft_grad_clip,
            precision=sft_precision,
        )

        def tar_dir_contents_to_b64(dir_path: Path) -> tuple[str, str]:
            buf = _BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                for p in sorted(dir_path.rglob("*")):
                    if not p.is_file():
                        continue
                    arc = p.relative_to(dir_path)
                    tf.add(str(p), arcname=str(arc))
            raw = buf.getvalue()
            sha = _hashlib.sha256(raw).hexdigest()
            return _base64.b64encode(raw).decode("ascii"), sha

        cpt_b64, cpt_sha = tar_dir_contents_to_b64(cpt_dir)
        sft_b64, sft_sha = tar_dir_contents_to_b64(sft_dir)

        return {
            "cpt": {
                "name": cpt_adapter_dir_name,
                "tar_gz_base64": cpt_b64,
                "sha256": cpt_sha,
            },
            "sft": {
                "name": sft_adapter_dir_name,
                "tar_gz_base64": sft_b64,
                "sha256": sft_sha,
            },
        }

    async def _run():
        return await train_adapters()

    result = asyncio.run(_run())

    cpt = result["cpt"]
    sft = result["sft"]

    local_models_dir.mkdir(parents=True, exist_ok=True)

    cpt_local_dir = local_models_dir / cpt_adapter_dir_name
    sft_local_dir = local_models_dir / sft_adapter_dir_name

    print(f"Extracting CPT adapter -> {cpt_local_dir}", flush=True)
    _extract_adapter_tar_gz_base64(
        tar_gz_b64=cpt["tar_gz_base64"],
        dest_dir=cpt_local_dir,
        overwrite=overwrite_local,
    )
    print(f"Extracting SFT adapter -> {sft_local_dir}", flush=True)
    _extract_adapter_tar_gz_base64(
        tar_gz_b64=sft["tar_gz_base64"],
        dest_dir=sft_local_dir,
        overwrite=overwrite_local,
    )


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="RunPod: create GPU pod, upload bundle, train, download, terminate")
    p.add_argument("--mode", choices=("sft", "cpt"), default="sft")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--gpu-type-id", default="NVIDIA GeForce RTX 4090", help="RunPod gpuTypeIds value")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="COMMUNITY")
    p.add_argument("--container-disk-gb", type=int, default=80)
    p.add_argument(
        "--volume-gb",
        type=int,
        default=50,
        help="Pod Volume Disk size (GB) at /workspace; ignored if --network-volume-id is set",
    )
    p.add_argument(
        "--network-volume-id",
        default=None,
        help="RunPod Network Volume id → /workspace (persists across pods). Env: RUNPOD_NETWORK_VOLUME_ID. Replaces --volume-gb.",
    )
    p.add_argument("--ssh-key", type=Path, default=None)
    p.add_argument(
        "--ssh-via-gateway",
        action="store_true",
        help="Use user@ssh.runpod.io:22 (copy user from pod Connect tab); file transfer uses exec/tar, not SFTP",
    )
    p.add_argument(
        "--ssh-gateway-user",
        default=None,
        help="Full SSH username e.g. podid-64411868 (or set RUNPOD_SSH_GATEWAY_USER). With --ssh-gateway-id-suffix, omit pod id.",
    )
    p.add_argument(
        "--ssh-gateway-id-suffix",
        default=None,
        help="If set with --ssh-via-gateway, SSH user is {pod_id}-this_suffix (matches Connect tab after the hyphen).",
    )
    p.add_argument("--pod-name", default="designmodel-train")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output-name", default="runpod-sft-lora", help="Adapter folder name under models/")
    p.add_argument("--data-sft", type=Path, default=ROOT / "data" / "sft_design_qa.jsonl")
    p.add_argument("--data-cpt", type=Path, default=ROOT / "data" / "book_corpus.jsonl")
    p.add_argument("--preload-adapter", type=Path, default=None, help="Local LoRA dir to upload and pass to SFT")
    p.add_argument("--precision", default="bf16", help="SFT only; bf16 on CUDA is stable")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B", help="HF hub id for base weights on the pod")
    p.add_argument("--keep-pod", action="store_true", help="Do not delete pod on failure/success")
    p.add_argument("--dry-run", action="store_true", help="Only build bundle + print pod payload; no API calls")
    p.add_argument(
        "--flash",
        action="store_true",
        help="Use Runpod Flash to run CPT->SFT on GPU and extract adapter tarballs locally.",
    )
    p.add_argument("--flash-endpoint-name", default="designmodel-cpt-sft-adapters")
    p.add_argument("--flash-gpu-type", default="ANY", help="Example: NVIDIA_L40S or NVIDIA_GEFORCE_RTX_4090")
    p.add_argument("--flash-volume-name", default=None)
    p.add_argument("--flash-volume-id", default=None)
    p.add_argument("--flash-volume-datacenter", default="US_GA_2")
    p.add_argument("--flash-volume-size-gb", type=int, default=100)
    p.add_argument("--flash-overwrite", action="store_true", help="Overwrite local adapter dirs in models/")
    p.add_argument("--flash-cpt-steps", type=int, default=500)
    p.add_argument("--flash-sft-steps", type=int, default=450)
    p.add_argument("--flash-cpt-lr", type=float, default=2e-4)
    p.add_argument("--flash-sft-lr", type=float, default=3e-5)
    p.add_argument("--flash-cpt-max-length", type=int, default=512)
    p.add_argument("--flash-sft-max-length", type=int, default=512)
    p.add_argument("--flash-sft-grad-clip", type=float, default=0.5)
    p.add_argument("--flash-sft-precision", default="bf16", help="auto|fp32|fp16|bf16")
    p.add_argument("--flash-cpt-output-name", default="design-books-lora-made-by-gpu")
    p.add_argument("--flash-sft-output-name", default="design-books-and-qa-by-gpu")
    a = p.parse_args()

    project_root = a.project_root.resolve()
    data_files: list[Path] = []
    if a.mode == "sft":
        data_files = [a.data_sft.resolve()]
    else:
        data_files = [a.data_cpt.resolve()]

    if a.flash:
        # runpod-flash expects RUNPOD_API_KEY; you may currently have RUNPOD_TOKEN.
        if not os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_TOKEN"):
            os.environ["RUNPOD_API_KEY"] = os.environ["RUNPOD_TOKEN"]
        run_flash_train_cpt_then_sft_and_return_adapters(
            project_root=project_root,
            local_models_dir=project_root / "models",
            endpoint_name=a.flash_endpoint_name,
            gpu_type_name=a.flash_gpu_type,
            volume_name=a.flash_volume_name,
            volume_id=a.flash_volume_id,
            volume_datacenter_name=a.flash_volume_datacenter,
            volume_size_gb=a.flash_volume_size_gb,
            cpt_steps=a.flash_cpt_steps,
            sft_steps=a.flash_sft_steps,
            cpt_lr=a.flash_cpt_lr,
            sft_lr=a.flash_sft_lr,
            cpt_max_length=a.flash_cpt_max_length,
            sft_max_length=a.flash_sft_max_length,
            sft_grad_clip=a.flash_sft_grad_clip,
            sft_precision=a.flash_sft_precision,
            cpt_output_name=a.flash_cpt_output_name,
            sft_output_name=a.flash_sft_output_name,
            overwrite_local=bool(a.flash_overwrite),
        )
        return

    key_path = _ssh_key_path(a.ssh_key)
    gw_user = (a.ssh_gateway_user or os.environ.get("RUNPOD_SSH_GATEWAY_USER") or "").strip()
    if a.ssh_via_gateway:
        if a.ssh_gateway_id_suffix:
            gw_user = ""  # filled after pod_id known
        elif not gw_user:
            print(
                "With --ssh-via-gateway, set --ssh-gateway-user or RUNPOD_SSH_GATEWAY_USER "
                "or pass --ssh-gateway-id-suffix (see RunPod Connect tab).",
                file=sys.stderr,
            )
            raise SystemExit(1)

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bundle.tar.gz"
        build_tarball(
            project_root=project_root,
            data_files=data_files,
            preload_adapter=a.preload_adapter,
            out_path=tar_path,
        )

        if a.dry_run:
            print(f"Bundle: {tar_path} ({tar_path.stat().st_size} bytes)")
            print("Dry run — not creating pod.")
            return

        pod_env: dict[str, str] = {}
        hf = os.environ.get("HF_TOKEN", "").strip()
        if hf:
            pod_env["HF_TOKEN"] = hf

        nv_id = (a.network_volume_id or os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or "").strip() or None
        pod = create_pod(
            name=a.pod_name,
            gpu_type_id=a.gpu_type_id,
            image=a.image,
            cloud_type=a.cloud_type,
            container_disk_gb=a.container_disk_gb,
            volume_gb=a.volume_gb,
            env=pod_env or None,
            network_volume_id=nv_id,
        )
        pod_id = pod.get("id")
        if not pod_id:
            print(json.dumps(pod, indent=2))
            raise RuntimeError("No pod id in API response")
        print(f"Created pod {pod_id}", flush=True)

        try:
            use_sftp = not a.ssh_via_gateway
            ssh_user = "root"
            if a.ssh_via_gateway:
                if a.ssh_gateway_id_suffix:
                    gw_user = f"{pod_id}-{a.ssh_gateway_id_suffix.strip()}"
                elif gw_user and "{pod_id}" in gw_user:
                    gw_user = gw_user.replace("{pod_id}", pod_id)
                ssh_user = gw_user
                print("Waiting for pod RUNNING (SSH gateway)…", flush=True)
                wait_for_pod_running(pod_id)
                host, ssh_port = RUNPOD_SSH_GATEWAY_HOST, RUNPOD_SSH_GATEWAY_PORT
                print(f"SSH {ssh_user}@{host}:{ssh_port}", flush=True)
                time.sleep(15)
            else:
                print("Waiting for SSH…", flush=True)
                host, ssh_port = wait_for_ssh(pod_id)
                print(f"SSH root@{host}:{ssh_port}", flush=True)
                time.sleep(10)

            print("Uploading bundle…", flush=True)
            sftp_put_file(
                host,
                ssh_port,
                key_path,
                tar_path,
                "/workspace/bundle.tar.gz",
                username=ssh_user,
                use_sftp=use_sftp,
            )

            unpack = """set -e
mkdir -p /workspace/DesignModel
tar -xzf /workspace/bundle.tar.gz -C /workspace
rm -f /workspace/bundle.tar.gz
"""
            code, out, err = ssh_run(
                host, ssh_port, key_path, unpack, username=ssh_user
            )
            if code != 0:
                print(err, file=sys.stderr)
                raise RuntimeError(f"Unpack failed: {code}")

            if a.mode == "sft":
                remote_train_sft(
                    host,
                    ssh_port,
                    key_path,
                    ssh_username=ssh_user,
                    output_name=a.output_name,
                    steps=a.steps,
                    preload=a.preload_adapter is not None,
                    precision=a.precision,
                    base_model=a.base_model,
                    preload_remote_path=None,
                    max_length=None,
                    lr=None,
                    grad_clip=None,
                )
            else:
                remote_train_cpt(
                    host,
                    ssh_port,
                    key_path,
                    ssh_username=ssh_user,
                    data_name=data_files[0].name,
                    output_name=a.output_name,
                    steps=a.steps,
                    base_model=a.base_model,
                )

            local_out = project_root / "models" / a.output_name
            local_out.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {a.output_name} → {local_out} …", flush=True)
            sftp_get_dir(
                host,
                ssh_port,
                key_path,
                f"/workspace/DesignModel/models/{a.output_name}",
                local_out,
                username=ssh_user,
                use_sftp=use_sftp,
            )
            print(f"Saved adapter to {local_out}")
        finally:
            if not a.keep_pod:
                print(f"Terminating pod {pod_id}…", flush=True)
                delete_pod(pod_id)
            else:
                print(f"Keeping pod {pod_id} (--keep-pod). Delete it in the RunPod UI when done.")


if __name__ == "__main__":
    main()
