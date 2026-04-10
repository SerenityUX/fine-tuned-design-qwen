#!/usr/bin/env python3
"""
Run CPT then SFT on a RunPod GPU (CUDA): stronger defaults than Mac (longer context, bf16 SFT).

Produces two local adapter folders under models/:
  - design-books-lora-made-by-gpu       (CPT on book corpus)
  - design-books-and-qa-by-gpu          (SFT on sft_design_qa.jsonl, continuing CPT LoRA)

Requires: RUNPOD_API_KEY or RUNPOD_TOKEN, SSH key in ~/.ssh (public key in RunPod).
Optional: HF_TOKEN in .env for Hugging Face downloads.

Usage:
  ./venv/bin/python runpod_gpu_pipeline.py

This cannot run on Apple Silicon training; it only orchestrates a remote CUDA pod.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from runpod_remote_train import (
    ROOT,
    RUNPOD_SSH_GATEWAY_HOST,
    RUNPOD_SSH_GATEWAY_PORT,
    build_tarball,
    create_pod,
    delete_pod,
    remote_train_cpt,
    remote_train_sft,
    sftp_get_dir,
    sftp_put_file,
    wait_for_pod_running,
    wait_for_ssh,
    _ssh_key_path,
)
import runpod_remote_train as rpt


def _resolve_corpus(project_root: Path) -> Path:
    for name in ("book_corpus.jsonl", "book_corpus_test.jsonl"):
        p = project_root / "data" / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No data/book_corpus.jsonl or data/book_corpus_test.jsonl under {project_root}"
    )


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description="GPU pipeline: CPT (book corpus) → SFT (design QA), download both adapters"
    )
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--gpu-type-id", default="NVIDIA GeForce RTX 4090")
    p.add_argument("--image", default=rpt.DEFAULT_IMAGE)
    p.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="COMMUNITY")
    p.add_argument("--container-disk-gb", type=int, default=100)
    p.add_argument(
        "--volume-gb",
        type=int,
        default=60,
        help="Pod Volume Disk (GB) at /workspace; not used when a Network Volume is attached",
    )
    p.add_argument(
        "--network-volume-id",
        default=None,
        help="Attach RunPod Network Volume at /workspace (data survives pod termination). Env: RUNPOD_NETWORK_VOLUME_ID",
    )
    p.add_argument("--ssh-key", type=Path, default=None)
    p.add_argument(
        "--ssh-via-gateway",
        action="store_true",
        help="Connect as user@ssh.runpod.io (from Connect tab); uploads use exec stream, not SFTP",
    )
    p.add_argument(
        "--ssh-gateway-user",
        default=None,
        help="Full gateway user e.g. podid-64411868, or use {pod_id}-suffix with literal {pod_id}",
    )
    p.add_argument(
        "--ssh-gateway-id-suffix",
        default=None,
        help="With --ssh-via-gateway: SSH user = {pod_id}-this value (stable across new pods if suffix is account-specific).",
    )
    p.add_argument("--pod-name", default="designmodel-gpu-pipeline")
    p.add_argument("--data-sft", type=Path, default=ROOT / "data" / "sft_design_qa.jsonl")
    p.add_argument("--cpt-name", default="design-books-lora-made-by-gpu")
    p.add_argument("--sft-name", default="design-books-and-qa-by-gpu")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B")
    p.add_argument("--cpt-steps", type=int, default=500, help="CPT steps (CUDA, seq 512)")
    p.add_argument("--sft-steps", type=int, default=450, help="SFT steps on top of CPT LoRA")
    p.add_argument("--cpt-max-length", type=int, default=512)
    p.add_argument("--sft-max-length", type=int, default=512)
    p.add_argument("--cpt-lr", type=float, default=2e-4)
    p.add_argument("--sft-lr", type=float, default=3e-5)
    p.add_argument("--sft-grad-clip", type=float, default=0.5)
    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    project_root = a.project_root.resolve()
    data_cpt = _resolve_corpus(project_root)
    data_sft = a.data_sft.resolve()
    if not data_sft.is_file():
        raise SystemExit(f"Missing SFT data: {data_sft}")

    key_path = _ssh_key_path(a.ssh_key)
    gw_user = (a.ssh_gateway_user or os.environ.get("RUNPOD_SSH_GATEWAY_USER") or "").strip()
    if a.ssh_via_gateway:
        if a.ssh_gateway_id_suffix:
            gw_user = ""
        elif not gw_user:
            raise SystemExit(
                "With --ssh-via-gateway, set --ssh-gateway-user or RUNPOD_SSH_GATEWAY_USER "
                "or --ssh-gateway-id-suffix (RunPod Connect tab shows user before @ssh.runpod.io)."
            )

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bundle.tar.gz"
        build_tarball(
            project_root=project_root,
            data_files=[data_cpt, data_sft],
            preload_adapter=None,
            out_path=tar_path,
        )

        if a.dry_run:
            print(f"Bundle: {tar_path} ({tar_path.stat().st_size} bytes)")
            print(f"CPT data: {data_cpt.name}  SFT data: {data_sft.name}")
            print("Dry run — no API calls.")
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

        remote_root = "/workspace/DesignModel"
        cpt_remote = f"{remote_root}/models/{a.cpt_name}"

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
                time.sleep(20)
            else:
                print("Waiting for SSH…", flush=True)
                host, ssh_port = wait_for_ssh(pod_id)
                print(f"SSH root@{host}:{ssh_port}", flush=True)
                time.sleep(20)

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
            code, out, err = rpt.ssh_run(
                host, ssh_port, key_path, unpack, username=ssh_user
            )
            if code != 0:
                print(err, file=sys.stderr)
                raise RuntimeError(f"Unpack failed: {code}")

            print("\n=== Stage 1: CPT (book corpus) →", a.cpt_name, "===\n", flush=True)
            remote_train_cpt(
                host,
                ssh_port,
                key_path,
                ssh_username=ssh_user,
                data_name=data_cpt.name,
                output_name=a.cpt_name,
                steps=a.cpt_steps,
                base_model=a.base_model,
                max_length=a.cpt_max_length,
                lr=a.cpt_lr,
                batch_size=1,
            )

            print("\n=== Stage 2: SFT (design QA) on CPT adapter →", a.sft_name, "===\n", flush=True)
            remote_train_sft(
                host,
                ssh_port,
                key_path,
                ssh_username=ssh_user,
                output_name=a.sft_name,
                steps=a.sft_steps,
                preload=False,
                precision="bf16",
                base_model=a.base_model,
                preload_remote_path=cpt_remote,
                max_length=a.sft_max_length,
                lr=a.sft_lr,
                grad_clip=a.sft_grad_clip,
            )

            for folder, label in (
                (a.cpt_name, "CPT adapter"),
                (a.sft_name, "CPT+SFT adapter"),
            ):
                local_out = project_root / "models" / folder
                local_out.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {label} → {local_out} …", flush=True)
                sftp_get_dir(
                    host,
                    ssh_port,
                    key_path,
                    f"{remote_root}/models/{folder}",
                    local_out,
                    username=ssh_user,
                    use_sftp=use_sftp,
                )
                print(f"Saved: {local_out}")
        finally:
            if not a.keep_pod:
                print(f"Terminating pod {pod_id}…", flush=True)
                delete_pod(pod_id)
            else:
                print(f"Keeping pod {pod_id} (--keep-pod).")


if __name__ == "__main__":
    main()
