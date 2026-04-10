#!/usr/bin/env python3
"""
LoRA supervised fine-tuning on chat JSONL (messages: user/assistant per line).
Trains loss only on assistant tokens. Compatible with data/sft_design_qa.jsonl.

Use with base model only, or continue from an existing PEFT adapter (stack on same base).

Defaults favor stability (lower LR, gentler LoRA scale on MPS/fp16) to reduce NaN loss;
training stops immediately if loss becomes non-finite.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from generate import BASE_MODEL

IGNORE = -100
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sft_design_qa.jsonl"


def _apply_chat_template_safe(tokenizer, messages: list[dict[str, str]], **kwargs: Any):
    """apply_chat_template; Qwen3 may accept enable_thinking=False."""
    for extra in (
        {"enable_thinking": False},
        {},
    ):
        try:
            return tokenizer.apply_chat_template(messages, **kwargs, **extra)
        except TypeError:
            continue
    raise RuntimeError("apply_chat_template failed")


def _normalize_messages(obj: dict[str, Any]) -> Optional[List[dict[str, str]]]:
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    out: List[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = (m.get("role") or "").strip()
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    for i in range(len(out) - 1):
        if out[i]["role"] == "user" and out[i + 1]["role"] == "assistant":
            return [out[i], out[i + 1]]
    return None


def _token_lists(tokenizer, messages: List[dict[str, str]]) -> Tuple[List[int], List[int]]:
    """Prompt tokens (through assistant start) and full conversation tokens."""
    user_only = [messages[0]]
    prompt_ids = _apply_chat_template_safe(
        tokenizer,
        user_only,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template_safe(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.squeeze().tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.squeeze().tolist()
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    return prompt_ids, full_ids


def _align_prefix(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Return prompt length so full_ids[:plen] == prompt_ids (handles optional BOS quirks)."""
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    # Rare mismatch: find longest common prefix
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    if i < max(len(prompt_ids) // 2, 10):
        raise ValueError("Could not align prompt and full tokenization")
    return i


class JsonlChatDataset(Dataset):
    """Single-turn user→assistant rows; labels mask user/prompt, train on assistant only."""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.rows: List[List[dict[str, str]]] = []
        skipped = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = _normalize_messages(obj)
                if msgs is None:
                    skipped += 1
                    continue
                try:
                    prompt_ids, full_ids = _token_lists(tokenizer, msgs)
                    plen = _align_prefix(prompt_ids, full_ids)
                except Exception:
                    skipped += 1
                    continue
                if len(full_ids) > max_length:
                    skipped += 1
                    continue
                if plen >= len(full_ids):
                    skipped += 1
                    continue
                self.rows.append(msgs)
        self.skipped = skipped
        if not self.rows:
            raise ValueError(f"No valid chat rows in {path} (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        messages = self.rows[i]
        prompt_ids, full_ids = _token_lists(self.tok, messages)
        plen = _align_prefix(prompt_ids, full_ids)
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id

        seq = full_ids[: self.max_length]
        slen = len(seq)
        input_ids = seq + [pad_id] * (self.max_length - slen)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.zeros(self.max_length, dtype=torch.long)
        attn[:slen] = 1

        labels = torch.full((self.max_length,), IGNORE, dtype=torch.long)
        plen_c = min(plen, slen)
        if plen_c < slen:
            labels[plen_c:slen] = input_ids[plen_c:slen]

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _resolve_training_dtype(device: torch.device, precision: str) -> torch.dtype:
    """MPS: prefer bf16 over fp16 (fewer NaNs); CUDA: bf16; CPU: fp32."""
    p = (precision or "auto").strip().lower()
    if p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    if p not in ("auto", ""):
        raise ValueError(f"precision must be auto|fp32|fp16|bf16, got {precision!r}")
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def _lora_targets(device_type: str) -> list[str]:
    if device_type == "mps":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_trainable_model(
    *,
    base_model: str,
    preload_adapter: Optional[Path],
    device: torch.device,
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
):
    """Base checkpoint + optional existing LoRA; or new LoRA on base/full weights."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    targets = _lora_targets(device.type)

    if preload_adapter is not None and preload_adapter.is_dir() and (preload_adapter / "adapter_config.json").is_file():
        print(f"Loading existing LoRA from {preload_adapter}…", flush=True)
        model = PeftModel.from_pretrained(model, str(preload_adapter), torch_dtype=dtype, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, tokenizer


def run_sft_on_design_qa(
    *,
    data_jsonl: Path,
    output_dir: Path,
    base_model: str = BASE_MODEL,
    preload_adapter: Optional[Path] = None,
    steps: int = 300,
    batch_size: int = 1,
    lr: float = 5e-5,
    max_length: Optional[int] = None,
    grad_clip: float = 0.5,
    lora_r: int = 8,
    lora_alpha: int = 8,
    seed: int = 42,
    precision: str = "auto",
) -> Path:
    """
    LoRA SFT on JSONL with {"messages": [{"role":"user",...},{"role":"assistant",...}], ...}.
    Saves adapter + tokenizer to output_dir.
    """
    data_jsonl = Path(data_jsonl).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if max_length is None:
        # Shorter context on MPS reduces activation overflow (fewer NaNs).
        max_length = 192 if device.type == "mps" else 512

    dtype = _resolve_training_dtype(device, precision)

    # Longer warmup on MPS helps avoid early blow-ups while LR ramps.
    if device.type == "mps":
        warmup_steps = max(1, min(120, steps // 3))
    else:
        warmup_steps = max(1, min(80, steps // 4))

    adam_eps = 1e-8 if dtype == torch.float32 else 1e-5

    print(
        f"SFT: base={base_model!r} preload_adapter={preload_adapter!r} "
        f"data={data_jsonl} out={output_dir} device={device} seq_len={max_length} "
        f"dtype={dtype} precision={precision!r} lr={lr} warmup_steps={warmup_steps} "
        f"grad_clip={grad_clip}",
        flush=True,
    )

    model, tokenizer = load_trainable_model(
        base_model=base_model,
        preload_adapter=preload_adapter,
        device=device,
        dtype=dtype,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    ds = JsonlChatDataset(data_jsonl, tokenizer, max_length)
    if ds.skipped:
        print(f"Skipped {ds.skipped} lines (bad format or too long).", flush=True)
    print(f"Training on {len(ds)} examples.", flush=True)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=adam_eps)
    for g in opt.param_groups:
        g["lr"] = 0.0
    model.train()

    it = iter(loader)
    skipped_nan = 0
    consecutive_nan = 0
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        lr_scale = min(1.0, step / warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr * lr_scale

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss_scalar = float(loss.item())

        if not math.isfinite(loss_scalar):
            skipped_nan += 1
            consecutive_nan += 1
            print(
                f"  warning: non-finite loss at step {step}; skipping batch "
                f"(skipped {skipped_nan}, consecutive {consecutive_nan})",
                file=sys.stderr,
                flush=True,
            )
            del out, loss, batch
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()
            if consecutive_nan >= 12:
                raise RuntimeError(
                    "Too many consecutive non-finite losses on MPS. "
                    "Try: --precision fp32 (needs RAM), --lr 1e-5, --max-length 128, "
                    "or train on CUDA with bf16."
                )
            continue

        consecutive_nan = 0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % 10 == 0 or step == 1:
            print(f"sft step {step:5d} loss {loss_scalar:.4f} lr {lr * lr_scale:.2e}", flush=True)

        del out, loss, batch
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    if skipped_nan:
        print(f"Note: skipped {skipped_nan} batches with non-finite loss.", flush=True)

    print(f"Saving adapter to {output_dir}…", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
    return output_dir


def sanitize_output_name(name: str) -> str:
    s = name.strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    s = s.strip("-.")
    if not s or len(s) > 96:
        raise ValueError("Output name must be 1–96 chars of letters, digits, ._-")
    return s


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="LoRA SFT on design QA JSONL (messages format)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL with messages per line")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output adapter directory (default: models/<name> if --name set)",
    )
    p.add_argument("--name", type=str, default=None, help="Subfolder under models/ (sanitized)")
    p.add_argument("--base", type=str, default=BASE_MODEL, help="Base model id or local path")
    p.add_argument(
        "--preload-adapter",
        type=Path,
        default=None,
        help="Existing PEFT adapter to continue from (optional)",
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5, help="Default lowered for fp16/MPS stability")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="auto: bf16 on CUDA/MPS, fp32 on CPU. Use fp32 on MPS if bf16 still NaNs (high RAM).",
    )
    a = p.parse_args()

    out = a.output
    if out is None:
        if not a.name:
            print("Set --output or --name", file=__import__("sys").stderr)
            raise SystemExit(1)
        out = ROOT / "models" / sanitize_output_name(a.name)

    run_sft_on_design_qa(
        data_jsonl=a.data,
        output_dir=out,
        base_model=a.base,
        preload_adapter=a.preload_adapter,
        steps=a.steps,
        batch_size=a.batch_size,
        lr=a.lr,
        max_length=a.max_length,
        grad_clip=a.grad_clip,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        precision=a.precision,
    )


if __name__ == "__main__":
    main()
