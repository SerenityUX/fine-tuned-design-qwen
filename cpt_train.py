#!/usr/bin/env python3
"""
Continued pre-training style LoRA on plain text (like alpaca tokenize + train loop,
but causal LM loss on full chunks). Saves adapter checkpoint for use with generate.py.
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from generate import BASE_MODEL

IGNORE = -100


class JsonlTextDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = (obj.get("text") or "").strip()
                if len(t) > 20:
                    self.rows.append(t)
        self.tok = tokenizer
        self.max_length = max_length
        if not self.rows:
            raise ValueError(f"No text rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        text = self.rows[i]
        enc = self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attn == 0] = IGNORE
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def continue_pretrain_on_model(
    *,
    data_jsonl: Path,
    output_dir: Path,
    base_model: str = BASE_MODEL,
    steps: int = 200,
    batch_size: int = 1,
    lr: float = 2e-4,
    max_length: Optional[int] = None,
    grad_clip: float = 1.0,
    lora_r: int = 4,
    lora_alpha: int = 8,
    seed: int = 42,
) -> Path:
    """
    LoRA fine-tune on text JSONL ({"text": "..."} per line). Saves PEFT adapter to output_dir.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_jsonl = Path(data_jsonl).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if max_length is None:
        # 16 GB unified memory: long sequences OOM on MPS even with LoRA.
        max_length = 128 if device.type == "mps" else 512

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    print(f"Loading {base_model} for LoRA CPT (device={device}, seq_len={max_length})...", flush=True)
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

    # Fewer MLP LoRA layers on MPS saves a little memory; attention is enough for CPT.
    if device.type == "mps":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    ds = JsonlTextDataset(data_jsonl, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss_scalar = float(loss.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % 10 == 0 or step == 1:
            print(f"cpt step {step:5d} loss {loss_scalar:.4f}", flush=True)

        del out, loss, batch
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    print(f"Saving adapter to {output_dir}...", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
    return output_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="book_corpus.jsonl")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "design-books-lora",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokens per example; default 128 on MPS, 512 on CUDA (omit = auto)",
    )
    p.add_argument("--base", default=BASE_MODEL)
    a = p.parse_args()

    continue_pretrain_on_model(
        data_jsonl=a.data,
        output_dir=a.output,
        base_model=a.base,
        steps=a.steps,
        batch_size=a.batch_size,
        lr=a.lr,
        max_length=a.max_length if a.max_length is not None else None,
    )


if __name__ == "__main__":
    main()
