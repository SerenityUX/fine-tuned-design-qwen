#!/usr/bin/env python3
"""Qwen3-4B via Hugging Face Transformers — load base or local checkpoint, stream generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Optional

_ROOT = Path(__file__).resolve().parent
_LOCAL_QWEN = _ROOT / "models" / "Qwen3-4B"
# Prefer vendored weights so HF Hub isn’t hit every cold start (see hf download below).
BASE_MODEL = (
    str(_LOCAL_QWEN)
    if _LOCAL_QWEN.is_dir() and (_LOCAL_QWEN / "config.json").is_file()
    else "Qwen/Qwen3-4B"
)

_LOAD_CACHE: dict[tuple[str | None, str], tuple] = {}


def clear_model_cache() -> None:
    """Call after training so the next load picks up new adapter weights."""
    _LOAD_CACHE.clear()


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_tokenizer(
    checkpoint_path: Optional[str] = None,
    *,
    base_model: str = BASE_MODEL,
):
    """Load model + tokenizer. If checkpoint_path is a PEFT adapter dir, merge onto base_model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    cache_key = (checkpoint_path, base_model)
    if cache_key in _LOAD_CACHE:
        return _LOAD_CACHE[cache_key]

    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and ckpt.is_dir() and (ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        print(f"Loading base {base_model} + adapter {ckpt}...", flush=True)
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(ckpt), torch_dtype=dtype)
        model = model.to(device)
    elif ckpt and ckpt.is_dir() and (ckpt / "config.json").exists():
        print(f"Loading full model from {ckpt}...", flush=True)
        tok_path = ckpt if (ckpt / "tokenizer_config.json").exists() else Path(base_model)
        tok = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
    else:
        if checkpoint_path:
            print(f"Checkpoint {checkpoint_path!r} not found or invalid; using {base_model}.", flush=True)
        print(f"Loading {base_model}...", flush=True)
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()
    _LOAD_CACHE[cache_key] = (model, tok, device)
    return model, tok, device


def generate_chat_stream(
    messages: list[dict[str, str]],
    *,
    max_new: int,
    temperature: float,
    checkpoint_path: Optional[str] = None,
    base_model: str = BASE_MODEL,
    think: bool = False,
) -> Iterator[str]:
    """Multi-turn chat: `messages` are user/assistant turns (OpenAI-style roles)."""
    from transformers import TextIteratorStreamer

    model, tokenizer, device = load_model_tokenizer(checkpoint_path, base_model=base_model)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    top_p = 0.95 if think else 0.8
    temp = temperature

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new,
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
        "top_k": 20,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for piece in streamer:
        if piece:
            yield piece
    thread.join()


def generate_text_stream(
    *,
    prompt: str,
    max_new: int,
    temperature: float,
    checkpoint_path: Optional[str] = None,
    base_model: str = BASE_MODEL,
    think: bool = False,
) -> Iterator[str]:
    """
    Single user message → stream (dumbass-llm-style `generate_text_stream`).
    Uses enable_thinking=False unless think=True (Qwen3 docs).
    """
    return generate_chat_stream(
        [{"role": "user", "content": prompt}],
        max_new=max_new,
        temperature=temperature,
        checkpoint_path=checkpoint_path,
        base_model=base_model,
        think=think,
    )


def generate_once(
    prompt: str,
    *,
    max_new: int = 256,
    temperature: float = 0.7,
    checkpoint_path: Optional[str] = None,
    base_model: str = BASE_MODEL,
    think: bool = False,
) -> str:
    return "".join(
        generate_text_stream(
            prompt=prompt,
            max_new=max_new,
            temperature=temperature,
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            think=think,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-4B HF generation")
    p.add_argument("--model", default=BASE_MODEL, help="Hub id when no local checkpoint")
    p.add_argument("--checkpoint", default=None, help="PEFT folder or full saved model dir")
    p.add_argument("--prompt", default="Say hello in one short sentence.")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--think", action="store_true")
    a = p.parse_args()

    try:
        for chunk in generate_text_stream(
            prompt=a.prompt,
            max_new=a.max_new,
            temperature=a.temperature,
            checkpoint_path=a.checkpoint,
            base_model=a.model,
            think=a.think,
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
