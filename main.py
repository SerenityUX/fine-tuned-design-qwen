#!/usr/bin/env python3
"""
DesignModel — numeric menus: 1 Chat, 2 Train, 3 Evaluate (0 = quit / back where noted).
Train: CPT on book text, or SFT on data/sft_design_qa.jsonl via sft_on_design_qa.py.
Free text only inside chat prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_DATA = ROOT / "data" / "book_corpus.jsonl"
BOOKS_DIR = ROOT / "books-to-fine-tune"
DESIGN_BOOKS_ADAPTER = ROOT / "models" / "design-books-lora"
DEFAULT_SFT_DATA = ROOT / "data" / "sft_design_qa.jsonl"


def continue_pretrain_on_model(
    data_jsonl: Path | None = None,
    output_dir: Path | None = None,
    **kwargs,
) -> Path:
    from cpt_train import continue_pretrain_on_model as _run

    data_jsonl = Path(data_jsonl or _resolve_corpus_path())
    output_dir = Path(output_dir or DESIGN_BOOKS_ADAPTER)
    return _run(data_jsonl=data_jsonl, output_dir=output_dir, **kwargs)


def _resolve_corpus_path() -> Path:
    if DEFAULT_DATA.is_file():
        return DEFAULT_DATA
    alt = ROOT / "data" / "book_corpus_test.jsonl"
    if alt.is_file():
        return alt
    return DEFAULT_DATA


def discover_models() -> list[tuple[str, str | None, str]]:
    from generate import BASE_MODEL

    models_dir = ROOT / "models"
    out: list[tuple[str, str | None, str]] = []
    if not models_dir.is_dir():
        return out

    for p in sorted(models_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "adapter_config.json").is_file():
            out.append((p.name, str(p.resolve()), BASE_MODEL))
            continue
        if (p / "config.json").is_file():
            has_weights = (
                any(p.glob("*.safetensors"))
                or any(p.glob("*.bin"))
                or (p / "pytorch_model.bin").is_file()
            )
            if has_weights:
                out.append((p.name, None, str(p.resolve())))
    return out


def _print_model_menu(entries: list[tuple[str, str | None, str]]) -> None:
    for i, (name, _, _) in enumerate(entries, start=1):
        print(f"  {i} — {name}")


def run_chat_session(
    *,
    adapter_path: str | None,
    base_model: str,
) -> None:
    from generate import generate_chat_stream

    messages: list[dict[str, str]] = []
    print("\nChat — type your message; empty line or /exit to leave.")
    print("-" * 50)

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not line or line.lower() in ("/exit", "/quit", ":q", "exit", "quit"):
            print("Bye.")
            return

        messages.append({"role": "user", "content": line})
        print("\nAssistant:", end="", flush=True)
        try:
            chunks: list[str] = []
            for piece in generate_chat_stream(
                messages,
                # Hard cap on new tokens; 512 often truncates long markdown answers mid-sentence.
                max_new=2048,
                temperature=0.7,
                checkpoint_path=adapter_path,
                base_model=base_model,
                think=False,
            ):
                print(piece, end="", flush=True)
                chunks.append(piece)
            print()
            reply = "".join(chunks).strip()
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[error] {e}", file=sys.stderr)
            messages.pop()


def _input_choice(prompt: str, valid: set[str]) -> str:
    s = input(prompt).strip()
    return s


def _flow_chat() -> None:
    from generate import BASE_MODEL

    entries = discover_models()
    if not entries:
        print(
            f"No models in {ROOT / 'models'}. Add Qwen3-4B or train an adapter.\n"
        )
        return

    print("\nWhich model?")
    _print_model_menu(entries)
    print("  0 — back")
    raw = _input_choice("> ", set())
    if raw == "0":
        return
    if not raw.isdigit() or int(raw) < 1 or int(raw) > len(entries):
        print("Invalid.")
        return

    name, adapter, base = entries[int(raw) - 1]
    print(f"\nUsing: {name}\n")
    run_chat_session(adapter_path=adapter, base_model=base)


def _flow_train() -> None:
    print("\n  1 — CPT on design books (LoRA → models/design-books-lora)")
    print("  2 — SFT on design QA (sft_on_design_qa.py → pick base/adapter, name new folder)")
    print("  0 — back")
    raw = _input_choice("> ", set())
    if raw == "0":
        return
    if raw == "1":
        _flow_train_cpt()
    elif raw == "2":
        _flow_train_sft_design_qa()
    else:
        print("Invalid.")


def _flow_train_cpt() -> None:
    from extract_data_from_books import extract_books
    from generate import BASE_MODEL, clear_model_cache

    print("\n  1 — start training")
    print("  0 — cancel")
    go = _input_choice("> ", set())
    if go != "1":
        print("Cancelled.")
        return

    corpus = _resolve_corpus_path()
    if not corpus.is_file():
        print(f"No corpus at {corpus}. Extracting PDFs from {BOOKS_DIR}...")
        try:
            extract_books(BOOKS_DIR, DEFAULT_DATA)
        except Exception as e:
            print(f"[error] extract: {e}", file=sys.stderr)
            return
        corpus = _resolve_corpus_path()
        if not corpus.is_file():
            print("[error] Still no corpus JSONL.")
            return

    steps_in = input("Steps (default 200): ").strip()
    steps = int(steps_in) if steps_in.isdigit() else 200

    print("\nTraining...")
    try:
        continue_pretrain_on_model(
            data_jsonl=corpus,
            output_dir=DESIGN_BOOKS_ADAPTER,
            steps=steps,
        )
    except Exception as e:
        print(f"[error] training: {e}", file=sys.stderr)
        return

    clear_model_cache()
    print("\nDone. Opening chat with the new adapter.\n")
    run_chat_session(
        adapter_path=str(DESIGN_BOOKS_ADAPTER.resolve()),
        base_model=BASE_MODEL,
    )


def _flow_train_sft_design_qa() -> None:
    from generate import clear_model_cache

    from sft_on_design_qa import run_sft_on_design_qa, sanitize_output_name

    entries = discover_models()
    if not entries:
        print(f"No models under {ROOT / 'models'}. Add Qwen3-4B or train an adapter first.\n")
        return

    print("\nTrain on which checkpoint? (base model or adapter = continue that LoRA)")
    _print_model_menu(entries)
    print("  0 — back")
    raw = _input_choice("> ", set())
    if raw == "0":
        return
    if not raw.isdigit() or int(raw) < 1 or int(raw) > len(entries):
        print("Invalid.")
        return

    name, adapter, base = entries[int(raw) - 1]
    preload = Path(adapter).resolve() if adapter else None

    data_path = DEFAULT_SFT_DATA
    if not data_path.is_file():
        print(f"[error] No SFT data at {data_path}. Generate sft_design_qa.jsonl first.\n")
        return

    custom_data = input(f"JSONL path [{data_path}]: ").strip()
    if custom_data:
        data_path = Path(custom_data).expanduser().resolve()
        if not data_path.is_file():
            print(f"[error] Not found: {data_path}")
            return

    out_raw = input("Name for new adapter folder under models/ (default design-sft-lora): ").strip()
    out_name = out_raw or "design-sft-lora"
    try:
        safe = sanitize_output_name(out_name)
    except ValueError as e:
        print(f"[error] {e}")
        return

    out_dir = ROOT / "models" / safe
    if out_dir.exists() and any(out_dir.iterdir()):
        yn = input(f"{out_dir} exists and is not empty. Overwrite? [y/N]: ").strip().lower()
        if yn != "y":
            print("Cancelled.")
            return

    steps_in = input("Steps (default 300): ").strip()
    steps = int(steps_in) if steps_in.isdigit() else 300

    kind = "adapter (continue LoRA)" if preload else "base weights (new LoRA)"
    print(f"\nSFT: train on {name!r} ({kind}) → {out_dir.name}/  steps={steps}\n")
    try:
        run_sft_on_design_qa(
            data_jsonl=data_path,
            output_dir=out_dir,
            base_model=base,
            preload_adapter=preload,
            steps=steps,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return

    clear_model_cache()
    print("\nDone. Opening chat with the new adapter.\n")
    run_chat_session(
        adapter_path=str(out_dir.resolve()),
        base_model=base,
    )


def _flow_evaluate() -> None:
    from evaluate_model import ModelSpec, run_evaluation

    entries = discover_models()
    if len(entries) < 2:
        print(
            f"\nNeed at least two entries under {ROOT / 'models'} "
            "(e.g. base Qwen + a LoRA). Found "
            f"{len(entries)}.\n"
        )
        return

    print("\nPick model 1 (first competitor):")
    _print_model_menu(entries)
    print("  0 — back")
    r1 = _input_choice("> ", set())
    if r1 == "0":
        return
    if not r1.isdigit() or int(r1) < 1 or int(r1) > len(entries):
        print("Invalid.")
        return
    i1 = int(r1) - 1

    print("\nPick model 2 (second competitor):")
    _print_model_menu(entries)
    print("  0 — back")
    r2 = _input_choice("> ", set())
    if r2 == "0":
        return
    if not r2.isdigit() or int(r2) < 1 or int(r2) > len(entries):
        print("Invalid.")
        return
    i2 = int(r2) - 1
    if i1 == i2:
        print("Model 1 and model 2 must be different.")
        return

    n1, ad1, b1 = entries[i1]
    n2, ad2, b2 = entries[i2]
    m1 = ModelSpec(name=n1, adapter_path=ad1, base_model=b1)
    m2 = ModelSpec(name=n2, adapter_path=ad2, base_model=b2)

    print("\nEval type:")
    print("  1 — Short answers (definitions, principles)")
    print("  2 — Apply design knowledge (scenarios → recommendations)")
    print("  0 — back")
    et = _input_choice("> ", set())
    if et == "0":
        return
    if et == "1":
        eval_type = "short-answers"
    elif et == "2":
        eval_type = "apply-design-knowledge"
    else:
        print("Invalid.")
        return

    print("\nRunning evaluation (local inference + OpenRouter judge)…\n")
    try:
        run_evaluation(model_1=m1, model_2=m2, eval_type=eval_type)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)


def run_interactive_cli() -> None:
    print("DesignModel")
    while True:
        print("\n  1 — Chat")
        print("  2 — Train")
        print("  3 — Evaluate")
        print("  0 — quit")
        top = _input_choice("> ", set())
        if top == "0":
            print("Bye.")
            return
        if top == "1":
            _flow_chat()
        elif top == "2":
            _flow_train()
        elif top == "3":
            _flow_evaluate()
        else:
            print("Invalid. Enter 1, 2, 3, or 0.")


def main() -> None:
    run_interactive_cli()


if __name__ == "__main__":
    main()
