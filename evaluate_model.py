#!/usr/bin/env python3
"""
Compare two local models on fixed design questions; judge via OpenRouter (larger model).
Set OPEN_ROUTER_TOKEN in .env (see .env.example).

Eval suites (see EVAL_SUITES / --eval-type):
  short-answers — definitions, principles, short UX questions
  apply-design-knowledge — product scenarios; judge favors applied recommendations (what/why/risks)
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent

DEFAULT_JUDGE = os.environ.get("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Short, knowledge-style prompts (definitions, principles, comparisons).
SHORT_ANSWER_QUESTIONS = [
    "In two or three sentences, what is affordance in interaction design and why does it matter?",
    "How would you reduce cognitive load on a long checkout form?",
    "What is the difference between recognition and recall in UI design?",
    "Give one concrete way to improve discoverability of secondary actions without cluttering the primary path.",
    "When is it appropriate to break a strict grid layout for emphasis?",
]

# Scenario → apply concepts: product context + constraint; expect what to do, why, and what to watch for.
APPLY_DESIGN_KNOWLEDGE_QUESTIONS = [
    "Context: A B2B web app’s settings area has 40+ toggles on one long page; support tickets say people ‘can’t find’ options. Role: You’re the lead product designer. What do you change first, why, and what do you watch out for?",
    "Context: A mobile news app uses a hamburger for almost everything; engagement with secondary features is flat. Platform: iOS/Android. What’s your move, why, and what failure mode are you avoiding?",
    "Context: Checkout has 12 fields on one screen; cart abandonment spiked after a redesign. Constraint: legal needs most fields. How do you improve the experience without dropping required data?",
    "Context: A dashboard for ops teams shows 30 KPIs at once; experts want ‘everything visible’ but new hires are overwhelmed. How do you reconcile that tension with concrete UI moves?",
    "Context: An enterprise table view has 80 columns; power users export to Excel instead of using the product. What’s your approach to information density and progressive disclosure here?",
    "Context: Primary action and destructive action sit next to each other in a modal; misclicks happen. What do you change, why, and how do you validate it?",
]

# Registry: eval type id → question list (used by CLI and main menu).
EVAL_SUITES: dict[str, list[str]] = {
    "short-answers": SHORT_ANSWER_QUESTIONS,
    "apply-design-knowledge": APPLY_DESIGN_KNOWLEDGE_QUESTIONS,
}

EVAL_SUITE_CHOICES = tuple(EVAL_SUITES.keys())
DEFAULT_EVAL_TYPE = "short-answers"

# Backward compatibility (older name for the short-answer list).
DESIGN_QUESTIONS = SHORT_ANSWER_QUESTIONS


class ModelSpec(NamedTuple):
    """(folder name, PEFT adapter path or None, full model path or base id)."""

    name: str
    adapter_path: str | None
    base_model: str


def _load_env_plain(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ if not already set (no python-dotenv required)."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_env() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass
    # Always merge plain parse so a missing python-dotenv install still loads .env.
    _load_env_plain(env_path)


def _openrouter_token() -> str | None:
    for key in (
        "OPEN_ROUTER_TOKEN",
        "OPENROUTER_API_KEY",
        "OPENROUTER_TOKEN",
    ):
        raw = os.environ.get(key)
        if not raw:
            continue
        v = raw.strip().strip('"').strip("'")
        if v:
            return v
    return None


def _judge_prompt_short_answers(
    *,
    question: str,
    response_a_label: str,
    response_a: str,
    response_b_label: str,
    response_b: str,
) -> str:
    return f"""You compare two answers to a design / UX question.

Question:
{question}

{response_a_label}:
{response_a.strip()}

{response_b_label}:
{response_b.strip()}

Which answer is better for a practitioner (clarity, correctness, usefulness)? 
You must pick exactly one. No ties.
Reply in this exact format:
CHOICE: [write exactly "{response_a_label}" or "{response_b_label}"]
REASON: [2-4 sentences, no markdown headings]
"""


def _judge_prompt_apply_design(
    *,
    question: str,
    response_a_label: str,
    response_a: str,
    response_b_label: str,
    response_b: str,
) -> str:
    return f"""You compare two answers to a product design scenario. The prompt asks for applied judgment (what to do, why, risks), not textbook definitions.

Scenario / prompt:
{question}

{response_a_label}:
{response_a.strip()}

{response_b_label}:
{response_b.strip()}

Which answer is better for a practitioner shipping the product? Prefer the answer that:
- Applies concrete interaction / information-architecture thinking to the situation
- Says what to do and why, and names tradeoffs or failure modes
- Is actionable without being generic fluff

You must pick exactly one. No ties.
Reply in this exact format:
CHOICE: [write exactly "{response_a_label}" or "{response_b_label}"]
REASON: [2-4 sentences, no markdown headings]
"""


def _judge_pair(
    *,
    question: str,
    response_a_label: str,
    response_a: str,
    response_b_label: str,
    response_b: str,
    api_key: str,
    judge_model: str,
    eval_type: str,
) -> str:
    import requests

    if eval_type == "apply-design-knowledge":
        prompt = _judge_prompt_apply_design(
            question=question,
            response_a_label=response_a_label,
            response_a=response_a,
            response_b_label=response_b_label,
            response_b=response_b,
        )
    else:
        prompt = _judge_prompt_short_answers(
            question=question,
            response_a_label=response_a_label,
            response_a=response_a,
            response_b_label=response_b_label,
            response_b=response_b,
        )
    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/DesignModel",
            "X-Title": "DesignModel evaluation",
        },
        json={
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def _parse_choice(verdict: str) -> str | None:
    """Return MODEL_1, MODEL_2, or None if unclear."""
    m = re.search(r"CHOICE:\s*([^\n]+)", verdict, re.I)
    if not m:
        return None
    line = m.group(1).strip()
    has1 = bool(re.search(r"\bMODEL_1\b", line, re.I))
    has2 = bool(re.search(r"\bMODEL_2\b", line, re.I))
    if has1 and not has2:
        return "MODEL_1"
    if has2 and not has1:
        return "MODEL_2"
    return None


def run_evaluation(
    *,
    model_1: ModelSpec,
    model_2: ModelSpec,
    judge_model: str | None = None,
    eval_type: str = DEFAULT_EVAL_TYPE,
    max_new_tokens: int = 384,
    temperature: float = 0.6,
) -> None:
    _load_env()
    token = _openrouter_token()
    if not token:
        print(
            "Missing OPEN_ROUTER_TOKEN in environment or .env — add your OpenRouter key.",
            file=sys.stderr,
        )
        return

    judge_model = judge_model or DEFAULT_JUDGE

    if eval_type not in EVAL_SUITES:
        print(
            f"Unknown eval type {eval_type!r}. Choose one of: {', '.join(EVAL_SUITE_CHOICES)}",
            file=sys.stderr,
        )
        return

    questions = EVAL_SUITES[eval_type]

    from generate import clear_model_cache, generate_once

    n1, n2 = model_1.name, model_2.name
    print(f"\nJudge model (OpenRouter): {judge_model}")
    print(f"Eval suite: {eval_type}")
    print(f"Model 1: {n1}")
    print(f"Model 2: {n2}\n")

    wins_1 = 0
    wins_2 = 0
    total = len(questions)

    for i, q in enumerate(questions, start=1):
        print("=" * 60)
        print(f"Question {i}/{total}  —  score so far:  {n1} {wins_1}  ·  {n2} {wins_2}")
        print(q)
        print("-" * 60)

        print(f"\n[Generating {n1}…]", flush=True)
        clear_model_cache()
        text_1 = generate_once(
            q,
            max_new=max_new_tokens,
            temperature=temperature,
            checkpoint_path=model_1.adapter_path,
            base_model=model_1.base_model,
        )

        print(f"[Generating {n2}…]", flush=True)
        clear_model_cache()
        text_2 = generate_once(
            q,
            max_new=max_new_tokens,
            temperature=temperature,
            checkpoint_path=model_2.adapter_path,
            base_model=model_2.base_model,
        )

        print(f"\n--- {n1} ---\n", text_1[:2000], sep="")
        print(f"\n--- {n2} ---\n", text_2[:2000], sep="")

        print("\n[Judge via OpenRouter…]", flush=True)
        try:
            if random.random() < 0.5:
                a_lab, a_txt = "MODEL_1", text_1
                b_lab, b_txt = "MODEL_2", text_2
            else:
                a_lab, a_txt = "MODEL_2", text_2
                b_lab, b_txt = "MODEL_1", text_1
            verdict = _judge_pair(
                question=q,
                response_a_label=a_lab,
                response_a=a_txt,
                response_b_label=b_lab,
                response_b=b_txt,
                api_key=token,
                judge_model=judge_model,
                eval_type=eval_type,
            )
            print("\n--- Judge ---\n", verdict, "\n", sep="")

            picked = _parse_choice(verdict)
            if picked == "MODEL_1":
                wins_1 += 1
                print(f"→ Winner this round: {n1}  (progress: {n1} {wins_1} — {n2} {wins_2})\n")
            elif picked == "MODEL_2":
                wins_2 += 1
                print(f"→ Winner this round: {n2}  (progress: {n1} {wins_1} — {n2} {wins_2})\n")
            else:
                print(
                    f"→ Could not parse judge choice (skipped).  (progress: {n1} {wins_1} — {n2} {wins_2})\n",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[error] OpenRouter: {e}", file=sys.stderr)

    print("=" * 60)
    print("Final score")
    print(f"  {n1}: {wins_1}/{total}")
    print(f"  {n2}: {wins_2}/{total}")
    if wins_1 + wins_2 > 0:
        pct1 = 100.0 * wins_1 / (wins_1 + wins_2)
        pct2 = 100.0 * wins_2 / (wins_1 + wins_2)
        print(f"  (of decided rounds: {n1} {pct1:.0f}% · {n2} {pct2:.0f}%)")
    print("Evaluation finished.\n")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model1-dir", type=Path, help="Path to model 1 (adapter or full)")
    p.add_argument("--model2-dir", type=Path, help="Path to model 2 (adapter or full)")
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument(
        "--eval-type",
        choices=EVAL_SUITE_CHOICES,
        default=DEFAULT_EVAL_TYPE,
        help="Evaluation suite: short-answers (principles) or apply-design-knowledge (scenarios)",
    )
    a = p.parse_args()
    if not a.model1_dir or not a.model2_dir:
        print("Use --model1-dir and --model2-dir, or run from main.py → 3.", file=sys.stderr)
        raise SystemExit(1)
    from generate import BASE_MODEL

    def spec_from_dir(d: Path) -> ModelSpec:
        d = d.resolve()
        if (d / "adapter_config.json").is_file():
            return ModelSpec(name=d.name, adapter_path=str(d), base_model=BASE_MODEL)
        return ModelSpec(name=d.name, adapter_path=None, base_model=str(d))

    run_evaluation(
        model_1=spec_from_dir(a.model1_dir),
        model_2=spec_from_dir(a.model2_dir),
        judge_model=a.judge,
        eval_type=a.eval_type,
    )


if __name__ == "__main__":
    main()
