#!/usr/bin/env python3
"""
Read PDFs from books-to-fine-tune/, segment by page windows, call OpenRouter to
produce scenario→concept SFT rows (OpenAI-style messages JSONL).

Requires OPEN_ROUTER_TOKEN in the environment (.env supported via python-dotenv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "books-to-fine-tune"
DEFAULT_OUTPUT = ROOT / "data" / "sft_design_qa.jsonl"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Emphasize scenario → apply concepts (short product context → what / why / watch for).
SYSTEM_PROMPT = """You are building supervised fine-tuning data for a UX and interaction design assistant.

Given an EXCERPT from a design book (only this text may be used), output training pairs where:
- The user message must read like a real chat: a SHORT product context (who, platform, constraint) plus a clear ask.
- The assistant reply must explain what to do, WHY it matters, and what to watch for or avoid, using concepts and vocabulary grounded in the excerpt.
- Prefer concrete scenarios over definitions or trivia.
- If the excerpt is useless (table of contents only, copyright page, index, mostly dots/numbers), return an empty pairs list.
- Do not cite invented page numbers or sources outside the excerpt. Do not mention that you are given an excerpt.

Respond with JSON only, no markdown fences."""

USER_TEMPLATE = """BOOK EXCERPT (pages {page_start}–{page_end} of {source}):
---
{excerpt}
---

Return a JSON object with this exact shape:
{{
  "pairs": [
    {{
      "user_message": "Natural user message: brief product context + question or request for guidance.",
      "assistant_reply": "Helpful answer grounded ONLY in the excerpt above."
    }}
  ]
}}

Produce between 1 and {max_pairs} pairs. Each pair must follow the scenario→apply-concepts pattern. If the excerpt is not substantive, use: {{"pairs": []}}"""


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_boilerplate(text: str) -> bool:
    t = _normalize_ws(text)
    if len(t) < 60:
        return True
    lower = t.lower()
    if lower.count(".") > 40 and len(t) < 400:
        return True
    if re.match(r"^table of contents\b", lower) and ".." in t:
        return True
    if "all rights reserved" in lower and len(t) < 500:
        return True
    if "isbn" in lower and "copyright" in lower and len(t) < 800:
        return True
    return False


def extract_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        raw = page.extract_text()
        pages.append(raw if raw else "")
    return pages


def iter_page_windows(
    pages: list[str],
    *,
    pages_per_window: int,
    min_chars: int,
) -> list[tuple[int, int, str]]:
    """Yield (page_start_1based, page_end_1based, joined text) for non-boilerplate windows."""
    out: list[tuple[int, int, str]] = []
    n = len(pieces := pages)
    i = 0
    while i < n:
        chunk_parts: list[str] = []
        j = i
        while j < n and len(chunk_parts) < pages_per_window:
            chunk_parts.append(pieces[j])
            j += 1
        combined = "\n\n".join(p for p in chunk_parts if p)
        combined_norm = _normalize_ws(combined)
        start = i + 1
        end = j
        i = j
        if len(combined_norm) < min_chars:
            continue
        if _looks_like_boilerplate(combined_norm):
            continue
        out.append((start, end, combined_norm))
    return out


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def call_openrouter(
    *,
    token: str,
    model: str,
    user_content: str,
    timeout: int = 120,
) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.4,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response: {data!r}") from e


def parse_pairs_json(content: str) -> list[dict[str, str]]:
    text = _strip_json_fences(content)
    obj = json.loads(text)
    raw_pairs = obj.get("pairs")
    if not isinstance(raw_pairs, list):
        return []
    out: list[dict[str, str]] = []
    for p in raw_pairs:
        if not isinstance(p, dict):
            continue
        um = (p.get("user_message") or "").strip()
        ar = (p.get("assistant_reply") or "").strip()
        if len(um) < 20 or len(ar) < 40:
            continue
        out.append({"user_message": um, "assistant_reply": ar})
    return out


def _window_records(
    pdf_name: str,
    page_start: int,
    page_end: int,
    pairs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": pair["user_message"]},
                    {"role": "assistant", "content": pair["assistant_reply"]},
                ],
                "source": pdf_name,
                "page_start": page_start,
                "page_end": page_end,
            }
        )
    return rows


def fetch_pairs_for_window(
    *,
    token: str,
    model: str,
    retries: int,
    pdf_name: str,
    page_start: int,
    page_end: int,
    user_content: str,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    Call OpenRouter and return JSONL-ready rows, or (empty, error_message).
    """
    content: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            content = call_openrouter(token=token, model=model, user_content=user_content)
            break
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2**attempt)
    if content is None:
        return [], f"pages {page_start}-{page_end}: {last_err}"
    try:
        pairs = parse_pairs_json(content)
    except json.JSONDecodeError as e:
        return [], f"pages {page_start}-{page_end}: JSON parse ({e})"
    return _window_records(pdf_name, page_start, page_end, pairs), None


def run(
    *,
    input_dir: Path,
    output_path: Path,
    glob_pat: str,
    pages_per_window: int,
    min_chars: int,
    max_pairs: int,
    max_windows: Optional[int],
    model: str,
    dry_run: bool,
    sleep_s: float,
    retries: int,
    append: bool,
    workers: int,
) -> int:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("OPEN_ROUTER_TOKEN", "").strip()
    if not token and not dry_run:
        print("Set OPEN_ROUTER_TOKEN in the environment or .env", file=sys.stderr)
        raise SystemExit(1)

    input_dir = input_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(input_dir.glob(glob_pat))
    if not pdfs:
        print(f"No files matching {glob_pat!r} in {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    n_written = 0
    mode = "a" if append else "w"

    # pdf_name, page_start, page_end, excerpt, user_content
    Job = tuple[str, int, int, str, str]

    jobs: list[Job] = []
    for pdf in pdfs:
        print(f"Scanning {pdf.name}...", flush=True)
        try:
            pages = extract_pdf_pages(pdf)
        except Exception as e:
            print(f"  skip ({e})", file=sys.stderr)
            continue

        windows = iter_page_windows(
            pages,
            pages_per_window=pages_per_window,
            min_chars=min_chars,
        )
        for page_start, page_end, excerpt in windows:
            if max_windows is not None and len(jobs) >= max_windows:
                break
            user_content = USER_TEMPLATE.format(
                page_start=page_start,
                page_end=page_end,
                source=pdf.name,
                excerpt=excerpt,
                max_pairs=max_pairs,
            )
            jobs.append((pdf.name, page_start, page_end, excerpt, user_content))
        if max_windows is not None and len(jobs) >= max_windows:
            break

    if max_windows is not None and len(jobs) >= max_windows:
        print(f"Capped at --max-windows {max_windows} ({len(jobs)} jobs).", flush=True)

    if dry_run:
        for pdf_name, page_start, page_end, excerpt, _ in jobs:
            print(f"  [dry-run] {pdf_name} pages {page_start}-{page_end} ({len(excerpt)} chars)", flush=True)
        print(f"Dry run: {len(jobs)} windows. Wrote 0 JSONL rows.")
        return 0

    write_lock = threading.Lock()

    def _one(job: Job) -> tuple[list[dict[str, Any]], Optional[str], str, int, int]:
        pdf_name, page_start, page_end, _excerpt, user_content = job
        rows, err = fetch_pairs_for_window(
            token=token,
            model=model,
            retries=retries,
            pdf_name=pdf_name,
            page_start=page_start,
            page_end=page_end,
            user_content=user_content,
        )
        return rows, err, pdf_name, page_start, page_end

    with output_path.open(mode, encoding="utf-8") as out:
        if workers <= 1:
            for job in jobs:
                rows, err, pdf_name, page_start, page_end = _one(job)
                if err:
                    print(f"  {pdf_name} {err}", file=sys.stderr)
                    continue
                for rec in rows:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += len(rows)
                print(
                    f"  {pdf_name} pages {page_start}-{page_end}: {len(rows)} row(s) (total {n_written})",
                    flush=True,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
        else:
            if sleep_s > 0:
                print(
                    "Note: --sleep is ignored when --workers > 1; reduce --workers if you hit rate limits.",
                    file=sys.stderr,
                )
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_one, job) for job in jobs]
                for fut in as_completed(futures):
                    rows, err, pdf_name, page_start, page_end = fut.result()
                    done += 1
                    if err:
                        print(f"  {pdf_name} {err}", file=sys.stderr)
                        continue
                    with write_lock:
                        for rec in rows:
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += len(rows)
                    print(
                        f"  [{done}/{len(jobs)}] {pdf_name} pages {page_start}-{page_end}: "
                        f"{len(rows)} row(s) (total {n_written})",
                        flush=True,
                    )

    print(f"Done. Wrote {n_written} JSONL rows to {output_path}")
    return n_written


def main() -> None:
    p = argparse.ArgumentParser(description="PDFs → scenario SFT JSONL via OpenRouter")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--glob", default="*.pdf")
    p.add_argument("--pages-per-window", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-pairs", type=int, default=3, help="Max pairs per window in the prompt")
    p.add_argument("--max-windows", type=int, default=None, help="Stop after this many API windows (across all PDFs)")
    p.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
        help="OpenRouter model id (default: google/gemini-2.5-flash-lite or OPENROUTER_MODEL)",
    )
    p.add_argument("--dry-run", action="store_true", help="List windows only; no API calls")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds between successful API calls (only when --workers 1)",
    )
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel OpenRouter requests (threads). Use 1 for sequential + --sleep throttling.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to output JSONL instead of overwriting",
    )
    a = p.parse_args()

    try:
        run(
            input_dir=a.input_dir,
            output_path=a.output,
            glob_pat=a.glob,
            pages_per_window=a.pages_per_window,
            min_chars=a.min_chars,
            max_pairs=a.max_pairs,
            max_windows=a.max_windows,
            model=a.model,
            dry_run=a.dry_run,
            sleep_s=a.sleep,
            retries=a.retries,
            append=a.append,
            workers=a.workers,
        )
    except SystemExit:
        raise


if __name__ == "__main__":
    main()
