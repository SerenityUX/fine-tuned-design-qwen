#!/usr/bin/env python3
"""Extract text from PDFs under books-to-fine-tune/ and write JSONL chunks for CPT/SFT."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "books-to-fine-tune"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "book_corpus.jsonl"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def extract_books(
    input_dir: Path,
    output_path: Path,
    *,
    chunk_size: int = 1800,
    overlap: int = 200,
    glob: str = "*.pdf",
) -> int:
    input_dir = input_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob(glob))
    if not pdfs:
        raise FileNotFoundError(f"No files matching {glob!r} in {input_dir}")

    n_lines = 0
    with output_path.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            print(f"Extracting {pdf.name}...", flush=True)
            try:
                raw = extract_pdf_text(pdf)
            except Exception as e:
                print(f"  skip ({e})", file=sys.stderr)
                continue
            for chunk in _chunk_text(raw, chunk_size, overlap):
                if len(chunk) < 50:
                    continue
                rec = {"text": chunk, "source": pdf.name}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1

    print(f"Wrote {n_lines} lines to {output_path}")
    return n_lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--chunk-size", type=int, default=1800)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--glob", default="*.pdf")
    a = p.parse_args()

    try:
        extract_books(
            a.input_dir,
            a.output,
            chunk_size=a.chunk_size,
            overlap=a.overlap,
            glob=a.glob,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
