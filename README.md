# fine-tuned-design-qwen

LoRA fine-tuning experiments on **Qwen3-4B** for design-related language: continued pre-training (CPT) on design book text, then supervised fine-tuning (SFT) on JSONL chat (question–answer) data. This repo holds the training and inference scripts used on a MacBook (Apple Silicon / MPS); weights are **not** committed (see [Model weights](#model-weights)).

---

## What we did

### Day 1 — CPT on design books

Starting from **Qwen3-4B**, we ran **continued pre-training** in the sense of LoRA on plain text: **200 steps** over a corpus built from three design books (*The Creative Act*, *The Design of Everyday Things*, and a second *Creative Act* / Rick Rubin–related source used in the experiment).

**Observation:** In side-by-side prompts (e.g. affordances, cognitive load, recognition vs recall, discoverability, breaking the grid), the **design-books-lora** adapter sometimes shifted wording toward terminology and cadence that felt closer to “having read those books,” and sometimes looked similar to the base model. Reading about design in text form does not by itself produce “good designer” behavior; it mainly nudges style and vocabulary.

**Next intuition:** Expose the model to **Q&A-style** data so it learns conversational application, not only book-like prose. Longer term, an **RL** loop (generate UIs or specs → receive critique → update) is an interesting direction but out of scope for this stage.

### Day 2 — SFT on JSON Q&A

We converted book-derived content into **JSONL** chat rows (`messages`: user then assistant) and ran **LoRA SFT** with loss only on assistant tokens (`sft_on_design_qa.py`).

**Hardware:** Training on a MacBook hit practical limits (memory, throughput). The run used **1566** usable examples after skipping malformed or over-length lines; **53** lines were skipped.

**Training log (illustrative):** Loss moved in the ~2–3 range for a while, then **rose sharply after roughly step 250** (into single-digit and then double-digit territory). That pattern usually means **overfitting**, **learning rate too high for the setup**, or **instability on MPS/fp16**—or a combination. The saved adapter at the end of a long unstable run can **degenerate** at inference (garbled tokens), which matches seeing nonsense output when loading `design-sft-lora` after training too long without early stopping or validation.

**Mitigations to try:** Stop earlier when validation loss or spot-check quality degrades; lower `--lr` (e.g. `1e-5`); fewer `--steps`; `--max-length` 128 on MPS; `--precision fp32` on MPS if bf16/fp16 still NaNs (RAM-heavy); or train on **CUDA** with bf16. The script also aborts if too many non-finite losses occur in a row.

---

## Repository layout

| Path | Role |
|------|------|
| `generate.py` | Load Qwen3-4B (local `models/Qwen3-4B` or Hub), optional PEFT adapter, streaming chat generation. |
| `main.py` | Interactive menu: chat, CPT, SFT. |
| `cpt_train.py` | LoRA CPT on `{"text": "..."}` JSONL lines. |
| `sft_on_design_qa.py` | LoRA SFT on `{"messages":[...]}` JSONL (one user + one assistant turn per line). |
| `extract_data_from_books.py` | PDFs → chunked `book_corpus.jsonl` for CPT. |
| `data/sft_design_qa.jsonl` | Example SFT dataset (chat JSONL). |
| `turn-data-into-question-answer-format.py` | Helper used when building Q&A JSONL from raw text. |
| `evaluate_model.py` | Evaluation helpers. |

---

## Requirements

- Python 3.9+ recommended  
- PyTorch with **MPS** (Mac) or **CUDA** (Linux/NVIDIA) when available  
- Hugging Face account/token if you pull `Qwen/Qwen3-4B` from the Hub  

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Model weights

The project expects a local base checkpoint at `models/Qwen3-4B` (see `generate.py`). If that folder is missing, the code falls back to **`Qwen/Qwen3-4B`** from the Hub.

**This repo’s `.gitignore` excludes `models/`** so clones stay small. After cloning, either:

- Download weights into `models/Qwen3-4B`, or  
- Rely on the Hub (first run will download).  

Trained adapters (e.g. `models/design-books-lora`, `models/design-sft-lora`) are also under `models/` and must be produced locally or shared separately (e.g. release artifact, private storage).

---

## Data: books → CPT corpus

1. Place PDFs under `books-to-fine-tune/` (you are responsible for **copyright/licensing**; do not publish PDFs you do not have rights to redistribute).  
2. Build JSONL:

```bash
python extract_data_from_books.py
```

By default this reads `books-to-fine-tune/` and writes `data/book_corpus.jsonl` (configurable via CLI; see `extract_data_from_books.py`).

CPT expects **one JSON object per line** with a `"text"` field (`cpt_train.py`).

---

## Training

### CPT (LoRA on book text)

Interactive:

```bash
python main.py
# Choose Train → CPT on design books
```

Or programmatically via `main.continue_pretrain_on_model` / `cpt_train.continue_pretrain_on_model` with `data/book_corpus.jsonl` and output under `models/design-books-lora` (defaults in `main.py`).

### SFT (LoRA on chat JSONL)

```bash
python sft_on_design_qa.py \
  --data data/sft_design_qa.jsonl \
  --name design-sft-lora \
  --steps 200 \
  --lr 1e-5 \
  --base models/Qwen3-4B
```

Adjust `--preload-adapter` to continue from an existing LoRA (e.g. CPT adapter). Use `--help` for all flags.

---

## Inference (chat)

```bash
python main.py
# Choose Chat → pick base or adapter
```

Or use `generate.py` directly for scripted generation.go