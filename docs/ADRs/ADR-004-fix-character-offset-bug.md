# ADR-004: Fix Character-Offset Bug in SOFC-Exp Parser

**Date:** 2026-08-06
**Status:** Accepted

---

## Context

After training Run 1 (10 epochs) and Run 2 (40 epochs), the NER model was still producing low F1 scores (0.035 and 0.086 respectively). When the `quick_facts_analysis.py` script was run on the structured facts output, the top "materials" extracted were mostly punctuation marks (`,`, `.`, `_`) and stop words (`the`, `of`).

## Root Cause

The SOFC-Exp text files (`.txt`) use **Windows-style line endings** (`\r\n`). Python's default `open()` call on Windows converts `\r\n` → `\n`, which silently removes one byte per line break. The annotation CSV files store **original byte-offset positions** from the source files.

As a result, for a document with N line breaks before a given position P, the sliced token was `text[P - N : end - N]` instead of `text[P : end]`. This caused the model to train on corrupted, mid-word or entirely misaligned tokens, making learning essentially impossible.

## Decision

Add `newline=""` to the `open()` call in `src/ner/prepare_data.py` when reading the text files. This prevents Python from translating `\r\n` → `\n`, preserving the original byte offsets that match the CSV annotations.

```python
# Before (buggy)
with open(txt_path, encoding="utf-8") as f:
    full_text = f.read()

# After (correct)
with open(txt_path, encoding="utf-8", newline="") as f:
    full_text = f.read()
```

## Verification

After the fix, the first parsed example from `PMC3564701.txt` correctly produces:
- Token: `gadolinium` → Tag: `B-MATERIAL` ✅
- Token: `yttria` → Tag: `B-MATERIAL` ✅
- Token: `YSZ` → Tag: `B-MATERIAL` ✅

Before the fix, these same positions produced `adolinium` (split mid-word).

## Consequences

- The `data/ner_processed` dataset **must be deleted and regenerated** before any subsequent training run.
- All F1 scores from Run 1 and Run 2 are **invalid** — they were measured on corrupted training data. They are kept in `docs/results/training_history.md` for transparency but should not be used in the final project report.
- Expected F1 improvement: significant. The model will now train on correctly aligned, clean token-tag pairs.
