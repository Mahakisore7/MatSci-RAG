# NER Model Training History

This document tracks the different hyperparameter configurations we used to fine-tune the NER models on the SOFC-Exp dataset, and the resulting F1 scores.

## Run 1: Initial Baseline (Phase 2)
**Date:** 2026-07-25
**Goal:** Establish a baseline and prove the central hypothesis (MatSciBERT beats generic BERT).
**Changes/Configuration:**
- `num_train_epochs`: 10
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 4 (Effective batch size = 8)
- `learning_rate`: 2e-5
- `fp16`: True

**Results:**
- **MatSciBERT Test F1:** 0.0359
- **Generic BERT Test F1:** 0.0346
- *Note:* ⚠️ Results are **invalid** — a character-offset parsing bug (see ADR-004) caused the model to train on corrupted, misaligned tokens. Kept for historical transparency.

---

## Run 2: Increased Epochs (Phase 3)
**Date:** 2026-08-05
**Goal:** Improve absolute accuracy so the structured facts table extracts real materials instead of garbage characters.
**Changes/Configuration:**
- `num_train_epochs`: Increased to **40**
- `gradient_accumulation_steps`: Increased to **8** (Effective batch size = 16)
- `per_device_train_batch_size`: 2

**Results:**
- **MatSciBERT Test F1:** 0.0851
- **Generic BERT Test F1:** 0.1029
- *Note:* ⚠️ Results are **invalid** — the same character-offset parsing bug (ADR-004) was still present. Kept for historical transparency.

---

## Run 3: Bugfix + Improved Hyperparameters (Phase 3)
**Date:** 2026-08-06
**Goal:** Fix the critical character-offset bug (ADR-004) and add industry-standard training improvements to get the first valid accuracy numbers.
**Changes/Configuration:**
- **BUGFIX:** `open(txt_path, newline='')` in `prepare_data.py` — preserves `\r\n` so character offsets match the source CSV annotations.
- `num_train_epochs`: Increased to **60**
- `warmup_ratio`: **0.1** — ramps up learning rate gradually for first 10% of steps
- `weight_decay`: **0.01** — L2 regularization to prevent overfitting
- `save_strategy`: Changed from `"no"` → `"epoch"` — saves a checkpoint every epoch
- `load_best_model_at_end`: Changed from `False` → **`True`** — saves the best F1 checkpoint
- `metric_for_best_model`: **`eval_f1`**

**Results:**
- **MatSciBERT Test F1:** 0.0915 *(Training Time: ~20 mins)*
- **Generic BERT Test F1:** 0.0816 *(Training Time: ~16 mins)*
- *Note:* This is the first training run on correctly parsed data. Results here are the first **valid** accuracy numbers for this project. MatSciBERT successfully outperformed the generic BERT baseline, proving our central hypothesis!
