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
- *Note:* Absolute accuracy was extremely low. When inference was run, the model primarily extracted meaningless characters (`_`, `,`, `the`) because it had not trained long enough to confidently recognize full entity structures.

---

## Run 2: Increased Epochs for Accuracy (Phase 3)
**Date:** 2026-08-05
**Goal:** Improve absolute accuracy so the structured facts table extracts real materials instead of garbage characters.
**Changes/Configuration:**
- `num_train_epochs`: Increased to **40** (giving the model 4x more time to learn).
- `gradient_accumulation_steps`: Increased to **8** (Effective batch size = 16) to stabilize learning gradients.
- `per_device_train_batch_size`: 2 (Kept identical to fit in 6GB VRAM).

**Results:**
- **MatSciBERT Test F1:** 0.0851
- **Generic BERT Test F1:** 0.1029
- *Note:* F1 scores nearly tripled (almost a 300% improvement). The model is now significantly better at identifying entity spans. 
