# ADR-001: Use SOFC-Exp Dataset for NER Training Instead of Manual Annotation

**Status:** Accepted  
**Date:** 2026-07-25  
**Author:** Mahakisore7

---

## Context

Phase 2 required a labeled Named Entity Recognition (NER) dataset to fine-tune MatSciBERT. Two options were considered:

1. **Manual annotation using Doccano:** Label our own HEA arXiv abstracts by hand.
2. **Use an existing open-source labeled dataset:** Use the Bosch Research SOFC-Exp corpus.

## Decision

We chose to use the **SOFC-Exp corpus** (`boschresearch/sofc-exp_textmining_resources`).

## Reasoning

- Manual annotation is extremely time-intensive. Labeling even 200 sentences to a quality standard that is good enough for training would take 15-20 hours of focused work.
- The SOFC-Exp corpus is professionally annotated by materials science domain experts over multiple years. It contains 45 full papers with 876 labeled sentences covering 4 entity types (MATERIAL, DEVICE, EXPERIMENT, VALUE).
- Using a well-known public benchmark makes our training results **reproducible and citable** — a reviewer can independently verify our MatSciBERT vs. BERT comparison.

## Consequences

**Positive:**
- Professional-quality labels with inter-annotator agreement.
- Reproducible benchmark results (F1 scores are directly comparable to published literature).

**Negative:**
- The trained model is tuned for **SOFC papers specifically**, not High-Entropy Alloys. Running inference on our HEA arXiv corpus produces empty extractions (domain shift problem).
- **Mitigation:** We fetched a separate SOFC arXiv corpus to run inference on, so the model operates in-domain.

**See also:** ADR-003 (decision to fetch SOFC-specific arXiv papers for inference).
