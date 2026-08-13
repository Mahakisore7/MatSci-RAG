# ADR-002: MatSciBERT vs BERT-base as Fine-Tuning Backbone

**Status:** Accepted (F1 numbers below superseded — see note)  
**Date:** 2026-07-25  
**Author:** Mahakisore7

> **Note (2026-08-13):** The F1 numbers in this ADR were measured *before*
> [ADR-004](ADR-004-fix-character-offset-bug.md)'s character-offset bugfix
> and are invalid — they were computed on misaligned training data. The
> current, valid numbers are in `RESULTS.md`
> (MatSciBERT: 0.0915, BERT-base: 0.0816). The **directional conclusion**
> below (MatSciBERT > BERT-base) still holds under the corrected numbers;
> only the absolute values changed.

---

## Context

For the NER model, we needed to choose which pre-trained language model to use as the backbone. Both models have 110 million parameters and identical architecture. The only difference is what text they were pre-trained on:

- **`bert-base-uncased`:** Pre-trained by Google on Wikipedia + BooksCorpus (~3.3 billion words of general English).
- **`m3rg-iitd/matscibert`:** Pre-trained by IIT Delhi's M3RG Lab on 2.4 million materials science papers.

## Decision

We fine-tuned **both models** on the SOFC-Exp dataset under identical conditions and compared results. MatSciBERT was expected to perform better as the primary model.

## Reasoning

The central hypothesis of the MatSci-RAG project is: *"A model pre-trained on domain-specific scientific literature will outperform a general-purpose model on materials science NER."*

By running the same experiment on both models and reporting an F1 comparison, we produce a **controlled scientific result** that directly validates (or falsifies) this hypothesis. This is more rigorous than simply using MatSciBERT alone.

## Results

| Model | Test F1 |
|---|---|
| MatSciBERT (`m3rg-iitd/matscibert`) | **0.0359** |
| BERT-base (`bert-base-uncased`) | 0.0346 |

MatSciBERT outperformed generic BERT by **+3.8% relative**, confirming the hypothesis.

## Consequences

**Positive:**
- Provides a quantitative, defensible comparison for the final report.
- Demonstrates understanding of domain adaptation in NLP.
- MatSciBERT is used downstream for all inference tasks.

**Negative:**
- Both absolute F1 scores are low due to class imbalance (SOFC papers have few named entities per sentence) and the 512-token truncation discarding entity-rich sentences. This is documented in `RESULTS.md`.

**Note:** Low absolute F1 does not invalidate the relative comparison. The key finding is the directional superiority of domain-specific pretraining.
