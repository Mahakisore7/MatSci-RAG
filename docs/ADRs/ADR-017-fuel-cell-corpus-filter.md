# ADR-017: Fuel-Cell Corpus Filter — Built, Measured, Not the Default

**Status:** Accepted (filter available as an option; full corpus stays the default)
**Date:** 2026-10-11
**Phase:** Phase 10

---

## Context

The arXiv keyword search behind the corpus matched generic words such as "cell", "anode" and
"electrolyte". Only 145 of the 400 papers mention a fuel-cell technology in their title or
abstract; the rest are mostly Li-ion batteries, catalysis and unrelated physics (one is about
vehicle path planning). After ADR-016, three of the seven benchmark questions were still answered
"I don't know", and the hypothesis was that off-topic chunks were crowding out relevant ones.

## Decision

- Added `src/rag/corpus_filter.py`. A paper is kept if its title or abstract matches
  *fuel cell, SOFC, solid oxide, SOEC, electrolysis cell, PEMFC* or *PCFC*.
- Wider terms (ceria, YSZ, GDC, LSCF, oxygen vacancy, oxide-ion …) were tried. They added only
  5 papers, all off-topic (Li-ion cathodes, water treatment, oxide-surface physics), so the list
  stays strict.
- `python -m src.rag.build_index --source full_paper --fuel-cell-only` builds the filtered index:
  145 papers, **7,633 vectors** (vs 22,659).

## Measurement

Same 7 questions, same judge, `qwen3.8-27b` generator (`docs/results/v2/fuelcell_qwen_*.csv`):

| Index | Expansion | Faithfulness | Answer Relevancy | Context Precision |
|---|:---:|---:|---:|---:|
| Full text, 400 papers | | 1.000 | 0.429 | 0.543 |
| Full text, 400 papers | on | 0.857 | **0.714** | 0.543 |
| Fuel-cell only, 145 papers | | 0.857 | 0.429 | **0.600** |
| Fuel-cell only, 145 papers | on | 0.786 | 0.571 | 0.514 |

## Findings

- **No measurable gain.** The differences are one or two questions flipping, within the noise of
  a 7-question benchmark. With expansion the filtered index is slightly worse (the GDC-interlayer
  question was answered on the full corpus and not on the filtered one).
- **The off-topic papers weren't the problem.** The questions that still fail (grain size and
  ceria conductivity, GDC interlayers, LSCF sintering temperatures) fail on both indexes.

> **Correction (ADR-018):** this ADR first said these fail "because no paper states the answer"
> and proposed fetching more papers. That was inferred, not checked. A diagnosis on the larger
> benchmark found on-topic papers already in the corpus that retrieval ranks outside the top 5.
> The measurements above stand; the explanation and the next step changed.

## Consequences

- The app keeps serving the full 400-paper index; filtering is a one-flag option.
- The filtered index is 3× smaller and builds 3× faster, which is useful for experiments.
- Next lever is **retrieval ranking** (ADR-018): re-ranking and a per-paper chunk cap. A larger
  benchmark (also ADR-018) lets smaller effects like this one be resolved.
