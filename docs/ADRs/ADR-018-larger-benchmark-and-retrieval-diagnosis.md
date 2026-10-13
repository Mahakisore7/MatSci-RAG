# ADR-018: A 28-Question Benchmark, and What It Says About the Real Bottleneck

**Status:** Accepted (expansion-on run pending, see Consequences)
**Date:** 2026-10-11
**Phase:** Phase 10
**Corrects:** a claim in ADR-016 and ADR-017 that unanswered questions lack an answering paper

---

## Context

Seven questions detect only large effects; the judge alone varies by about ±0.03. ADR-017's
filter experiment ended with differences of one or two questions, which the benchmark could not
resolve. It also left an explanation — "no paper states the answer, so coverage is the limit" —
that had been inferred from reading retrieved passages, not checked against the corpus.

## Decision

- Added `scripts/benchmark_questions.py` with two sets: **v1** (the original 7) and **v2**
  (28 = v1 + 21 new). The new questions were written from corpus titles and abstracts, across
  SOFC materials, microstructure and degradation, modelling and diagnostics, SOEC and PEM fuel
  cells. They were not chosen from retrieval output.
- `scripts/evaluate_ragas.py` takes `--benchmark v1|v2`. The default stays v1, so every earlier
  number is still reproducible.
- Judge, generator and index are unchanged, so v2 results are comparable across configurations
  but **not** with v1 numbers: v2 averages 28 questions and includes easier ones.
- The evaluator now saves each scored question as it goes and accepts `--resume`. Groq's free
  tier allows 200,000 judge tokens per day and one 28-question run uses nearly all of it; two runs
  were lost to that cap before this was added.
- Console output no longer crashes on symbols that the Windows codepage cannot print (this killed
  the first v2 attempt).

## Measurement

Full-text index (400 papers), `qwen3.8-27b`, expansion off, `docs/results/bench28/`:

| Set | Faithfulness | Answer Relevancy | Context Precision | "I don't know" |
|---|---:|---:|---:|---:|
| v2, all 28 | 0.982 | 0.607 | 0.582 | 11 / 28 |
| the original 7 within it | | 0.429 | | 3 / 7 |
| the 21 new ones | | 0.667 | | 8 / 21 |

No question was answered with a relevancy of 0 without also being an abstention; every
non-abstaining answer scored well.

## Diagnosis of the abstentions

For five of the eleven "I don't know" questions I checked whether a paper on that exact topic
exists in the corpus and where it ranks (hybrid retrieval, top 5):

| Question topic | Paper in corpus? | In top 5? |
|---|---|---|
| ML prediction of SOFC polarization resistance | yes | **no** (Li-ion ML papers returned) |
| Distribution of relaxation times, PEM | yes | yes, all 5 chunks; generator still declined |
| SOEC durability under dynamic load | yes | **no** (PEM prognostics papers returned) |
| Oxygen partial pressure in two-layer SOEC electrolyte | yes | **no** (grain-boundary paper returned) |
| Gas diffusion layer thickness, PEM | yes | **no** (other PEM papers returned) |

So in at least four of five cases the evidence exists and retrieval ranks it too low; in the
fifth it was retrieved five times, all from one paper, and the generator abstained. This
overturns the "coverage is the limit" reading: that held for none of the cases checked here. It is
five cases, not all eleven, and the claim about the original seven questions was never verified
either way.

## Consequences

- **The next lever is retrieval ranking, not more papers.** Cross-encoder re-ranking of the fused
  candidates, and a cap on chunks per paper so one paper cannot fill the whole top 5, are now
  ahead of any corpus fetch.
- ADR-016 and ADR-017 carry correction notes. Their measurements stand; only the explanation of
  the remaining abstentions changed.
- **Update:** the retrieval-ranking explanation was confirmed by ADR-019, whose re-ranker cut
  abstentions from 11 to 5 on this benchmark.
- **Pending:** the expansion-on v2 run reached question 19 of 28 before the daily judge cap
  stopped it, and its partial results were lost (the resume feature did not exist yet). Rerun it
  with `--resume` once the cap allows. Until then no v2 figure is claimed for expansion.
- v2 is still small and written by the same person who built the system; a held-out set written by
  someone else would be a stronger test.
