# ADR-015: Generator Size — flan-t5-large Evaluated and Not Adopted

**Status:** Accepted
**Date:** 2026-10-06
**Phase:** Phase 10

---

## Context

The Phase 9 follow-up (RESULTS.md) found that combining full-paper ingestion with query
expansion collapsed Answer Relevancy to 0.000, and attributed it to `flan-t5-base` (250M
parameters) producing incoherent fragments from long, noisy PDF context. The obvious test of
that hypothesis is a bigger model from the same family: `flan-t5-large` (780M), which drops
into the existing pipeline with no other change.

Two things had to happen first:

1. **Full-text coverage was only 55 / 400 papers.** It is now **399 / 400** (one PDF returns
   404 on arXiv; that paper falls back to its abstract so it is not dropped from the index).
2. **The evaluation judge had been retired.** Groq removed `groq/compound-mini`; every
   configuration was re-measured under `openai/gpt-oss-120b` (ADR-010 update). Old and new
   scores are not comparable.

## Measurement

Eight configurations, same 7 benchmark questions, same judge (`docs/results/v2/`):

| Index | Generator | Expansion | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|---|
| Abstracts | flan-t5-base | off | 0.857 | 0.000 | 0.314 |
| Abstracts | flan-t5-base | on | 0.857 | **0.143** | 0.486 |
| Abstracts | flan-t5-large | off | 0.857 | 0.000 | 0.343 |
| Abstracts | flan-t5-large | on | 0.857 | 0.071 | 0.457 |
| Full text | flan-t5-base | off | 0.857 | 0.071 | **0.586** |
| Full text | flan-t5-base | on | 0.857 | 0.000 | 0.571 |
| Full text | flan-t5-large | off | 0.714 | 0.000 | **0.586** |
| Full text | flan-t5-large | on | 0.857 | 0.000 | 0.543 |

Context Precision depends only on retrieval, so its spread between generators on the same
index (0.314 vs 0.343) is a direct read of judge noise: roughly ±0.03.

## Findings

- **Retrieval is no longer the bottleneck.** Full text raises Context Precision from 0.314 to
  0.586; on the cathode and anode questions the judge rates the retrieved passages 1.0 and 0.9.
- **flan-t5-large does not fix generation.** On full text it answers *"I don't know"* to 6 of 7
  questions — including those two, where the right passages were retrieved. It is more
  conservative, not more capable. It never beats flan-t5-base on Answer Relevancy.
- **The earlier "collapse" was not specific to full text + expansion.** Under the stricter
  judge, Answer Relevancy is at or near zero in every configuration. The Flan-T5 family —
  base or large — cannot reliably synthesize an answer from several technical passages.

## Decision

- Keep `flan-t5-base` as the default generator. `flan-t5-large` costs ~3× the memory and
  latency for no measured gain.
- Keep the generator configurable (`GEN_MODEL_NAME`), since the next experiment is a different
  model, not a different size.
- Serve the **full-text index** by default: it is the configuration with the best retrieval.

## Consequences

- The highest-value next step is a modern instruction-tuned generator (≈7–8B parameters,
  local or via API). The pipeline already isolates generation behind `answer_query()`, and
  the eval harness can measure the change in one command.
- Seven questions detect large effects only. The near-zero Answer Relevancy floor is a large
  effect; small differences between rows above are not significant.
