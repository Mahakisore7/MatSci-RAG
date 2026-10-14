# ADR-019: Cross-Encoder Re-Ranking After Hybrid Retrieval

**Status:** Accepted (default for query expansion under review, see Consequences)
**Date:** 2026-10-11
**Phase:** Phase 10

---

## Context

ADR-018 found that most abstentions on the 28-question benchmark were retrieval failures: the
answering paper was in the corpus but ranked outside the top 5. The first stage (BGE dense search +
BM25, fused by RRF, plus the NER boost of ADR-009) scores the query and each passage separately.
That is fast, but it cannot tell "this passage mentions the same words" from "this passage answers
the question", so Li-ion battery ML papers outranked the SOFC one for an SOFC machine-learning query.

## Decision

- Add a **cross-encoder second stage** (`src/rag/rerank.py`). It takes the fused top-20 candidates,
  scores each (query, passage) pair jointly with `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M
  parameters, runs locally, no API), and keeps the best 5.
- Add an optional **per-paper cap** (`per_paper_cap`), which limits how many of the final chunks
  may come from one paper. It is implemented and measured but **off by default** (see below).
- Re-ranking is **on by default**. `USE_RERANK=0` turns it off, `RERANK_MODEL` swaps the model.
- Measurement is judge-free: `scripts/evaluate_retrieval.py` checks whether a gold paper is in the
  top 5. It makes no LLM calls, so it does not spend Groq's daily judge budget.

## Measurement

21 benchmark-v2 questions that have gold target papers (`scripts/retrieval_gold.py`), top 5,
`docs/results/bench28/retrieval_gold.json`:

| Configuration | hit@5 | MRR | Distinct papers in top 5 |
|---|---:|---:|---:|
| Baseline (RRF + NER boost) | 0.714 | 0.448 | 2.71 |
| Cap 2 chunks per paper | 0.762 | 0.463 | 3.43 |
| **Cross-encoder** | **0.952** | **0.849** | 2.86 |
| Cross-encoder + cap 2 | 0.952 | 0.849 | 3.52 |

- Re-ranking moves the answering paper into the top 5 for 20 of 21 questions (baseline: 15).
- The per-paper cap adds source diversity but no additional hits once the cross-encoder is on, and
  it was not tested for answer quality, so it stays optional.
- Cost: about +100 ms per query (218 → 317 ms on this machine).
- Smoke test, answers from the live pipeline: the SOEC-durability question, an abstention before,
  is now answered from the right paper. The SOFC machine-learning question now retrieves the right
  paper but the generator still says "I don't know", so retrieval was necessary but not sufficient.

## Judged end-to-end result

Full-text index, `qwen3.8-27b`, expansion off, same 28 questions and same judge, with re-ranking
off vs on (`docs/results/bench28/fullpaper_qwen_{,rerank_}exp-off.csv`):

| | Faithfulness | Answer Relevancy | Context Precision | "I don't know" |
|---|---:|---:|---:|---:|
| Before (no re-ranking) | 0.982 | 0.607 | 0.582 | 11 / 28 |
| **With re-ranking** | 0.964 | **0.821** | **0.689** | **5 / 28** |
| Original 7 within it: Answer Relevancy | | 0.429 → **0.714** | | 3 → 0 abstentions |
| New 21 within it: Answer Relevancy | | 0.667 → **0.857** | | 8 → 5 abstentions |

- Answer Relevancy improved on 7 questions and worsened on 1 (anode microstructure over time,
  1 → 0); 20 were unchanged. The two original questions that had been abstaining since ADR-016
  (temperature effects, LSCF sintering temperatures) are now answered, which is further evidence
  that the earlier "coverage" explanation was wrong.
- Context Precision rose 0.582 → 0.689, as expected from better ranking.
- **Faithfulness fell slightly (0.982 → 0.964).** One answer (grain boundaries) dropped from 1.0 to
  0.0: it states "1.35 eV", which does not appear verbatim in the retrieved passages (three other
  figures it quotes do), so this looks like a genuinely unsupported detail rather than judge noise,
  although text-formatting differences could explain part of it. One other answer rose 0.5 → 1.0.
  Answering more questions means more chances to add an unsupported detail.
- Single run per configuration; the judge varies by about ±0.03 on Context Precision, and Answer
  Relevancy is nearly binary per question, so one question moves it by 0.036.

## Limits of this evidence

- The gold targets were written by the system's author from corpus titles (see the file's
  docstring). Other valid sources may exist, which makes hit@5 slightly pessimistic, but an author-chosen
  gold set can also favour how the author thinks about the topics.
- 21 questions; one miss changes hit@5 by about 5 points.
- Gold targets exist only for the 21 new questions; the original 7 are not covered.
- Only one cross-encoder was tried. A larger one (for example `bge-reranker-base`) may do better.
- hit@5 measures retrieval, not answer quality; the judged effect is measured separately below.

## Consequences

- Retrieval is no longer the main known bottleneck on this benchmark; the remaining failures are
  cases like the GDL-thickness question, whose target paper is not in the 20-candidate pool at all
  (the fix there is a larger pool), and cases where the right paper is retrieved but the answer is
  not stated in the retrieved chunks.
- Re-ranking is therefore confirmed end to end on this benchmark: abstentions fell from 11 to 5
  and Answer Relevancy rose from 0.607 to 0.821.
- **Query expansion on top of re-ranking** was judged on 2026-10-14
  (`docs/results/bench28/fullpaper_qwen_rerank_exp-on.csv`):

  | | Faithfulness | Answer Relevancy | Context Precision | "I don't know" |
  |---|---:|---:|---:|---:|
  | Re-ranking | 0.964 | 0.821 | 0.689 | 5 / 28 |
  | Re-ranking + expansion | 0.929 | 0.821 | 0.671 | 4 / 28 |

  Relevancy is identical (4 questions better, 4 worse); one fewer question abstains; faithfulness
  falls 0.964 → 0.929 (two answers scored 0.0). With re-ranking in place the rewrite is redundant
  and costs an extra LLM call, so it no longer earns its place.
- **Open decision:** the app still defaults expansion on (ADR-016). It is one run per configuration,
  so the default is unchanged until a repeat run confirms the result; turning it off is a one-line
  change in `src/ui/views/chat.py`.
- The first query after start-up loads the cross-encoder (a few seconds, one-time).
