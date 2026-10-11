# ADR-019: Cross-Encoder Re-Ranking After Hybrid Retrieval

**Status:** Accepted (end-to-end judge evaluation pending, see Consequences)
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

## Limits of this evidence

- The gold targets were written by the system's author from corpus titles (see the file's
  docstring). Other valid sources may exist, which makes hit@5 slightly pessimistic, but an author-chosen
  gold set can also favour how the author thinks about the topics.
- 21 questions; one miss changes hit@5 by about 5 points.
- Gold targets exist only for the 21 new questions; the original 7 are not covered.
- Only one cross-encoder was tried. A larger one (for example `bge-reranker-base`) may do better.
- hit@5 measures retrieval, not answer quality. The judged end-to-end effect has not been measured.

## Consequences

- Retrieval is no longer the main known bottleneck on this benchmark; the remaining failures are
  cases like the GDL-thickness question, whose target paper is not in the 20-candidate pool at all
  (the fix there is a larger pool), and cases where the right paper is retrieved but the answer is
  not stated in the retrieved chunks.
- **Pending:** a judged 28-question run with re-ranking, to measure the effect on Answer Relevancy
  and the abstention count (11/28 before). It needs about 190,000 judge tokens and Groq's daily cap
  is 200,000, so it must run on a day when the cap has reset (`--resume` is available).
- The first query after start-up loads the cross-encoder (a few seconds, one-time).
