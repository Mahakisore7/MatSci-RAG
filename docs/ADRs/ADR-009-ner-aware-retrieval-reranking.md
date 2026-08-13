# ADR-009: NER-Aware Retrieval Reranking (Replacing the Broken Metadata Filter)

**Date:** 2026-08-13
**Status:** Accepted — Supersedes the "Known Limitation" section of [ADR-008](ADR-008-phase8-bugfixes-import-order-double-import.md)

---

## Context

ADR-008 logged, but did not root-cause or fix, an observation that the NER metadata filter in `hybrid_retrieve()` returned 0 dense results on every query:

```python
filter_dict = {"$or": [{"devices": {"$contains": "fuel"}}, {"devices": {"$contains": "oxide"}}]}
```

Separately, `detect_ambiguity()` in `rag_query.py` was found to be dead code: it read `doc.metadata.get("source_id")`, but `build_index.py` stores each chunk's paper id under the key `"paper_id"` — `source_id` never exists on a retrieved `Document`, so the function always returned `None` and the "uncertainty abstention" feature (RESULTS.md Phase 8) never actually ran.

## Root Cause #1: `$contains` Is Not a Metadata Operator

Reproduced in isolation against an in-memory ChromaDB collection (this project's `chromadb==1.5.9`): `$contains` is a `where_document` operator (full-text substring search over document content), not a valid `where` (metadata) operator. Used on metadata, it silently matches zero rows rather than raising:

```python
col.get(where={"devices": {"$contains": "fuel"}})   # -> [] , no exception
col.get(where_document={"$contains": "fuel"})         # -> matches, as expected
```

This exactly matches the "Merged 0 dense and 5 BM25 results" behavior logged in ADR-008.

## Root Cause #2: Even a Correct Filter Would Be the Wrong Design

Before fixing the operator, `data/structured_facts.jsonl` was inspected directly: **272 of 400 papers (68%) have an empty `devices` field.** A hard filter — even a mechanically correct one — would silently exclude most of the corpus from dense retrieval on every query. The `materials` field is also very noisy: raw NER output is full of punctuation, subword fragments (`##2`), and non-entity tokens, consistent with the low measured F1 in `RESULTS.md` (MatSciBERT: 0.0915, BERT-base: 0.0816).

## Decision

Replace the hard pre-filter with two additive, soft mechanisms that can't zero out the corpus:

1. **Reciprocal Rank Fusion (RRF).** Dense and BM25 result lists are combined via `score = Σ 1/(60 + rank)` per list (standard RRF constant), replacing the previous arbitrary "concatenate lists, keep first k by insertion order" merge.
2. **Entity-overlap boost.** For each candidate document's paper, its cleaned `materials`/`devices`/`experiments` entities (via the new shared `ner_utils.clean_entities`) are checked for substring overlap with the query text; each match adds a small fixed bonus (`0.02`, roughly one extra rank-1 RRF appearance) to that document's fused score.

The candidate pool pulled from each retriever was widened (`k*4`, minimum 20) before fusion, so there's a real pool for the boost to act on rather than just re-sorting 5 items. Implemented in `src/rag/ner_utils.py::fuse_and_rerank`, called from `src/rag/hybrid_retrieval.py::hybrid_retrieve`.

Measured effect on the known Phase 5 failure mode (*"Which cathode materials show high oxygen reduction activity?"*, which previously returned 4/5 battery cathode papers instead of SOFC ones): the genuinely relevant SOFC paper ("Material Discovery and Design Principles for Stable, High Activity Perovskite Cathodes for Solid Oxide Fuel Cells") now ranks in the top 3 instead of 4th, and is represented by two chunks in the top 5. This is a real, measured improvement — not a complete fix; 2 of 5 results are still off-domain battery papers, limited by how semantically similar "cathode" battery/SOFC abstracts are to the dense embedding model in the first place.

## Root Cause #3: The Ambiguity Threshold Fires on Noise, Not Signal

Fixing the `paper_id`/`source_id` key bug made `detect_ambiguity()` run for the first time. Its original logic — abstain if retrieved papers span more than 3 distinct `materials` — then immediately misfired: it triggered on ordinary, clearly on-topic questions (e.g. *"What is the effect of temperature on SOFC performance?"*), because the underlying signal is dominated by noise.

Verified directly: only 2 of 400 papers in `structured_facts.jsonl` contain a recognizable real material keyword (YSZ, GDC, LSCF, nickel, zirconia, ...) anywhere in their extracted `materials` list. A sweep across 15 varied test queries — including clearly irrelevant ones like *"What is the capital of France?"* — found noise-driven distinct-candidate counts of 4 to 18, with no separation between relevant and irrelevant queries. No fixed threshold can reliably distinguish signal from this much noise.

**Decision:** raise `MIN_DISTINCT_MATERIALS_FOR_AMBIGUITY` from 3 to 25 — comfortably above the observed noise ceiling — so the feature fails safe (never blocks an ordinary answer) instead of misfiring. In practice this means the feature will rarely trigger until NER precision improves (see "Future Work").

## Consequences

**Positive:**
- The "Hybrid (Dense + BM25 + NER)" retrieval toggle now genuinely differs from BM25-only, with a measured ranking improvement on the known cathode-ambiguity failure case.
- `detect_ambiguity()` is reachable code again and covered by unit tests, without reintroducing the false-positive regression found during this fix (confirmed via `tests/test_rag_generation.py`'s guardrail test, which failed at the naive `>3` threshold and passes at 25).
- Both fixes are pure-logic and unit-tested (`tests/test_ner_utils.py`) without requiring any model or index to be loaded.

**Negative:**
- The ambiguity/abstention feature is now honestly closer to dormant-by-design than a reliably firing feature — this trades "broken" for "rarely useful," not for "fully working." A genuinely useful version needs materially better NER precision.
- The `0.02` NER boost weight and `k*4` pool size are reasonable, empirically-checked defaults, not formally tuned hyperparameters.

**Future Work:** If Phase 2 NER training is revisited (more epochs, more data, or a larger backbone) and `materials` F1 improves substantially, `MIN_DISTINCT_MATERIALS_FOR_AMBIGUITY` should be re-calibrated downward using the same empirical sweep method used here, and the NER boost weight could reasonably increase.
