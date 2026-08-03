# ADR-003: Fetch SOFC-Specific arXiv Corpus for NER Inference

**Status:** Accepted  
**Date:** 2026-08-03  
**Author:** Mahakisore7

---

## Context

Phase 3 requires running our trained MatSciBERT NER model on real papers to produce a structured facts table (`data/structured_facts.jsonl`).

The existing corpus (`data/corpus.jsonl`) contains 400 arXiv abstracts about **High-Entropy Alloys (HEA)** — the original topic chosen for Phase 1. However, our NER model was trained exclusively on **SOFC (Solid Oxide Fuel Cell)** papers.

When we ran inference on the HEA corpus, every paper returned empty entity lists. This is a **domain shift** problem: the model never encountered HEA-specific vocabulary (e.g., `CoCrFeMnNi`, `BCC phase`, `FCC structure`) during training, so it classifies every token as `O` (outside/not an entity).

## Decision

Fetch a **separate 400-paper SOFC corpus** (`data/sofc_corpus.jsonl`) from arXiv using the same `fetch_corpus.py` script with an SOFC-specific query, and run Phase 3 inference on that corpus instead.

## Reasoning

- The HEA corpus remains intact for potential future use (no data is deleted).
- Running in-domain inference (SOFC model on SOFC papers) produces real, meaningful extractions.
- This is the correct scientific approach: a model should be evaluated and deployed in the domain it was trained on.
- The `fetch_corpus.py` script already supports a configurable query — this requires changing one constant, not rewriting code.

## SOFC arXiv query used

```
solid oxide fuel cell SOFC electrolyte anode cathode material
```

## Consequences

**Positive:**
- The structured facts table will contain real, correct material/device extractions (e.g., YSZ, GDC, LSCF).
- Demonstrates correct understanding of domain adaptation limitations.

**Negative:**
- The project now has two separate corpora (HEA and SOFC). This is a deliberate, documented tradeoff.
- The RAG system in Phase 4 will be built on the SOFC corpus, not the original HEA corpus.

**Future Work:** A production system would either (a) train on a multi-domain dataset covering both HEA and SOFC, or (b) train separate models for each domain and route queries to the appropriate model.
