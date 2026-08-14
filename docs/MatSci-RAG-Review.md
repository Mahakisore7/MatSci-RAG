<!-- Title: MatSci-RAG Project Review -->

# MatSci-RAG: A Domain-Specific RAG System for Solid Oxide Fuel Cell Literature

**Project Review Document**
Prepared: 2026-08-14 · Status: v1.0.0, Phases 0–8 complete, post-implementation bug-fix and hardening pass complete

> **Sourcing note for the presenter:** every number, code reference, and behavioral claim in this
> document was independently re-verified against the live codebase and test suite while writing
> it (not copied from memory) — see the "Preliminary Results" section for what was re-run and
> when. The external literature references in Section 3 are transcribed as this project's own ADRs
> and build guide already characterize them; cross-check exact author/venue/year details against
> your Zeroth Review document before citing them in a formal report, since that was not
> independently re-verified against the original papers.

---

## At a Glance

| | |
|---|---|
| **Domain** | Solid Oxide Fuel Cell (SOFC) materials science literature |
| **Corpus** | 400 arXiv abstracts → 1,330 indexed chunks |
| **NER backbone** | `m3rg-iitd/matscibert` (domain-pretrained), benchmarked against `bert-base-uncased` |
| **Retrieval** | Hybrid dense (ChromaDB / BGE-small) + BM25, fused via Reciprocal Rank Fusion + NER-entity boost |
| **Generation** | `google/flan-t5-base`, strictly context-grounded, with self-verification |
| **Retrieval hit-rate** | 100% (20/20 hand-written fact-lookup questions, re-verified) |
| **Test suite** | 20 automated tests, `ruff` clean, GitHub Actions CI |

---

## 1. Introduction

Materials science publishes faster than any one researcher can read. A question like *"what
electrolyte materials have been reported for intermediate-temperature SOFCs?"* has an answer
somewhere in the literature — but finding it means searching by exact terminology (chemical
formulas, device acronyms like **YSZ**, **GDC**, **LSCF**) that generic search and generic
language models don't reliably handle, and generic LLMs asked the question directly will often
answer fluently and confidently from *general* training knowledge rather than admit they don't
know a specific paper's finding.

**MatSci-RAG** is a Retrieval-Augmented Generation (RAG) system built to answer materials-science
questions **only** from a curated, verifiable literature corpus — Solid Oxide Fuel Cell (SOFC)
research abstracts pulled from arXiv — instead of from a model's general knowledge. It combines
two things that are usually built separately:

1. A **Named Entity Recognition (NER)** model, fine-tuned specifically to recognize
   materials-science entities (materials, devices, experiments, measured values) in this text,
   producing a structured facts table alongside the raw corpus.
2. A **RAG pipeline** — dense + keyword retrieval, grounded generation, and reliability
   safeguards (self-verification, ambiguity abstention) — that uses those entities not just to
   *display* structure, but to make retrieval itself more precise.

The project is built and documented as a sequence of 9 phases (0–8), each with its own
Architecture Decision Records (ADRs) explaining what was chosen and why — including several real
bugs found and fixed along the way, which this document reports honestly rather than omitting.

---

## 2. Problem Statement

> Given a natural-language question about Solid Oxide Fuel Cell materials science, retrieve the
> most relevant passages from a curated arXiv literature corpus and generate an answer **grounded
> strictly in that retrieved text** — using domain-specific entity extraction (NER) to make
> retrieval measurably more precise than semantic search alone — while explicitly **refusing** to
> answer when the corpus doesn't contain the answer, and **flagging** (rather than silently
> guessing) when the retrieved evidence is genuinely ambiguous across multiple candidate
> materials.

Three sub-problems fall out of that statement, and map directly onto the system's three main
technical components:

| Sub-problem | Addressed by |
|---|---|
| Generic embedding search misses exact domain terms/formulas and returns off-domain results (e.g. battery papers for a "cathode" query) | Hybrid dense + BM25 retrieval, NER-boosted reranking |
| A fluent language model will answer confidently even without real supporting evidence | Strict context-only prompting + a second LLM call that checks the answer against the context |
| A short passage can span more than one material, risking a silently wrong attribution | NER-driven ambiguity detection that can abstain and ask the user to narrow the question |

---

## 3. Literature Review

This project is built as a **direct, documented extension of a specific base paper**: **Maharana
et al. (2025)**, which proposed a RAG + NER pipeline for mining materials-science literature (in
that paper's case, the solid-state hydrogen-storage domain). This project's own literature review
(conducted prior to implementation, recorded in the project's Zeroth Review document) identified
specific, reported limitations of that base paper's approach, which directly motivate this
project's design — detailed in Section 4.

Supporting references drawn on for individual components:

- **Polak & Morgan — "ChatExtract."** A conversational, prompt-engineering approach to extracting
  accurate materials data from papers, using follow-up verification prompts to reach roughly 90%
  precision. This project's self-verification step (`verify_answer()` — a second LLM call asking
  "does the context actually support this claim?") is a direct, smaller-scale application of that
  technique.
- **Brack et al. (2020) — the SOFC-Exp corpus.** A CC-BY-4.0, human-annotated materials-science
  NER dataset (876 sentences; `MATERIAL`, `DEVICE`, `VALUE`, `EXPERIMENT` entity types), used here
  as the labeled training data for the NER component (Phase 2).
- **Trewartha et al. (2022).** Quantifies the advantage of domain-specific language model
  pretraining over generic pretraining for materials-science NER, particularly in low-data,
  entity-sparse regimes, and uses SOFC-Exp as one of its own benchmark datasets. This project's
  Phase 2 — a controlled, identical-hyperparameter comparison of `m3rg-iitd/matscibert` (trained on
  2.4M materials-science papers) against `bert-base-uncased` (trained on general English) — is
  designed as a direct, smaller-scale replication of that finding on this project's own data split.
- **General RAG retrieval literature.** It is a well-documented property of dense embedding
  retrieval that it prioritizes semantic similarity over exact wording, and so can miss an exact
  alloy name, chemical formula, or acronym that a keyword search would catch — the standard
  motivation for combining dense retrieval with sparse/keyword methods (BM25) in production RAG
  systems, which this project does in Phase 8.

---

## 4. Research Gap / Motivation

The base paper (Maharana et al., 2025) and the supporting literature above report specific,
concrete limitations. Each stretch feature in this project's design traces back to one of them —
not invented from scratch, but scoped as direct responses:

| Reported limitation | This project's response |
|---|---|
| No automatic step to check whether a generated answer is actually supported by the retrieved evidence | **Self-verification**: a second LLM call, independent of the answer-generation call, explicitly judges "yes/no, does the context support this claim" and surfaces it in the UI as a ✅/⚠️ badge |
| When a retrieved passage (or set of passages) discusses multiple materials, the system risks silently attributing a fact to the wrong one | **NER-driven ambiguity detection**: counts distinct materials named across the retrieved evidence's structured facts and can abstain, asking the user to narrow the question, instead of guessing |
| Dense/embedding-only retrieval misses exact terminology and can pull semantically-similar-but-wrong-domain results | **Hybrid retrieval**: dense (ChromaDB/BGE-small) + keyword (BM25), combined via Reciprocal Rank Fusion |

A fourth gap was found **during this project's own engineering**, not in the source literature —
worth reporting honestly to a reviewer as a sign of process rigor, not a weakness: an early
implementation of NER-informed retrieval (a hard metadata filter gating dense search to papers
whose extracted "device" entities mentioned certain keywords) was audited and found to be both
**mechanically broken** (the ChromaDB filter operator used doesn't apply to metadata fields — it
silently matched zero documents on every query) and, independent of that bug, **conceptually the
wrong design** for this corpus: 68% of the 400 papers have no extracted "device" entity at all, so
even a working hard filter would have discarded most of the corpus on every search. This was
replaced with a soft, entity-overlap **reranking boost** that improves ranking without ever being
able to zero out results — documented in full, including the empirical evidence, in ADR-009.

**The core motivation, restated:** general-purpose RAG treats retrieval as a purely semantic
problem. This project's premise is that a domain that already has structured facts extractable via
NER (materials, devices) shouldn't discard that structure at retrieval time — but that using it
correctly requires being honest about how reliable the extraction actually is, and designing the
retrieval mechanism to degrade gracefully, not catastrophically, when it isn't perfect.

---

## 5. Project Objectives

1. Build an automated pipeline to collect a domain-specific corpus (SOFC materials-science
   abstracts) from arXiv.
2. Fine-tune a materials-science domain-pretrained language model for NER, and empirically test —
   under a controlled, identical-hyperparameter comparison — whether domain-specific pretraining
   measurably outperforms a generic backbone on this task.
3. Apply the trained NER model to extract a structured facts table (materials, devices,
   experiments, values) from the project's own corpus.
4. Build a dense vector index over the corpus for semantic retrieval, with per-chunk NER metadata
   attached.
5. Implement a grounded question-answering pipeline that answers **only** from retrieved context
   and explicitly refuses to answer questions the corpus doesn't support.
6. Extend retrieval beyond pure dense search with keyword (BM25) matching and NER-informed
   reranking, addressing dense retrieval's known blind spot for exact terminology.
7. Add reliability layers on top of generation: self-verification of generated answers against
   their source context, and abstention when retrieved evidence is genuinely ambiguous.
8. Package the system behind an interactive UI, validate it with an automated test suite and CI,
   and quantitatively evaluate both retrieval and generation quality — honestly, including where
   the system's current limitations are.

---

## 6. Methodology / Proposed Work Pipeline

The system is built in four dependent layers, each phase assuming the previous one's output
already exists and is verified.

### 6.1 System Architecture

```mermaid
flowchart TB
    subgraph DATA["DATA LAYER — Phases 1–3"]
        direction TB
        arxiv["arXiv API<br/><i>SOFC-focused query</i>"] --> corpus["sofc_corpus.jsonl<br/>400 abstracts"]
        sofcexp["SOFC-Exp dataset<br/><i>Brack et al. 2020, labeled</i>"] --> train["Fine-tune:<br/>MatSciBERT vs BERT-base"]
        train -->|"F1: 0.0915 vs 0.0816"| pick["MatSciBERT selected"]
        corpus -->|"apply trained model"| pick
        pick --> facts["structured_facts.jsonl<br/>materials · devices · experiments · values"]
    end

    subgraph RETR["RETRIEVAL LAYER — Phase 4 / 8"]
        direction TB
        corpus2["sofc_corpus.jsonl"] -.->|"same corpus"| chunk["Chunk<br/>500 chars / 50 overlap"]
        facts -.->|"joined as metadata"| chunk
        chunk --> embed["Embed<br/>BAAI/bge-small-en-v1.5"]
        embed --> chroma[("ChromaDB<br/>1,330 vectors")]
    end

    subgraph GEN["GENERATION + RELIABILITY LAYER — Phase 5 / 8"]
        direction TB
        query["User query"] --> hybrid["hybrid_retrieve()<br/>dense + BM25 → RRF fusion<br/>+ NER entity-overlap boost"]
        chroma --> hybrid
        hybrid --> amb{"detect_ambiguity()<br/>too many distinct<br/>materials?"}
        amb -->|"no (usual case)"| prompt["build_prompt()<br/>context-only instruction"]
        amb -->|"yes (rare)"| abstain["Abstain:<br/>ask user to narrow"]
        prompt --> flant5["Flan-T5-base<br/>generate()"]
        flant5 --> verify["verify_answer()<br/>2nd LLM call: supported?"]
    end

    subgraph UI["INTERFACE + OPS LAYER — Phase 6–7"]
        direction TB
        streamlit["Streamlit chat UI<br/>answer + ✅/⚠️ badge + sources"]
        cieval["pytest + ruff + GitHub Actions CI"]
        evalscript["evaluate_rag.py<br/>20-question hit-rate"]
    end

    verify --> streamlit
    abstain --> streamlit

    style DATA fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    style RETR fill:#eff6ff,stroke:#3b82f6,color:#1e293b
    style GEN fill:#fff7ed,stroke:#f97316,color:#431407
    style UI fill:#f0fdf4,stroke:#22c55e,color:#052e16
```

### 6.2 NER Pipeline (Phases 2–3)

```mermaid
flowchart LR
    A["SOFC-Exp corpus<br/>876 sentences<br/>CSV + TXT annotations"] --> B["Parse + align<br/>character-offset spans"]
    B --> C["70/15/15<br/>train/val/test split"]
    C --> D1["Fine-tune<br/>m3rg-iitd/matscibert"]
    C --> D2["Fine-tune<br/>bert-base-uncased"]
    D1 --> E{"Compare F1<br/>identical hyperparameters"}
    D2 --> E
    E -->|"MatSciBERT wins:<br/>0.0915 vs 0.0816"| F["Select MatSciBERT<br/>for inference"]
    G["Project's own SOFC<br/>arXiv corpus (400 papers)"] --> H["Run inference<br/>512-token truncation"]
    F --> H
    H --> I["structured_facts.jsonl<br/>per-paper: materials, devices,<br/>experiments, values"]

    style E fill:#fef3c7,stroke:#d97706,color:#451a03
    style F fill:#dcfce7,stroke:#16a34a,color:#052e16
```

### 6.3 Query-Time Sequence (What Happens for One Question)

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant H as hybrid_retrieve()
    participant C as ChromaDB (dense)
    participant B as BM25 (keyword)
    participant N as ner_utils
    participant A as detect_ambiguity()
    participant L as Flan-T5-base

    U->>H: question
    H->>C: dense search (pool = k×4)
    H->>B: keyword search (pool = k×4)
    C-->>H: ranked candidates
    B-->>H: ranked candidates
    H->>N: fuse_and_rerank(dense, bm25, query, facts)
    Note over N: Reciprocal Rank Fusion score<br/>+ bonus per NER entity mentioned in query
    N-->>H: top-k reranked passages
    H-->>U: retrieved passages
    U->>A: check retrieved passages
    alt too many distinct materials (rare)
        A-->>U: abstain — "please narrow your question"
    else normal case
        A-->>U: proceed
        U->>L: grounded prompt (context + question)
        L-->>U: generated answer
        U->>L: "does context support this claim?"
        L-->>U: verification (yes/no + quote)
        U->>U: render answer + ✅/⚠️ badge + sources
    end
```

### 6.4 Phase-by-Phase Summary

| Phase | Deliverable |
|---|---|
| 0 | Environment: `uv`, `ruff`, pre-commit, logging, CI scaffold |
| 1 | Corpus collection: 400 SOFC abstracts from arXiv |
| 2 | NER training: MatSciBERT vs. BERT-base, controlled comparison |
| 3 | NER inference: structured facts extracted from the project's own corpus |
| 4 | Indexing: chunking, embedding, ChromaDB persistence with NER metadata attached |
| 5 | Generation: retrieval + strictly context-grounded answering |
| 6 | Evaluation: hand-written test questions, measured retrieval hit-rate |
| 7 | Interface: Streamlit chat UI, GitHub Actions CI |
| 8 | Reliability: self-verification, ambiguity abstention, hybrid NER-boosted retrieval |

**Tech stack:** `uv` (env/deps) · `ruff` (lint/format) · `pytest` (tests) · HuggingFace
`transformers`/`datasets` (NER fine-tuning) · `m3rg-iitd/matscibert` & `bert-base-uncased`
(NER backbones) · `BAAI/bge-small-en-v1.5` (dense embeddings) · `ChromaDB` (vector store) ·
`rank-bm25` (keyword retrieval) · `google/flan-t5-base` (generation) · `LangChain` (document/
retriever plumbing) · `Streamlit` (UI) · GitHub Actions (CI).

---

## 7. Preliminary Results

All figures below were re-verified against the live system while preparing this document (not
carried forward from older notes) — see `RESULTS.md` for the full evidence trail.

### 7.1 NER: Domain Pretraining vs. Generic Pretraining

| Model | Test F1 | Training time (60 epochs) |
|---|---|---|
| **MatSciBERT** (`m3rg-iitd/matscibert`) | **0.0915** | ~20 min |
| BERT-base (`bert-base-uncased`) | 0.0816 | ~16 min |

MatSciBERT outperforms the generic backbone — confirming the central hypothesis that
domain-specific pretraining helps, consistent with Trewartha et al.'s finding for this task. Getting
to this valid result required finding and fixing a real bug along the way, reported honestly rather
than only showing the final number:

| Run | Config change | Result | Valid? |
|---|---|---|---|
| 1 | Baseline, 10 epochs | F1 0.0359 / 0.0346 | ❌ invalidated — character-offset bug (ADR-004) |
| 2 | 40 epochs | F1 0.0851 / 0.1029 | ❌ same bug still present |
| 3 | **Bugfix** (`newline=""` preserving `\r\n` offsets) + 60 epochs, warmup, weight decay | **F1 0.0915 / 0.0816** | ✅ first valid numbers |

**Honest limitation:** even the valid F1 is low in absolute terms. Verified directly against
`structured_facts.jsonl`: only 2 of 400 papers contain a recognizable real material name (YSZ, GDC,
nickel, zirconia, ...) anywhere in their extracted `materials` list — most raw extractions are
noise (punctuation, subword fragments, non-entity tokens). This is a real, current constraint on
the system, not a hidden one — see Section 8 and ADR-009 for how the system was designed to stay
safe despite it, and Section 8's "remaining work" for how it could be improved.

### 7.2 Retrieval Quality

- **Hit-rate:** 20/20 (100%) on a hand-written set of specific, fact-based test questions
  (`scripts/evaluate_rag.py`, k=5, dense retrieval) — re-run today, result unchanged.
- **Known failure mode, and a live, answer-level fix — not just a ranking shift.** The query
  *"Which cathode materials show high oxygen reduction activity?"* is a documented Phase 5 failure
  mode (RESULTS.md): it pulls off-domain battery-cathode papers under dense-only search. Run live
  in the app for this document:

  | Retrieval strategy | Generated answer | Verified? |
  |---|---|---|
  | Dense Only | **"FePO4"** — a battery cathode material, off-domain | ✅ (self-verification confirms it against its *own*, wrong, retrieved context — verification can only check internal consistency, not domain correctness, which is exactly why retrieval quality matters) |
  | Hybrid (Dense + BM25 + NER) | **"Perovskite."** — the correct SOFC-domain cathode material class, sourced from the paper "Material Discovery and Design Principles for Stable, High Activity Perovskite Cathodes for Solid Oxide Fuel Cells" | ✅ |

  ![Dense-only retrieval answering FePO4 for a cathode-materials question](assets/screenshot-dense-only-cathode-query.png)
  ![Hybrid retrieval answering Perovskite for the identical question](assets/screenshot-hybrid-cathode-query.png)

  This is the same underlying case as ADR-009's ranking analysis (the correct paper moved from
  4th to 2nd–3rd place), captured here at the level that actually matters to a user: it changed
  what answer they receive, not just where a document sits in an internal ranking. It is still not
  a complete fix — the improvement is a measured ranking shift, not a guarantee, and a harder or
  differently-phrased query could still retrieve the wrong domain.

### 7.3 Generation Quality

| Query | Result |
|---|---|
| *"What is considered a major reason for anode degradation in Ni-YSZ SOFCs?"* | **"Ni coarsening"** — correct, matches the known source paper exactly, self-verified ✅ |
| *"Who won the Super Bowl last year?"* | **"I don't know based on the given context."** — guardrail correctly refuses an off-domain question |

![Hybrid retrieval answer with verification badge and expandable sources](assets/screenshot-hybrid-answer.png)
![Expanded source card showing the retrieved passage and its arXiv link](assets/screenshot-sources-expanded.png)

### 7.4 Engineering Quality

- **20 automated tests** (10 pure-logic unit tests that always run in CI; 10 integration tests
  against local data/model artifacts that skip cleanly, rather than failing, when those
  gitignored artifacts aren't present).
- `ruff check .` clean; `uv sync --locked` reproducible; GitHub Actions CI on every push.
- 9 Architecture Decision Records documenting every non-trivial technical choice and every bug
  found along the way — including this project's own post-hoc audit (Section 8).

---

## 8. Work Completed & Work Remaining

### 8.1 Completed

All 9 phases (0–8) are implemented and merged to `main`, **plus** a follow-up audit-and-harden
pass that found and fixed real defects in the "done" system — worth presenting as evidence of
process rigor, not just feature completion:

- ✅ Corpus collection, NER training + inference, dense indexing, grounded generation, evaluation,
  UI, and CI (Phases 0–7) — all functioning and tested.
- ✅ Self-verification (Phase 8) — functioning as designed.
- ✅ **Fixed:** a metadata-key bug (`paper_id` vs. `source_id`) that silently made the ambiguity
  abstention feature dead code since Phase 8 was first written.
- ✅ **Fixed:** the NER-informed retrieval filter, which returned zero results on every query due
  to a ChromaDB operator misuse — root-caused, and replaced with a soft reranking boost
  (Reciprocal Rank Fusion + NER entity overlap) that measurably improves ranking without the
  fragility of a hard filter.
- ✅ **Fixed:** the ambiguity-abstention threshold, which — once the key bug above was fixed —
  was found (by testing, not assumption) to misfire on ordinary questions due to NER noise;
  recalibrated empirically against a 15-query sweep and documented as a known, honestly-reported
  limitation rather than a silently "working" feature.
- ✅ 10 new unit tests added for the fixed retrieval/ambiguity logic; CI hardened so tests that
  need local data/model artifacts skip cleanly instead of erroring on a fresh checkout.
- ✅ Housekeeping: removed unused dependencies, unified Python version declarations, made the
  corpus-fetching script's query reproducible, synced all documentation (README, RESULTS.md, 9
  ADRs) to match actual system behavior.

### 8.2 Remaining / Future Work

Reported honestly rather than inflated:

1. **NER precision is the single biggest constraint on the system's ceiling.** ~9% F1 (Phase 2/3)
   limits how much the NER-driven features can contribute today; the ambiguity-abstention feature
   in particular is currently designed to fail safe (rarely trigger) rather than reliably trigger,
   specifically because of this. Improving it — more training data or epochs, a larger backbone,
   or a hybrid rules+model extraction approach — is the highest-leverage remaining work.
2. **No formal, corpus-wide evaluation of hybrid retrieval vs. dense-only yet** — the current
   evidence is one deeply-verified before/after case (Section 7.2), not a full re-run of the
   20-question evaluation set under hybrid retrieval (the existing `evaluate_rag.py` measures
   dense-only hit-rate only). Worth doing next, especially since dense-only already hits 100% on
   that particular test set — a harder, more ambiguity-prone test set would be needed to show
   hybrid retrieval's benefit quantitatively, not just on the one known failure case.
3. **Generation quality is capped by Flan-T5-base's size** — it sometimes cannot synthesize an
   answer across multiple retrieved passages even when the right source material was retrieved
   (observed and documented in RESULTS.md's Phase 5 results). A larger instruction-tuned model is
   a natural next step if generation quality needs to improve further.
4. **Retrieval fusion parameters are reasonable defaults, not tuned.** The NER-boost weight and
   candidate pool size were sanity-checked against real data but not formally grid-searched.
5. **Two pieces of flagged-but-deliberately-untouched cleanup:** a bloated leftover model
   checkpoint directory (~1.3GB, local disk only, not part of the tracked project) and several
   stale, already-merged git branches on the remote — both low-risk, low-priority, and left for an
   explicit decision rather than done silently.

---

*Full technical detail, including the isolated bug repro and empirical calibration data behind
Sections 4, 7.1, and 8, is in [`docs/ADRs/ADR-009-ner-aware-retrieval-reranking.md`](ADRs/ADR-009-ner-aware-retrieval-reranking.md)
and [`RESULTS.md`](../RESULTS.md).*
