# 🔬 MatSci-RAG: Domain-Specific RAG for Materials Science

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](.python-version)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-de5fe9)](https://docs.astral.sh/uv/)
[![Linted with ruff](https://img.shields.io/badge/linted-ruff-261230)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Mahakisore7/MatSci-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahakisore7/MatSci-RAG/actions/workflows/ci.yml)

A complete, end-to-end Retrieval-Augmented Generation (RAG) system built for Materials Science literature, focusing on Solid Oxide Fuel Cells (SOFCs).

It demonstrates how to upgrade a "generic" RAG pipeline into a **Domain-Specific NER-RAG** system using a custom-trained MatSciBERT model to extract material entities and use them to intelligently search a literature corpus.

**Contents:** [Status](#-project-status-v100--complete) · [Setup & Run](#-setup--run) · [Architecture](#-architecture) · [Key Results](#-key-results) · [Screenshots](#-screenshots) · [Project Structure](#-project-structure) · [Documentation](#-documentation)

---

## 🚀 Project Status: v1.0.0 — Complete

All 9 phases (0–8) are implemented, tested, and committed.

| Phase | What Was Built | Status |
|---|---|---|
| 0 | Environment setup: `uv`, `ruff`, pre-commit, logging, CI | ✅ |
| 1 | Data ingestion: 400 SOFC abstracts scraped from ArXiv | ✅ |
| 2 | NER training: MatSciBERT fine-tuned on SOFC-Exp corpus | ✅ |
| 3 | NER inference: structured facts extracted from 400 papers | ✅ |
| 4 | Vector indexing: 1,330 chunks embedded into ChromaDB | ✅ |
| 5 | RAG generation: grounded answers via `google/flan-t5-base` | ✅ |
| 6 | Evaluation: retrieval hit-rate measured on 20 test questions | ✅ |
| 7 | Streamlit UI + GitHub Actions CI | ✅ |
| 8 | Hybrid retrieval (BM25), self-verification, uncertainty abstention | ✅ |

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) package manager

### Install
```bash
git clone https://github.com/Mahakisore7/MatSci-RAG.git
cd MatSci-RAG
uv sync
```

### Run the Chat UI
```bash
uv run streamlit run app.py
```
Opens at `http://localhost:8501`. Use the sidebar to toggle between **Hybrid** (Phase 8) and **Dense Only** (Phase 5) retrieval.

### Run Tests
```bash
uv run pytest tests/ -v
```
All 20 tests pass locally (with `data/`/`models/` populated). In CI, 10 pure-logic unit tests
always run; the 10 that depend on local data/model artifacts skip cleanly instead of failing.

---

## 🏗️ Architecture

The system has two halves that run at different times: an **offline indexing pipeline** (run once,
whenever the corpus changes) and a **live query pipeline** (run on every question). Both use the
same NER facts, extracted once in Phase 3.

### Offline: Corpus → NER Facts → Vector Index

```mermaid
flowchart LR
    SOFCEXP[("SOFC-Exp<br/>876 labeled sentences")] --> TRAIN["Fine-tune NER<br/>MatSciBERT vs BERT-base"]
    ARXIV["arXiv corpus<br/>400 SOFC abstracts"] --> INFER["Run trained NER model"]
    TRAIN -->|"MatSciBERT wins<br/>F1 0.0915 vs 0.0816"| INFER
    INFER --> FACTS["structured_facts.jsonl<br/>materials · devices · experiments"]
    ARXIV --> CHUNK["Chunk text<br/>500 chars / 50 overlap"]
    FACTS -->|"joined as<br/>chunk metadata"| CHUNK
    CHUNK --> EMBED["Embed<br/>BAAI/bge-small-en-v1.5"]
    EMBED --> CHROMA[("ChromaDB<br/>1,330 vectors")]

    style TRAIN fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    style CHROMA fill:#eff6ff,stroke:#3b82f6,color:#1e293b
```

### Live: Question → Grounded, Verified Answer

```mermaid
flowchart TB
    Q["User question"] --> H["hybrid_retrieve()<br/>see zoom-in below"]
    CHROMA2[("ChromaDB")] --> H
    H --> AMB{"detect_ambiguity()<br/>too many distinct<br/>materials in results?"}
    AMB -->|"rare — see ADR-009<br/>on NER precision limits"| ABST["Abstain:<br/>ask user to narrow"]
    AMB -->|"usual case"| PROMPT["build_prompt()<br/>context-only instruction"]
    PROMPT --> GEN["Flan-T5-base<br/>generate()"]
    GEN --> VER["verify_answer()<br/>2nd LLM call:<br/>'does context support this?'"]
    VER --> UI["Streamlit UI<br/>answer + ✅/⚠️ badge + sources"]
    ABST --> UI

    style AMB fill:#fef3c7,stroke:#d97706,color:#451a03
    style UI fill:#f0fdf4,stroke:#22c55e,color:#052e16
```

### Zoom-in: `hybrid_retrieve()` — Why It's NER-*Aware*, Not NER-*Gated*

An earlier version hard-filtered dense search to papers whose NER tags matched certain keywords —
which sounds reasonable until you check the data: 68% of papers have no extracted `devices` field
at all, so a hard filter would silently discard most of the corpus on every query. ADR-009 replaced
it with a **soft boost** that can improve ranking but never zero out results:

```mermaid
flowchart LR
    Q["Query"] --> D["Dense search<br/>BGE-small, pool = k×4"]
    Q --> B["BM25 search<br/>keyword, pool = k×4"]
    D --> RRF["Reciprocal Rank Fusion<br/>score = Σ 1/(60 + rank)"]
    B --> RRF
    FACTS2["Per-paper NER facts"] -->|"+0.02 per cleaned entity<br/>mentioned in the query"| RRF
    RRF --> SORT["Sort + dedupe<br/>by paper_id + chunk"]
    SORT --> TOPK["Top-k results"]

    style RRF fill:#fff7ed,stroke:#f97316,color:#431407
```

*Full derivation, the `$contains`-operator bug it replaced, and the 68%-empty-field evidence are in*
*[ADR-009](docs/ADRs/ADR-009-ner-aware-retrieval-reranking.md).*

---

## 📊 Key Results

| Metric | Result |
|---|---|
| MatSciBERT NER F1 | **Higher than BERT baseline** (see [RESULTS.md](RESULTS.md)) |
| RAG retrieval hit-rate (20 questions) | **100% — correct paper in top-5 every time** |
| Flan-T5 guardrails | Refuses off-topic questions ("Who won the Super Bowl?") |
| Phase 8 live test — anode degradation query | ✅ **"Ni coarsening is considered a major reason..."** |
| Phase 8 live test — vague electrolyte query | ✅ **"I don't know based on the given context"** (correct refusal) |

Full benchmark data → **[RESULTS.md](RESULTS.md)**

---

## 🖼️ Screenshots

**Anode-degradation query, Hybrid retrieval — grounded answer, self-verified, with expandable sources:**

![Hybrid retrieval answer with verification badge](docs/assets/screenshot-hybrid-answer.png)

![Expanded source cards showing retrieved passages and arXiv links](docs/assets/screenshot-sources-expanded.png)

**The same query, two retrieval strategies — this is the Phase 8 hybrid-retrieval improvement (ADR-009) actually happening live, not just measured in the background:**

*"Which cathode materials show high oxygen reduction activity?" under* **Dense Only** *— the known failure mode (RESULTS.md, Phase 5): retrieval pulls an off-domain battery-cathode paper, and the model answers from it:*

![Dense-only retrieval answering FePO4 — an off-domain battery cathode material](docs/assets/screenshot-dense-only-cathode-query.png)

*The identical question under* **Hybrid (Dense + BM25 + NER)** *— retrieval now surfaces the genuinely relevant SOFC paper, and the answer changes accordingly:*

![Hybrid retrieval answering Perovskite — the correct SOFC-domain cathode material](docs/assets/screenshot-hybrid-cathode-query.png)

---

## 📁 Project Structure

```
MatSci-RAG/
├── app.py                          ← Streamlit chat UI (entry point)
├── src/
│   ├── core/
│   │   ├── config.py
│   │   └── logging_config.py
│   ├── ingestion/
│   │   └── fetch_corpus.py
│   ├── ner/
│   │   ├── prepare_ner_data.py
│   │   ├── train_ner.py
│   │   └── run_ner_inference.py
│   └── rag/
│       ├── build_index.py
│       ├── rag_query.py            ← core pipeline + self-verification
│       ├── ner_utils.py            ← shared NER cleaning, ambiguity check, RRF+boost reranking
│       └── hybrid_retrieval.py     ← BM25 + dense, fused via ner_utils (Phase 8 / ADR-009)
├── scripts/
│   ├── evaluate_rag.py
│   ├── quick_retrieval_test.py
│   └── verify_gpu.py
├── tests/                          ← 20 pytest tests (10 pure unit tests always run in
│                                      CI; 10 artifact-dependent ones skip cleanly without
│                                      local data/models instead of erroring)
├── data/                           ← gitignored (corpus, chroma_db, facts)
├── models/                         ← gitignored (fine-tuned weights ~400MB)
├── docs/
│   ├── ADRs/                       ← 9 Architecture Decision Records
│   ├── Phases/                     ← Step-by-step build guide (Phases 0–8)
│   ├── results/                    ← Phase-level explanation docs
│   └── sample_questions.md         ← Ready-to-paste demo questions
├── .streamlit/config.toml          ← Disables file watcher (Windows fix)
├── .github/workflows/ci.yml        ← Linting + smoke tests on every push
├── RESULTS.md                      ← Full benchmark evidence
├── pyproject.toml
└── uv.lock
```

---

## 📝 Documentation

- **[docs/MatSci-RAG-Review.md](docs/MatSci-RAG-Review.md)** — full project review: problem statement, literature review, research gap, methodology, and results, written for a mentor/committee audience
- **[RESULTS.md](RESULTS.md)** — NER F1, retrieval hit-rates, live test output, and bug table
- **[docs/ADRs/](docs/ADRs/)** — 9 Architecture Decision Records (every major technical choice)
- **[docs/sample_questions.md](docs/sample_questions.md)** — Ready-to-paste demo questions for the UI

---

> **CI Note:** GitHub Actions runs `ruff` linting and `pytest tests/ -v` on every push. `data/` and `models/` are gitignored (fine-tuned weights are ~400MB–1.7GB and corpus/index files are locally generated), so the 10 tests that depend on those artifacts (NER training/inference, retrieval, generation) skip cleanly on a fresh checkout via `pytest.mark.skipif`, rather than failing. The 10 tests in `tests/test_ner_utils.py` are pure-logic unit tests with no such dependency and always run.
