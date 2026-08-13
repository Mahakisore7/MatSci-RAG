# 🔬 MatSci-RAG: Domain-Specific RAG for Materials Science

A complete, end-to-end Retrieval-Augmented Generation (RAG) system built for Materials Science literature, focusing on Solid Oxide Fuel Cells (SOFCs).

It demonstrates how to upgrade a "generic" RAG pipeline into a **Domain-Specific NER-RAG** system using a custom-trained MatSciBERT model to extract material entities and use them to intelligently search a literature corpus.

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
| 6 | Evaluation: retrieval hit-rate measured on 10 test questions | ✅ |
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

```
Query
  │
  ▼
Hybrid Retrieval  ──►  BM25 keyword match (exact terms)
  │               ──►  Dense vector search (ChromaDB / BGE-small-en)
  │               ──►  Fused via Reciprocal Rank Fusion, boosted by
  │                     NER entity overlap with the query (ADR-009)
  │
  ▼
detect_ambiguity()   ──►  abstain if an unusually large number of
                            distinct materials are found (rarely fires —
                            see ADR-009 on current NER precision limits)
  │
  ▼
Flan-T5-base         ──►  grounded answer from retrieved context only
  │
  ▼
verify_answer()      ──►  second LLM call: "does context support the answer?"
  │
  ▼
Streamlit UI         ──►  answer + ✅/⚠️ badge + source papers
```

---

## 📊 Key Results

| Metric | Result |
|---|---|
| MatSciBERT NER F1 | **Higher than BERT baseline** (see [RESULTS.md](RESULTS.md)) |
| RAG retrieval hit-rate (10 questions) | **100% — correct paper in top-5 every time** |
| Flan-T5 guardrails | Refuses off-topic questions ("Who won the Super Bowl?") |
| Phase 8 live test — anode degradation query | ✅ **"Ni coarsening is considered a major reason..."** |
| Phase 8 live test — vague electrolyte query | ✅ **"I don't know based on the given context"** (correct refusal) |

Full benchmark data → **[RESULTS.md](RESULTS.md)**

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

- **[RESULTS.md](RESULTS.md)** — NER F1, retrieval hit-rates, live test output, and bug table
- **[docs/ADRs/](docs/ADRs/)** — 9 Architecture Decision Records (every major technical choice)
- **[docs/sample_questions.md](docs/sample_questions.md)** — Ready-to-paste demo questions for the UI

---

> **CI Note:** GitHub Actions runs `ruff` linting and `pytest tests/ -v` on every push. `data/` and `models/` are gitignored (fine-tuned weights are ~400MB–1.7GB and corpus/index files are locally generated), so the 10 tests that depend on those artifacts (NER training/inference, retrieval, generation) skip cleanly on a fresh checkout via `pytest.mark.skipif`, rather than failing. The 10 tests in `tests/test_ner_utils.py` are pure-logic unit tests with no such dependency and always run.
