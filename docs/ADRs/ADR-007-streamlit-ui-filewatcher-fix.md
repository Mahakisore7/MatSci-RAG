# ADR-007: Streamlit for UI and `fileWatcherType=none` for Windows Compatibility

**Date:** 2026-08-12
**Status:** Accepted
**Phase:** 7 — Streamlit UI + GitHub Actions CI

---

## Context

After Phases 1–6, the entire RAG pipeline was fully operational but only accessible via Python scripts in a terminal. To present the project to mentors and non-technical reviewers, a browser-based interface was required. Additionally, Phase 0 had only stubbed out a placeholder CI workflow (`echo "Tests will go here"`), which needed to be replaced with a real lint-and-test pipeline.

Two decisions were made in this phase that are worth documenting:

1. **Which UI framework to use**
2. **How to fix the Streamlit + Transformers file-watcher crash on Windows**

---

## Decision 1: Use Streamlit for the UI

### Options Considered
- **Streamlit** — Single-file Python web apps, designed for data science demos
- **Gradio** — Similar to Streamlit, slightly more ML-focused
- **Flask/FastAPI + HTML** — Full control, but requires writing frontend HTML/JS/CSS

### Decision: Streamlit

Streamlit was chosen for the following reasons:
- The entire interface (`app.py`) can be implemented in under 60 lines of pure Python with no HTML or JavaScript
- It ships with a native `st.chat_message` and `st.chat_input` API that creates a professional, ChatGPT-style interface without any extra work
- It requires no separate frontend/backend architecture — the existing `src/rag/rag_query.py` can be imported directly
- The `st.expander()` widget provides a native UI for showing source papers as collapsible dropdowns, which directly supports the project's core requirement of making AI answers verifiable

### Consequence
Running with `uv run streamlit run app.py` is all that is needed.

---

## Decision 2: Disable Streamlit's File Watcher via `.streamlit/config.toml`

### The Problem
When Streamlit starts, it scans all imported Python modules to enable its live-reload feature. Our project imports the `transformers` library, which contains hundreds of sub-modules including vision model processors (e.g., `detr`, `deepseek_vl`) that internally import `torchvision`. Since `torchvision` is not installed in this project (we only use text models), the file-watcher crashed with:

```
ModuleNotFoundError: No module named 'torchvision'
```

This is **not a bug in our code**. It is a known consequence of Streamlit's eager module inspection combined with Hugging Face `transformers`' large, multi-domain architecture.

### Options Considered
- **Install `torchvision`** — This would add a ~500MB+ dependency for a feature we don't need. Rejected.
- **Pass `--server.fileWatcherType none` as a CLI flag** — Works, but forces every user to remember the flag. Fragile.
- **Create `.streamlit/config.toml`** — Persists the setting in the project permanently. One-time fix that all users benefit from automatically.

### Decision: Create `.streamlit/config.toml`

The configuration file `.streamlit/config.toml` was added to the project with the following content:

```toml
[server]
fileWatcherType = "none"
```

This disables the file watcher once for all future runs, making `uv run streamlit run app.py` the simple, correct command with no extra flags needed.

### Consequence
Live-reload will not work in development (a minor tradeoff). For this project, live-reload is not needed since UI changes are infrequent and a manual server restart is trivial.

---

## Decision 3: `retrieve()` Helper Function Added to `rag_query.py`

### Context
The new smoke test (`tests/test_pipeline.py`) needed to call a pure retrieval function (without loading the LLM) to test quickly in CI. The existing `answer_query()` function loads the full `flan-t5-base` model which takes 30+ seconds — too slow for a smoke test.

### Decision
A new `retrieve(query, k)` function was added to `src/rag/rag_query.py` that performs only the vector search step and returns results as a list of plain dicts (containing `text`, `title`, and `source_id`). This is a clean, testable public API that does not trigger LLM loading.

---

## Outcome
- `uv run streamlit run app.py` launches a working Materials Science literature chatbot in the browser
- The chatbot correctly answered the question *"What is considered a major reason for anode degradation in Ni-YSZ SOFC anodes?"* with *"Ni coarsening"* — the exact correct answer — using the GPU
- `uv run pytest tests/` passes all 10 tests (1 passed + 9 from previous phases)
- GitHub Actions CI workflow is live and real
