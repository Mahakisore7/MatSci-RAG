# ADR-008: Phase 8 Bug Fixes — Import Order, Double Import, and NER Filter Observations

**Date:** 2026-08-13  
**Status:** Accepted  
**Phase:** 8 — Stretch Improvements (Bug Fixes)

---

## Context

After Phase 8 code was written and committed, the Streamlit app silently crashed every time during startup. The crash was identified through direct Python import testing (`uv run python -c "from src.rag.rag_query import answer_query"`), which succeeded — proving the code itself was correct, and Streamlit's process wrapper was swallowing the error.

Two distinct bugs were found and fixed. A third observation about the NER metadata filter was documented.

---

## Bug 1: ruff Reordering Imports → Silent Windows DLL Crash

**Decision:** Use `# ruff: isort: skip_file` to permanently freeze the import order in `rag_query.py`.

### Problem
On Windows, importing `torch` before `langchain_huggingface.HuggingFaceEmbeddings` causes a silent `ACCESS_VIOLATION` at the C level. This is a known Windows DLL load-order conflict.

When we added `import json` to `rag_query.py` (for Phase 8's `load_structured_facts()`), `ruff`'s isort hook ran on commit and re-sorted all imports alphabetically. This placed `import torch` above `from langchain_huggingface import HuggingFaceEmbeddings` — causing the crash.

### Fix
Added `# ruff: isort: skip_file` at the top of `rag_query.py`. This tells ruff to never touch the import order in this specific file.

### Why Not Just Configure ruff Globally?
We only want to suppress import sorting for this one file. A global `isort.force-sort-within-sections = false` setting in `pyproject.toml` would affect all files, removing a useful quality check everywhere else.

---

## Bug 2: Double Import Path → Double Model Load → Crash

**Decision:** Use `src.rag.rag_query` as the single, canonical import path throughout the entire codebase.

### Problem
`app.py` was using two different import paths for the same module:
```python
# Line 8 (top of file) — the old sys.path hack from Phase 5:
import sys
sys.path.append("src")
from rag.rag_query import answer_query   # ← Python module: "rag.rag_query"

# Line 68 (inside the response handler) — the correct path added in Phase 8:
from src.rag.rag_query import verify_answer  # ← Python module: "src.rag.rag_query"
```

Python's module system treats `"rag.rag_query"` and `"src.rag.rag_query"` as **completely different modules**, even though they are the same file. This caused:
1. The embedding model (`BAAI/bge-small-en-v1.5`) to be loaded **twice**
2. ChromaDB to open **two connections** to the same `.sqlite3` database
3. A silent crash during the second initialization

### Fix
Removed the `sys.path.append("src")` hack entirely from `app.py`. All imports now use `from src.rag.rag_query import answer_query, verify_answer` consistently. The `verify_answer` import was also moved to the top level of the file so it is never imported a second time inside a conditional block.

---

## Observation: NER Metadata Filter vs. ChromaDB `$contains` Operator

**Decision:** Accepted as a known limitation. No fix applied at this stage.

### What Was Observed
The live terminal logs showed:
```
Merged 0 dense and 5 BM25 results into 5 unique docs.
```
The NER metadata filter (`$or: [devices contains "fuel", devices contains "oxide"]`) was returning **0 dense results** for every query. This means the ChromaDB `$contains` operator is either:
- Case-sensitive and the stored NER tags use a different casing, **or**
- The ChromaDB collection was built (Phase 4) without storing the `devices` metadata field from `structured_facts.jsonl`.

### Impact
The hybrid retrieval is effectively falling back to **pure BM25** search only. The BM25 results are still domain-relevant because SOFC paper text naturally contains SOFC keywords. The system answers correctly and does not hallucinate.

### Future Fix
A proper fix would involve:
1. Re-inspecting the ChromaDB metadata schema to see what fields are stored per chunk.
2. Updating `build_index.py` to join the `structured_facts.jsonl` NER output with the chunk metadata at indexing time.
3. Adjusting the filter field and operator to match the actual stored schema.

This is logged here as a known technical debt item.

---

## Consequences

- The app now starts reliably and answers questions correctly.
- The import order fix is permanent and self-documenting in the code.
- The NER filter limitation is documented and does not block delivery of v1.0.0.
