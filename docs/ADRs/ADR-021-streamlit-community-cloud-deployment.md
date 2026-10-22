# ADR-021: Deployment to Streamlit Community Cloud with Knowledge Base Bootstrapping

**Status:** Accepted  
**Date:** 2026-10-22  
**Phase:** Phase 11 — Production Deployment  
**Supersedes:** ADR-020 (Hugging Face Spaces SDK deprecated / Docker restricted to paid tiers)

---

## Context

We evaluated public hosting platforms for the MatSci-RAG application. While Hugging Face Spaces was initially considered in [ADR-020](ADR-020-hugging-face-spaces-deployment.md), Hugging Face officially deprecated its native Streamlit SDK in 2025 and placed Docker Spaces behind a paid subscription (Hugging Face Pro).

**Streamlit Community Cloud (`streamlit.io/cloud`)** offers free, unlimited public hosting specifically designed for Streamlit applications directly linked to GitHub repositories.

However, deploying MatSci-RAG to Streamlit Community Cloud faces two primary technical hurdles:
1. **Repository File Limits & Gitignored Data**: The ChromaDB vector store (`~202 MB` uncompressed) and parsed corpus JSONL files are gitignored and exceed GitHub's 100 MB single-file limit. A fresh `git clone` on Streamlit Cloud would boot with an empty database, causing queries to fail.
2. **Container RAM Constraints**: Streamlit Community Cloud provides 1 vCPU and ~1 GB of allocated memory (with hard termination ~3 GB). Heavy local models (`flan-t5`) or CUDA PyTorch wheels risk build timeouts or `OOMKilled` evictions.

---

## Decision

Deploy the application to **Streamlit Community Cloud** with the following production architecture:

1. **Remote Knowledge Base Bootstrapping (`src/core/data_bootstrapper.py`)**:
   - The offline-generated knowledge base (`data/chroma_db/`, `data/structured_facts.jsonl`, `data/sofc_corpus.jsonl`, `data/full_papers.jsonl`) is packaged into a compressed archive `data_bundle.tar.gz` (`120 MB`).
   - Hosted publicly as a GitHub Release asset or Hugging Face Dataset file.
   - On container startup in [src/ui/app.py](../../src/ui/app.py), `ensure_data_ready()` checks if `data/chroma_db` is present. If missing (as on a clean cloud container), it automatically downloads and extracts the archive with an in-UI status indicator, then deletes the archive to conserve container disk space.
   - On local development, `ensure_data_ready()` returns immediately in `0.0ms` without network overhead.

2. **CPU-Optimized Pip Build (`requirements.txt`)**:
   - Uses `--extra-index-url https://download.pytorch.org/whl/cpu torch>=2.2.0` to pull the lightweight (~180 MB) CPU wheel rather than the ~2.5 GB CUDA wheel, completing builds in under 2 minutes.

3. **API-Driven Cloud Generation**:
   - Relies on Groq (`qwen/qwen3.8-27b` generation and `openai/gpt-oss-20b` verification), keeping resident container memory under 800 MB (well within Streamlit Cloud's memory ceiling).
   - `st.secrets` are automatically mapped to `os.environ` on boot.

4. **Root Entrypoint Compatibility (`app.py`)**:
   - Provides a standard root [app.py](../../app.py) entrypoint that forwards to `src/ui/app.py`.

---

## Consequences

**Positive**
- **100% Free Public Hosting**: Zero infrastructure costs on Streamlit Community Cloud.
- **Preserves Full UI**: Retains all custom styling, CSS design tokens, multi-page router, analytics dashboard, and interactive PyVis knowledge graph.
- **Fast Startup**: Pre-indexed ChromaDB vectors download once on boot; zero re-embedding required.
- **No Git Bloat**: The GitHub repository stays lightweight and clean (<10 MB of code).

**Negative / Accepted Trade-offs**
- Initial cloud container cold-start takes ~10–15 seconds to download the 120 MB compressed knowledge base bundle.
- Requires publishing the `data_bundle.tar.gz` file to a public release or cloud bucket once when the index updates.
