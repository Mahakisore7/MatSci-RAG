# ADR-020: Deployment to Hugging Face Spaces

**Status:** Accepted  
**Date:** 2026-10-22  
**Phase:** Phase 11 — Production Deployment  

---

## Context

A publicly accessible cloud deployment is required to showcase the MatSci-RAG research assistant, evaluation analytics, and interactive knowledge graph without requiring users to set up a local Python environment or download 400 papers.

Two primary cloud hosting options were evaluated:
1. **Streamlit Community Cloud (`streamlit.io/cloud`)**:
   - Free tier resource limit: **1 vCPU and 1 GB RAM** (hard eviction ceiling ~3 GB).
   - No GPU support.
   - Standard Git file limit of 100 MB without native Git LFS integration.
   - Build environment downloads default CUDA PyTorch packages if unconstrained, exhausting build disk/timeouts.
   - Resident memory for our pipeline (`BGE-small` dense embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder, ChromaDB vector store, BM25 in-memory index) approaches 900 MB–1.2 GB, and any accidental invocation of local generator fallback (`flan-t5-base` or `large`) causes an immediate `OOMKilled` termination.

2. **Hugging Face Spaces (`huggingface.co/spaces`)**:
   - Free tier resource limit: **2 vCPUs and 16 GB RAM** (16× the memory of Streamlit Cloud).
   - 50 GB free persistent container disk space.
   - Native Git LFS support with generous bandwidth for hosting the ChromaDB vector database (~202 MB) directly inside the space.
   - First-class support for the Streamlit SDK and Hugging Face Hub ecosystems.
   - Option for low-cost or free hardware upgrades (e.g., T4/A10G GPU) if local LLMs are required.

---

## Decision

Deploy the MatSci-RAG Streamlit application to **Hugging Face Spaces** using the Streamlit SDK:

1. **Space Metadata & Configuration**:
   - Embed standard Hugging Face YAML metadata in repository documentation or Space configuration (`sdk: streamlit`, `sdk_version: "1.42.0"`, `app_file: src/ui/app.py`).
   - Create a top-level bridge [app.py](../../app.py) to support default root detection on platforms that require it.

2. **Optimized Dependency Management (`requirements.txt`)**:
   - Provide a CPU-optimized `requirements.txt` pointing to the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu torch`) so container builds remain fast (~1–2 minutes) and disk-light (~180 MB instead of ~2.5 GB for CUDA wheels).

3. **Secret & Environment Bridging**:
   - Bridge Streamlit Secrets (`st.secrets`) into `os.environ` on startup in [src/ui/app.py](../../src/ui/app.py) so credentials like `GROQ_API_KEY` and `HF_TOKEN` configured in the Space settings dashboard are automatically available to all downstream modules (`generation.py`, `query_expansion.py`, `rag_query.py`).

4. **Runtime Knowledge Base Provisioning**:
   - Support direct repository tracking via Git LFS for `data/chroma_db/`, `data/structured_facts.jsonl`, and `data/sofc_corpus.jsonl`.
   - Provide a startup check (`ensure_data_present`) that verifies knowledge base integrity upon container initialization.

5. **Deployment Tooling**:
   - Provide [scripts/deploy_to_hf.py](../../scripts/deploy_to_hf.py) and a GitHub Actions workflow template for automated sync from GitHub to Hugging Face Spaces.

---

## Consequences

**Positive**
- **16 GB RAM Headroom**: Easily accommodates in-memory BM25 index, ChromaDB, dense embeddings, and cross-encoder re-ranking without memory pressure or container eviction.
- **Fast Startup**: CPU-only wheels and pre-indexed ChromaDB load in seconds.
- **Unified Domain URL**: Generates a clean, shareable URL (`https://huggingface.co/spaces/<username>/MatSci-RAG`) with integrated embedding support.
- **Native Hub Integration**: Direct interoperability with Hugging Face models and datasets.

**Negative / Accepted Trade-offs**
- The repository requires maintaining a `requirements.txt` alongside `pyproject.toml` specifically for pip-based cloud builders like Hugging Face Spaces.
- Git LFS bandwidth on external remotes must be managed when syncing large vector store revisions.
