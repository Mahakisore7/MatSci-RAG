# ADR-005: Advanced RAG Stack — ChromaDB + BGE Embeddings over FAISS + MiniLM

**Date:** 2026-08-07
**Status:** Accepted

---

## Context

Phase 4 requires choosing a Vector Database and Embedding Model for the RAG pipeline. The original `docs/Phases/04_rag_indexing.md` plan specified FAISS (Meta's vector index library) with `sentence-transformers/all-MiniLM-L6-v2`.

After evaluating alternatives, we decided to upgrade both choices.

## Decision

We will use the following **Advanced Stack** for Phase 4:

| Component | Original Plan | Chosen (Advanced) |
|---|---|---|
| **Vector Database** | FAISS | **ChromaDB** |
| **Embedding Model** | all-MiniLM-L6-v2 | **BAAI/bge-small-en-v1.5** |
| **Framework** | Raw sentence-transformers | **LangChain** |
| **Chunking** | Basic splitter | **RecursiveCharacterTextSplitter (overlap=50)** |

## Rationale

### Why ChromaDB over FAISS?
FAISS is a raw numerical index library — it stores vectors efficiently but is not a database. It has no concept of metadata, persistence, or filtering. ChromaDB is a purpose-built vector database that:
- Stores vectors **and** metadata in the same collection (critical for our NER metadata filtering).
- Persists to disk automatically between sessions.
- Supports `where` filtering on metadata, enabling hybrid retrieval (vector similarity + NER tag filter) in a single query.

### Why BGE over MiniLM?
The `BAAI/bge-small-en-v1.5` model consistently ranks in the **top tier of the MTEB Leaderboard** (the standard benchmark for embedding models). It is trained with a specific technique (bidirectional attention) that produces more accurate semantic representations for retrieval tasks. `all-MiniLM-L6-v2` is an older, much simpler model that is primarily designed for speed over quality.

### Why Overlapping Chunks?
Using a `chunk_overlap=50` means that if a key material name or experimental condition spans the boundary between two chunks, it will appear in both, ensuring no context is ever lost. This is an industry-standard best practice for RAG pipelines.

### Why LangChain?
LangChain provides production-grade wrappers around ChromaDB and sentence-transformers that handle boilerplate like batching, tokenization, and persistence. It is the industry-standard framework for all RAG applications and its use here makes Phase 5 (generation) trivial to wire up.

## Consequences

- New dependencies required: `chromadb`, `langchain`, `langchain-community`, `langchain-huggingface`.
- The database will be stored at `data/chroma_db/` (gitignored).
- Phase 5 will use the LangChain `Chroma` object directly as a retriever — no extra adapter code needed.
- BGE model requires a ~33MB download on first run (cached automatically thereafter).
