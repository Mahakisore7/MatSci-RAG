# ADR-005: Advanced RAG Stack — ChromaDB + BGE Embeddings over FAISS + MiniLM

**Date:** 2026-08-07
**Status:** Accepted
**Phase:** 4 — RAG Indexing

---

## Context

Phase 4 requires choosing two things:
1. An **Embedding Model** — a neural network that converts text into a list of numbers (a "vector") that captures the *meaning* of the text.
2. A **Vector Database** — a specialized database that stores those vectors and can quickly search for the most similar ones.

The original `docs/Phases/04_rag_indexing.md` specified using FAISS with `all-MiniLM-L6-v2`. The user asked if we could use **Cohere** (a cloud API) and also whether we should look for more advanced tools.

---

## Options Evaluated

### Option A: FAISS + all-MiniLM-L6-v2 (Original Plan)
- **FAISS** (Facebook AI Similarity Search): A C++ library from Meta that stores vectors in memory and searches them extremely quickly. However, it is NOT a database — it cannot store metadata (like paper titles, materials) alongside the vectors. You have to manage that separately in a JSON file.
- **all-MiniLM-L6-v2**: A small, fast embedding model released in 2021. It is easy to use and lightweight, but it was not designed specifically for retrieval tasks and is outperformed by newer models.

### Option B: Cohere Embed API + Pinecone
- **Cohere**: A cloud-based API that generates high-quality embeddings. Requires internet, an API key, and costs money after free tier limits.
- **Pinecone**: A managed cloud vector database. Also requires an account and has costs.
- **Why Rejected**: This makes the project non-reproducible for reviewers without accounts. Not suitable for a university project. 

### Option C: ChromaDB + BAAI/bge-small-en-v1.5 (CHOSEN)
- **ChromaDB**: A purpose-built, open-source vector database that runs locally on your laptop. It stores vectors AND metadata together, has built-in persistence to disk, and supports filtering queries by metadata fields.
- **BAAI/bge-small-en-v1.5**: The BGE ("Beijing Academy of AI General Embedding") model consistently ranks in the top tier of the MTEB Leaderboard (the official benchmark for embedding quality). It uses a special technique where you add an "instruction prefix" to text before embedding, which makes the resulting vectors significantly more accurate for retrieval.

---

## Decision

Use **Option C**: ChromaDB + BGE-small + LangChain framework.

LangChain acts as the "glue" connecting all parts (the embedding model, the chunking logic, the database). It is the industry-standard framework for building RAG applications.

---

## Detailed Technical Explanation of the Implementation

### Step 1: Loading data
We loaded two files:
- `data/sofc_corpus.jsonl` — 400 abstracts from arXiv. Each has an `id`, `title`, and `abstract`.
- `data/structured_facts.jsonl` — Our NER output. Each entry has a `source_id`, `materials`, `devices`, `experiments` list.

We merge them by matching `id` from the corpus to `source_id` from the facts.

### Step 2: Cleaning NER metadata
The NER model extracted noisy tokens like `"##oxide"` (BERT subword tokens), single-character tokens, and punctuation. Before storing these as metadata in ChromaDB, we filter them out with a function called `clean_entities()` that:
- Removes any token starting with `##`
- Removes any token shorter than 3 characters
- Deduplicates repeated tokens

### Step 3: Chunking
Abstracts average about 1,200 characters. If we store each abstract as a single chunk, a retrieval result will always return the entire abstract. This wastes context window and dilutes the relevance signal.

Instead, we use `RecursiveCharacterTextSplitter` with:
- `chunk_size=500` characters per chunk
- `chunk_overlap=50` characters overlap between adjacent chunks

The overlap is critical — if the phrase `"YSZ electrolyte degradation"` spans the boundary between chunk 3 and chunk 4, with overlap, it appears in BOTH chunks. Without overlap, it might be split in the middle, losing context.

**Result:** 400 papers → 1,330 chunks.

### Step 4: Embedding (The Most Important Step)
BGE has an unusual but important requirement: you must add a text prefix to documents before embedding them, and a different prefix to queries before embedding them.

- **Document prefix:** `"Represent this science document for retrieval: "` (added only at embedding time, NOT stored in the database)
- **Query prefix:** `"Represent this question for searching relevant documents: "` (added before every search query)

This prefix technique teaches the model that it is performing a retrieval task, which dramatically improves the accuracy of the resulting vectors.

**Critical Bug Fixed:** In the first version, we accidentally stored the document prefix IN the text that gets saved to ChromaDB. This meant when a result was retrieved, the user saw `"Represent this science document for retrieval: Solid oxide fuel cells enable..."` instead of just `"Solid oxide fuel cells enable..."`. We fixed this by storing clean abstract text in the database and only adding the prefix during the `encode_kwargs` call.

### Step 5: Storing in ChromaDB
Each chunk is stored in ChromaDB with:
- The raw abstract text (clean, no prefix)
- A vector of 384 numbers representing its meaning
- Metadata: `paper_id`, `title`, `materials`, `devices`, `experiments`

The full database is persisted to `data/chroma_db/` automatically by ChromaDB.

---

## Problems Encountered and Solutions

| Problem | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: langchain.text_splitter` | LangChain moved this to a separate package in newer versions | Changed import to `langchain_text_splitters` |
| `KeyError: 'arxiv_id'` | The corpus uses key `id`, not `arxiv_id` | Changed the load function to use `doc["id"]` |
| BGE prefix appearing in stored document text | We prepended the prefix to the text string before creating the Document | Store clean text in Document; pass prefix via `encode_kwargs["prompt"]` |
| `ValidationError: query_instruction Extra inputs not permitted` | Older parameter name, no longer accepted by `HuggingFaceEmbeddings` | Removed it; add prefix manually in the query call |
| `langchain-community` Chroma deprecation warning | LangChain sunsetted `langchain-community`, moving Chroma to its own package | Installed `langchain-chroma` and updated all imports |
| Disk space crash during NER training (previous phase) | `save_strategy="epoch"` was creating 60+ 400MB checkpoint folders | Added `save_total_limit=1` to keep only the best checkpoint |

---

## Consequences

- `data/chroma_db/` is added to `.gitignore` — it is a generated artifact and should be rebuilt by running `src/rag/build_index.py`.
- Phase 5 (generation) will call `Chroma(..., persist_directory="data/chroma_db/")` to load the persisted index — no rebuild needed.
- The `clean_entities()` function improves metadata quality but does not fix the underlying NER noise problem. That is a known limitation documented in Phase 3.
- Query 3 in the retrieval test ("Which cathode materials show high oxygen reduction activity?") returns off-domain battery papers. This is because our corpus contains some non-SOFC papers and the query does not explicitly mention SOFC. In Phase 5, the system prompt will instruct the LLM to answer only from SOFC-relevant context.
