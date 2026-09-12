# ADR-011: LLM-Based Query Expansion

**Status:** Accepted  
**Date:** 2026-09-12  
**Phase:** Phase 9 Extension  
**Branch:** `feature/query-expansion`

---

## Context

During the RAGAS-style evaluation in Phase 9 (see ADR-010), we identified that **Context Precision** was consistently low (0.200) for specific domain questions. 

A key root cause of this low precision is the mismatch between a user's conversational query and the dense, technical language used in the academic literature corpus. For example, a user asking "How does grain size affect the conductivity of ceria?" might miss papers that discuss "nanocrystalline CeO2" or "microstructural effects on ionic transport" because the embeddings and BM25 keywords don't perfectly align with the user's natural language phrasing.

Furthermore, with the introduction of Multi-Turn Memory (ADR-010), users can ask follow-up questions with pronouns (e.g., "What temperature do they operate at?"). Searching a vector database with unresolved pronouns yields very poor retrieval results.

---

## Decision

We implemented an **LLM-Based Query Expansion module** (`src/rag/query_expansion.py`) that optionally rewrites the user's input before it hits the retrieval pipeline.

**Chosen approach:**
We use LangChain interfacing with the Groq API (specifically the `groq/compound-mini` model, identical to our evaluator) to rewrite the query. 
The LLM is given the conversation history and the user's current question, and instructed to:
1. Resolve any ambiguous pronouns using the chat history.
2. Expand the query with domain-specific synonyms and keywords optimized for semantic search.
3. Output *only* the rewritten search string (no conversational filler).

**Integration:**
- The expansion is exposed as an optional toggle in the Streamlit sidebar (`Enable LLM Query Expansion`).
- When enabled, `app.py` passes `expand_q=True` to `answer_query()`.
- The expanded query is used *strictly for retrieval* against ChromaDB and BM25. The original user query is still passed to the generation LLM (Flan-T5) to ensure the final answer naturally addresses the user's exact phrasing.

---

## Consequences

**Positive:**
- **Improved Context Precision:** The retrieval step now has access to a much richer set of keywords (synonyms, chemical formulas) increasing the likelihood of pulling the correct technical chunks.
- **True Conversational Memory:** Follow-up questions with pronouns are correctly resolved into standalone search queries before hitting the vector database, preventing catastrophic retrieval failures on conversational follow-ups.

**Negative / Trade-offs:**
- **Latency:** Query expansion adds an extra network call to the Groq API before retrieval begins, adding ~500ms to 1s to the response time. This is why it is exposed as an optional toggle rather than forced on.
- **Dependency:** It relies on an external API (`GROQ_API_KEY`). If the key is missing or rate-limited, the system gracefully falls back to using the original unexpanded query, ensuring the app remains usable offline (assuming local models for everything else).
