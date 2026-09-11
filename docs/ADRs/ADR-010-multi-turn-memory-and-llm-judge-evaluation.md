# ADR-010: LLM-Judge Evaluation + Multi-Turn Memory

**Status:** Accepted  
**Date:** 2026-09-10  
**Phase:** Phase 9 Extension  
**Branch:** `feature/multi-turn-memory`, `feature/ragas-evaluation`

---

## Context

Two improvements were implemented as part of the Phase 9 extension pass:

### Part A — Multi-Turn Conversation Memory

The original `app.py` and `rag_query.py` treated every user query as fully independent. Each call to `answer_query()` built a prompt that contained only the current question and retrieved context — no memory of what was previously asked or answered. This meant follow-up questions like "What temperature range do they operate at?" would fail because the model had no idea what "they" referred to.

### Part B — Automated RAG Quality Evaluation

The only existing quality measurement was a manual 20-question retrieval hit-rate test (`scripts/evaluate_rag.py`, k=5, dense only). This measures *whether the right paper was retrieved* but not:
- Whether the generated answer was faithful to the retrieved context
- Whether the generated answer was relevant/useful to the user
- Whether the retrieved chunks were actually useful for answering the question

The industry standard for measuring these properties in RAG systems is [RAGAS](https://github.com/explodinggradients/ragas), an open-source evaluation framework with dedicated metrics for Faithfulness, Answer Relevancy, and Context Precision.

---

## Decision

### Part A — Multi-Turn Memory via Rolling History Window

**Chosen approach:** Pass the last N conversation turns (as a list of `{role, content}` dicts already maintained in `st.session_state.messages`) to `answer_query()`, which forwards them to `build_prompt()`. The prompt prepends a "Recent Conversation History" section before the retrieved context.

**Alternatives considered:**

| Approach | Why Rejected |
|---|---|
| Pass full unlimited history | Flan-T5-base has a hard 1024-token limit. Full history would overflow it, silently truncating the retrieved context — which is more important than old history. |
| Vector-store-based memory (store past answers in ChromaDB, retrieve them alongside literature) | Overkill for this scale and would complicate the architecture significantly. Reserve for a future "long-term memory" extension. |
| LangChain `ConversationBufferMemory` | LangChain's memory APIs are coupled to the chain/pipeline pattern we already moved away from (see ADR-006). Keeping the raw PyTorch approach is more stable. |

**Implementation detail:** The window is capped at the last **6 messages (3 turns)** — small enough to never overflow the 1024-token limit even with large retrieved contexts, large enough to cover natural follow-up conversational depth.

A **"New Chat" button** was added to the sidebar to explicitly clear history, so the user can always start a fresh independent session.

---

### Part B — Custom Groq-Based LLM Judge (Not RAGAS)

**Chosen approach:** A custom evaluator script (`scripts/evaluate_ragas.py`) that calls the **Groq API** with `groq/compound-mini` as the judge LLM, scoring three metrics independently: Faithfulness, Answer Relevancy, and Context Precision.

**Why NOT the RAGAS library directly:**

RAGAS `0.4.x` and `0.3.x` both have a **broken hardcoded import** at startup:
```python
# ragas/llms/base.py line 12
from langchain_community.chat_models.vertexai import ChatVertexAI
```
This module (`langchain_community.chat_models.vertexai`) was removed from `langchain-community` in versions `>=0.4`. Since this project requires `langchain-community>=0.4.x` for other components, RAGAS `0.3.x`/`0.4.x` is incompatible — it throws `ModuleNotFoundError` on any import. RAGAS `0.1.x` requires `langchain-community<0.3`, which conflicts in the opposite direction. There is no version of RAGAS that resolves cleanly with our current lockfile.

**Decision:** Build a lightweight custom evaluator that measures the same three core metrics using direct Groq API calls, without any RAGAS dependency. This is actually cleaner — the scoring logic is explicit, auditable Python rather than hidden inside a library abstraction.

**Groq model selection:**
- `llama-3.1-8b-instant` — **decommissioned** (404)
- `llama3-8b-8192` — **decommissioned** (400)
- `groq/compound-mini` — ✅ available on the provided API key, free tier

---

## Results

### Multi-Turn Memory
- The `build_prompt()` signature now accepts an optional `chat_history: list = None` parameter — fully backward-compatible, all existing tests pass unchanged.
- The "New Chat" sidebar button resets `st.session_state.messages` and calls `st.rerun()`.

### Evaluation Scores (7 SOFC benchmark questions)

| Metric | Score |
|---|---|
| **Faithfulness** | **1.000** |
| **Answer Relevancy** | 0.214 |
| **Context Precision** | 0.200 |

**Faithfulness = 1.0** is the core design goal: Flan-T5 never fabricates. It either extracts directly from the retrieved text or outputs "I don't know based on the provided context." This is a direct result of the strict grounding prompt (ADR-006).

**Relevancy and Precision are low** for a structurally honest reason: 4 of 7 benchmark questions require information that is only found in the *Methods/Results sections of full papers* — which are not in our abstracts-only corpus. This is not a retrieval bug; it is a corpus scope limitation. The evaluation provides quantitative motivation for the planned full-paper ingestion extension.

**Results file:** `docs/results/ragas_evaluation.csv`

---

## Consequences

- `answer_query()` signature changed from `(query, use_hybrid)` to `(query, use_hybrid, chat_history=None)` — backward-compatible, default is `None` (same behaviour as before).
- `build_prompt()` signature changed similarly — backward-compatible.
- `scripts/evaluate_ragas.py` requires `GROQ_API_KEY` in `.env`. If not present, the script will error on the first Groq call.
- `langchain-groq` added as a project dependency.
- The evaluation script can be re-run at any time to measure regression/improvement as corpus and retrieval improve.
