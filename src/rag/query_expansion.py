"""
Phase 9: Query Expansion / Rewriting module.
Uses the Groq API (via LangChain) to rewrite conversational, ambiguous user
queries into keyword-rich standalone search queries optimized for vector retrieval.
"""

import os

from langchain_groq import ChatGroq

from src.core.config import GROQ_EXPANSION_MODEL
from src.core.logging_config import get_logger

log = get_logger(__name__)

_expansion_llm = None


def get_expansion_llm():
    """Lazily load the LangChain Groq client."""
    global _expansion_llm
    if _expansion_llm is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            log.warning("GROQ_API_KEY not found. Query expansion disabled.")
            return None
        try:
            _expansion_llm = ChatGroq(
                model_name=GROQ_EXPANSION_MODEL, temperature=0.0, reasoning_effort="low"
            )
        except Exception as e:
            log.error(f"Failed to initialize Groq LLM: {e}")
            return None
    return _expansion_llm


def expand_query(query: str, chat_history: list = None) -> str:
    """
    Rewrites a conversational query into a standalone, keyword-rich search query.
    If chat_history is provided, it resolves pronouns (e.g., 'they', 'it')
    based on the previous context.
    """
    llm = get_expansion_llm()
    if not llm:
        return query

    history_str = ""
    if chat_history:
        history_str = "Recent Conversation History:\n"
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"

    prompt = (
        "You are an expert in materials science and Solid Oxide Fuel Cells (SOFCs). "
        "Your task is to rewrite a user's question into a standalone, keyword-rich search query "
        "optimized for semantic vector retrieval. "
        "If there is conversation history, use it to resolve any ambiguous pronouns "
        "in the original question. "
        "Include synonyms and related domain terms. Do NOT answer the question. "
        "Return ONLY the rewritten query string, with no quotes or extra text.\n\n"
        f"{history_str}"
        f"Original Question: {query}\n"
        "Rewritten Query:"
    )

    try:
        response = llm.invoke(prompt)
        expanded = response.content.strip().strip('"').strip("'")
        log.info(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        log.error(f"Query expansion failed: {e}")
        return query
