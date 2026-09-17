"""Unit tests for src/rag/query_expansion.py.

These test the module's fallback and glue logic only -- no real network
call to Groq is made. get_expansion_llm() (and the module-level LLM
cache) is mocked/patched so these run without a GROQ_API_KEY, offline,
in CI.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag import query_expansion  # noqa: E402


def test_expand_query_returns_original_when_no_llm_available():
    with patch.object(query_expansion, "get_expansion_llm", return_value=None):
        result = query_expansion.expand_query("What is YSZ?", chat_history=None)
    assert result == "What is YSZ?"


def test_expand_query_returns_original_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Groq API unreachable")
    with patch.object(query_expansion, "get_expansion_llm", return_value=mock_llm):
        result = query_expansion.expand_query("What is YSZ?", chat_history=None)
    assert result == "What is YSZ?"


def test_expand_query_strips_quotes_from_llm_response():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='"yttria stabilized zirconia properties"')
    with patch.object(query_expansion, "get_expansion_llm", return_value=mock_llm):
        result = query_expansion.expand_query("What is YSZ?", chat_history=None)
    assert result == "yttria stabilized zirconia properties"


def test_expand_query_includes_chat_history_in_prompt():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    history = [
        {"role": "user", "content": "Tell me about YSZ electrolytes."},
        {"role": "assistant", "content": "YSZ is a common SOFC electrolyte material."},
    ]
    with patch.object(query_expansion, "get_expansion_llm", return_value=mock_llm):
        query_expansion.expand_query("What temperature does it operate at?", chat_history=history)

    sent_prompt = mock_llm.invoke.call_args[0][0]
    assert "YSZ electrolytes" in sent_prompt
    assert "Recent Conversation History" in sent_prompt


def test_expand_query_omits_history_block_when_none():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    with patch.object(query_expansion, "get_expansion_llm", return_value=mock_llm):
        query_expansion.expand_query("What is YSZ?", chat_history=None)

    sent_prompt = mock_llm.invoke.call_args[0][0]
    assert "Recent Conversation History" not in sent_prompt


def test_get_expansion_llm_returns_none_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        query_expansion._expansion_llm = None
        result = query_expansion.get_expansion_llm()
    assert result is None
