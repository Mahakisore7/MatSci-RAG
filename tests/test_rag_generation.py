import os
import sys

import pytest

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.skipif(
    not os.path.exists("data/chroma_db"),
    reason="data/chroma_db not present (gitignored local artifact)",
)
def test_answer_query_returns_answer_and_sources():
    from src.rag.rag_query import answer_query

    # We test with a known SOFC question
    answer, sources = answer_query("What is the effect of temperature on SOFC performance?")

    # Verify we got a string answer back
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Verify the hallucination guardrails work for unrelated questions
    unrelated_answer, _ = answer_query("Who won the Super Bowl last year?")
    assert "I don't know" in unrelated_answer

    # Verify we got source documents
    assert isinstance(sources, list)
    assert len(sources) > 0

    # Verify the source document format
    first_doc = sources[0]
    assert hasattr(first_doc, "page_content")
    assert hasattr(first_doc, "metadata")
    assert "title" in first_doc.metadata
