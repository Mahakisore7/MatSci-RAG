import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.skipif(
    not os.path.exists("data/chroma_db"),
    reason="data/chroma_db not present (gitignored local artifact)",
)
def test_retrieval_returns_results():
    from src.rag.rag_query import retrieve

    # Corpus is SOFC papers (see ADR-003) — an HEA query was left here from
    # before the project's domain pivot and didn't reflect what's indexed.
    results = retrieve("electrolyte materials in solid oxide fuel cells", k=3)
    assert len(results) == 3
    assert all("text" in r for r in results)
