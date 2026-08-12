import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_retrieval_returns_results():
    from src.rag.rag_query import retrieve

    results = retrieve("hardness of high entropy alloys", k=3)
    assert len(results) == 3
    assert all("text" in r for r in results)
