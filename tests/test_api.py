"""Tests for the FastAPI layer (src/api/main.py).

The RAG pipeline is mocked, so these check the HTTP contract only — no
models, ChromaDB, or network needed.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src.api import main  # noqa: E402

DOCS = [
    Document(
        page_content="YSZ is the most common SOFC electrolyte.",
        metadata={"title": "Paper A", "paper_id": "http://arxiv.org/abs/1", "section": "abstract"},
    ),
    Document(page_content="GDC works at lower temperatures.", metadata={"title": "Paper B"}),
]


@pytest.fixture
def pipeline(monkeypatch):
    answer_query = MagicMock(return_value=("YSZ.", DOCS))
    verify_answer = MagicMock(return_value="yes: YSZ is the most common SOFC electrolyte.")
    monkeypatch.setattr(main, "get_pipeline", lambda: (answer_query, verify_answer))
    return answer_query, verify_answer


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_answer_sources_and_verification(client, pipeline):
    r = client.post("/query", json={"question": "Which electrolyte is most common?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "YSZ."
    assert body["abstained"] is False
    assert body["verification"].startswith("yes")
    assert [s["rank"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["section"] == "abstract"
    assert body["sources"][1]["paper_id"] == ""


def test_query_forwards_options_to_pipeline(client, pipeline):
    answer_query, _ = pipeline
    client.post(
        "/query",
        json={
            "question": "And at lower temperature?",
            "use_hybrid": False,
            "expand_query": True,
            "chat_history": [{"role": "user", "content": "Which electrolyte?"}],
        },
    )
    answer_query.assert_called_once_with(
        "And at lower temperature?",
        use_hybrid=False,
        chat_history=[{"role": "user", "content": "Which electrolyte?"}],
        expand_q=True,
    )


def test_query_skips_verification_when_disabled(client, pipeline):
    _, verify_answer = pipeline
    r = client.post("/query", json={"question": "q", "verify": False})
    assert r.json()["verification"] is None
    verify_answer.assert_not_called()


def test_query_skips_verification_on_abstention(client, pipeline):
    answer_query, verify_answer = pipeline
    answer_query.return_value = ("Multiple candidate materials found across sources: A, B.", DOCS)
    body = client.post("/query", json={"question": "q"}).json()
    assert body["abstained"] is True
    assert body["verification"] is None
    verify_answer.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "q", "chat_history": [{"role": "system", "content": "hi"}]},
    ],
)
def test_query_rejects_invalid_input(client, pipeline, payload):
    assert client.post("/query", json=payload).status_code == 422
