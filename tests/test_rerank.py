from langchain_core.documents import Document

from src.rag import rerank
from src.rag.rerank import cap_per_paper, cross_encoder_rerank


def _doc(paper: str, text: str = "") -> Document:
    return Document(page_content=text, metadata={"paper_id": paper, "title": paper})


def test_cap_limits_chunks_per_paper_and_keeps_order():
    docs = [_doc("a"), _doc("a"), _doc("a"), _doc("b"), _doc("a"), _doc("c")]
    kept = cap_per_paper(docs, 2)
    assert [d.metadata["paper_id"] for d in kept] == ["a", "a", "b", "c"]


def test_cap_none_is_a_noop():
    docs = [_doc("a"), _doc("a"), _doc("a")]
    assert cap_per_paper(docs, None) == docs


def test_cross_encoder_orders_by_score(monkeypatch):
    class StubModel:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]  # longer text scores higher

    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: StubModel())
    docs = [_doc("short", "x"), _doc("long", "xxxxx"), _doc("mid", "xxx")]
    ranked = cross_encoder_rerank("q", docs)
    assert [d.metadata["paper_id"] for d in ranked] == ["long", "mid", "short"]


def test_cross_encoder_empty_input():
    assert cross_encoder_rerank("q", []) == []
