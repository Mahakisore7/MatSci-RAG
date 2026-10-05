"""Tests for build_index.build_full_paper_chunks -- pure document assembly,
no embedding model or ChromaDB involved."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.build_index import build_full_paper_chunks  # noqa: E402

CORPUS = {
    "p1": {"id": "p1", "title": "Parsed paper", "abstract": "Abstract of the parsed paper."},
    "p2": {"id": "p2", "title": "Unparsed paper", "abstract": "YSZ electrolyte abstract text."},
}
FULL_PAPERS = {
    "p1": {
        "paper_id": "p1",
        "title": "Parsed paper",
        "sections": {
            "methods": "We sintered LSCF cathodes at 1100 C for two hours in air.",
            "references": "[1] A citation that should never be indexed at all.",
        },
    }
}


def test_papers_without_full_text_fall_back_to_their_abstract():
    chunks = build_full_paper_chunks(FULL_PAPERS, CORPUS, facts={})
    by_paper = {}
    for c in chunks:
        by_paper.setdefault(c.metadata["paper_id"], []).append(c)

    assert set(by_paper) == {"p1", "p2"}
    assert [c.metadata["section"] for c in by_paper["p1"]] == ["methods"]
    assert [c.metadata["section"] for c in by_paper["p2"]] == ["abstract"]
    assert "YSZ" in by_paper["p2"][0].page_content


def test_parsed_papers_use_sections_not_abstract():
    chunks = build_full_paper_chunks(FULL_PAPERS, CORPUS, facts={})
    p1_text = " ".join(c.page_content for c in chunks if c.metadata["paper_id"] == "p1")
    assert "Abstract of the parsed paper" not in p1_text
    assert "citation" not in p1_text
