"""Unit tests for src/rag/ner_utils.py.

These are pure-logic tests against hand-built Document/dict fixtures — no
local model or data artifacts required, so they run on every CI push.
"""

import os
import sys

from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.ner_utils import (  # noqa: E402
    clean_entities,
    detect_ambiguity,
    fuse_and_rerank,
    score_entity_overlap,
)


def test_clean_entities_filters_noise_and_dedupes():
    raw = ["the", "YSZ", "##ide", "or", "cathode", "YSZ", "a", "cathode"]
    assert clean_entities(raw) == ["YSZ", "cathode"]


def test_clean_entities_preserves_order():
    assert clean_entities(["zirconia", "nickel", "zirconia"]) == ["zirconia", "nickel"]


def test_detect_ambiguity_fires_above_threshold():
    # The threshold is set high (see ner_utils.MIN_DISTINCT_MATERIALS_FOR_AMBIGUITY)
    # to stay well above the noise floor of the current NER model — so
    # exercise it with enough distinct materials to actually cross it.
    docs = [
        Document(page_content="...", metadata={"paper_id": "p1"}),
        Document(page_content="...", metadata={"paper_id": "p2"}),
    ]
    materials_a = [f"material-{i}" for i in range(15)]
    materials_b = [f"material-{i}" for i in range(15, 30)]
    facts_by_id = {
        "p1": {"materials": materials_a},
        "p2": {"materials": materials_b},
    }
    result = detect_ambiguity(docs, facts_by_id)
    assert result is not None
    assert set(result) == set(materials_a) | set(materials_b)


def test_detect_ambiguity_silent_for_few_materials():
    docs = [Document(page_content="...", metadata={"paper_id": "p1"})]
    facts_by_id = {"p1": {"materials": ["YSZ", "GDC"]}}
    assert detect_ambiguity(docs, facts_by_id) is None


def test_detect_ambiguity_ignores_docs_with_no_matching_facts():
    # Regression test for the paper_id/source_id key-mismatch bug: a doc
    # whose paper_id isn't present in facts_by_id must not blow up or
    # falsely contribute candidates.
    docs = [Document(page_content="...", metadata={"paper_id": "unknown"})]
    facts_by_id = {"p1": {"materials": ["YSZ", "GDC", "LSCF", "nickel"]}}
    assert detect_ambiguity(docs, facts_by_id) is None


def test_score_entity_overlap_counts_query_mentions_only():
    query = "What cathode materials are used in SOFCs?"
    entities = ["cathode", "the", "zirconia"]
    # "the" is filtered out as a stopword before matching, "zirconia" isn't
    # mentioned in the query, so only "cathode" should count.
    assert score_entity_overlap(query, entities) == 1


def test_fuse_and_rerank_boosts_docs_agreeing_across_both_lists():
    doc_a = Document(page_content="Paper about electrolytes.", metadata={"paper_id": "a"})
    doc_b = Document(page_content="Paper about cathodes.", metadata={"paper_id": "b"})
    doc_c = Document(page_content="Paper about anodes.", metadata={"paper_id": "c"})

    # doc_a appears in both lists at rank 0; doc_b and doc_c each appear in
    # only one list.
    dense_docs = [doc_a, doc_b]
    bm25_docs = [doc_a, doc_c]

    result = fuse_and_rerank(dense_docs, bm25_docs, "query", facts_by_id={}, k=3)

    assert [d.metadata["paper_id"] for d in result[:1]] == ["a"]


def test_fuse_and_rerank_applies_ner_boost():
    # Both docs rank equally by RRF alone (same position in a single list),
    # but only doc_b's paper has NER facts matching the query.
    doc_a = Document(page_content="...", metadata={"paper_id": "a"})
    doc_b = Document(page_content="...", metadata={"paper_id": "b"})

    facts_by_id = {"b": {"materials": ["zirconia"], "devices": [], "experiments": []}}

    result = fuse_and_rerank(
        dense_docs=[doc_a],
        bm25_docs=[doc_b],
        query="What is zirconia used for?",
        facts_by_id=facts_by_id,
        k=2,
    )

    assert result[0].metadata["paper_id"] == "b"


def test_fuse_and_rerank_dedupes_same_chunk_from_both_lists():
    shared = Document(page_content="Shared chunk text.", metadata={"paper_id": "a"})
    other = Document(page_content="Other chunk text.", metadata={"paper_id": "b"})

    result = fuse_and_rerank(
        dense_docs=[shared, other],
        bm25_docs=[shared],
        query="query",
        facts_by_id={},
        k=5,
    )

    assert len(result) == 2


def test_fuse_and_rerank_respects_k():
    docs = [Document(page_content=f"chunk {i}", metadata={"paper_id": str(i)}) for i in range(10)]
    result = fuse_and_rerank(docs, [], "query", facts_by_id={}, k=3)
    assert len(result) == 3
