"""
Shared utilities that bridge the Phase 3 NER output (structured_facts.jsonl)
into the Phase 4/5/8 retrieval layer: cleaning noisy entity tokens, indexing
per-paper facts, detecting ambiguous (multi-material) result sets, and
scoring how well a paper's extracted entities overlap with a query.

Deliberately dependency-light (no torch/transformers/chromadb) so it can be
imported and unit-tested without any local model/data artifacts.
"""

import json

from langchain_core.documents import Document

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

# Common non-entity tokens that survive raw NER decoding: stopwords, and
# generic domain words too common to be useful as a retrieval signal.
STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "of",
    "in",
    "on",
    "with",
    "solid",
    "sof",
    "fuel",
    "reaction",
    "distinct",
    "screened",
    "potential",
    "predicted",
    "materials",
    "material",
    "high",
    "metal",
}


def clean_entities(entities: list[str]) -> list[str]:
    """Dedupe and filter noisy NER output: wordpieces, punctuation/short
    tokens, and common stopwords. Order-preserving."""
    seen = set()
    result = []
    for raw in entities:
        e = raw.strip()
        if not e or e.startswith("##") or len(e) <= 2:
            continue
        if e.lower() in STOPWORDS:
            continue
        if e in seen:
            continue
        seen.add(e)
        result.append(e)
    return result


def load_structured_facts(filepath: str = "data/structured_facts.jsonl") -> list[dict]:
    facts = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                facts.append(json.loads(line))
    except FileNotFoundError:
        log.warning(f"Could not load {filepath} - NER-aware features will be disabled")
    return facts


def facts_by_paper_id(facts: list[dict]) -> dict[str, dict]:
    """Index structured_facts records by their source_id (== the paper_id
    stored in each chunk's Chroma metadata) for O(1) lookup."""
    return {fact["source_id"]: fact for fact in facts if "source_id" in fact}


# The trained NER model's precision on the "materials" entity type is low
# (~9% F1, see RESULTS.md) — most extractions are noise, not real material
# names (verified: only 2/400 papers in structured_facts.jsonl contain a
# recognizable real material keyword anywhere in their "materials" list).
# A naive low threshold (the original "> 3") fires on noise alone: an
# empirical sweep across 15 varied queries — including clearly irrelevant
# ones like "what is the capital of France?" — found noise-driven distinct
# candidate counts of 4-18 with no separation between relevant and
# irrelevant queries. This threshold is set well above that observed
# ceiling so the feature fails safe (never blocks an ordinary answer)
# rather than misfiring on noise. It will rarely trigger until NER
# precision improves — see ADR-009.
MIN_DISTINCT_MATERIALS_FOR_AMBIGUITY = 25


def detect_ambiguity(
    retrieved_docs: list[Document], facts_by_id: dict[str, dict]
) -> list[str] | None:
    """If the retrieved chunks span an unusually large number of distinct
    (cleaned) materials across their source papers, flag it so the caller
    can abstain and ask the user to narrow the question."""
    paper_ids = {
        doc.metadata.get("paper_id") for doc in retrieved_docs if doc.metadata.get("paper_id")
    }
    candidates = set()
    for paper_id in paper_ids:
        fact = facts_by_id.get(paper_id)
        if fact:
            candidates.update(clean_entities(fact.get("materials", [])))

    if len(candidates) > MIN_DISTINCT_MATERIALS_FOR_AMBIGUITY:
        return list(candidates)
    return None


def score_entity_overlap(query: str, entities: list[str]) -> int:
    """Count distinct cleaned entities that appear as a substring of the
    (lowercased) query — a cheap, robust proxy for "this paper's NER facts
    are actually relevant to what was asked"."""
    query_lower = query.lower()
    return sum(1 for e in clean_entities(entities) if e.lower() in query_lower)


# Reciprocal Rank Fusion constant. 60 is the standard value from the
# original RRF paper (Cormack et al.) — large enough that fusion isn't
# dominated by whichever single list a doc happens to rank #1 in.
_RRF_K = 60
# Weight of a single matched NER entity, expressed in RRF-score terms:
# roughly equivalent to one extra rank-1 appearance per match, enough to
# meaningfully move ranking without letting one noisy match dominate.
_NER_BOOST_WEIGHT = 0.02


def _doc_key(doc: Document) -> str:
    return str(doc.metadata.get("paper_id", "")) + doc.page_content[:50]


def fuse_and_rerank(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    query: str,
    facts_by_id: dict[str, dict],
    k: int,
) -> list[Document]:
    """Combine dense + BM25 result lists via Reciprocal Rank Fusion, add a
    small boost for documents whose paper-level NER entities are mentioned
    in the query, dedupe, and return the top k."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = _doc_key(doc)
        docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    for rank, doc in enumerate(bm25_docs):
        key = _doc_key(doc)
        docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    for key, doc in docs_by_key.items():
        paper_id = doc.metadata.get("paper_id")
        fact = facts_by_id.get(paper_id) if paper_id else None
        if fact:
            all_entities = (
                fact.get("materials", []) + fact.get("devices", []) + fact.get("experiments", [])
            )
            overlap = score_entity_overlap(query, all_entities)
            scores[key] += overlap * _NER_BOOST_WEIGHT

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:k]]
