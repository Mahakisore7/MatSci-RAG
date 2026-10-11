"""
Second-stage ranking for hybrid retrieval (ADR-019).

Two independent pieces:

* ``cap_per_paper`` -- limit how many chunks one paper may contribute, so a
  single paper cannot fill every slot of the final top-k.
* ``cross_encoder_rerank`` -- re-score candidates with a cross-encoder, which
  reads the query and the passage together and is far better at "does this
  passage answer this question" than the bi-encoder + BM25 first stage.

The model is loaded lazily so importing this module costs nothing.
"""

from functools import lru_cache

from langchain_core.documents import Document

from src.core.config import RERANK_MODEL
from src.core.logging_config import get_logger

log = get_logger(__name__)


def cap_per_paper(docs: list[Document], cap: int | None) -> list[Document]:
    """Keep order, but drop chunks beyond ``cap`` per paper. ``None`` disables the cap."""
    if cap is None:
        return list(docs)
    seen: dict[str, int] = {}
    kept = []
    for doc in docs:
        paper = doc.metadata.get("paper_id") or doc.metadata.get("title", "")
        if seen.get(paper, 0) < cap:
            kept.append(doc)
            seen[paper] = seen.get(paper, 0) + 1
    return kept


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    log.info(f"Loading cross-encoder: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def cross_encoder_rerank(query: str, docs: list[Document]) -> list[Document]:
    """Return ``docs`` sorted by cross-encoder relevance to ``query`` (best first)."""
    if not docs:
        return []
    scores = _get_cross_encoder().predict([(query, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order]
