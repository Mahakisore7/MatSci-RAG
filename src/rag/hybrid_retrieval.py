import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

# pyrefly: ignore [missing-import]
from src.rag.ner_utils import facts_by_paper_id, fuse_and_rerank, load_structured_facts

# pyrefly: ignore [missing-import]
from src.rag.rerank import cap_per_paper, cross_encoder_rerank

log = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this question for searching relevant documents: "
# How many candidates to pull from each of dense/BM25 before fusing — wider
# than the final k so the NER boost (see ner_utils.fuse_and_rerank) has a
# real pool of candidates to re-rank rather than just re-sorting 5 items.
POOL_MULTIPLIER = 4
MIN_POOL_SIZE = 20

log.info("Initializing Hybrid Retriever (Dense + BM25)")

# 1. Setup Dense Retriever (Chroma)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# 2. Setup BM25 Retriever
# Extract all documents from ChromaDB to build the BM25 index in memory
db_data = vectorstore.get()
# Convert Chroma output to Langchain Document objects
_all_docs = []
for i in range(len(db_data["ids"])):
    _all_docs.append(
        Document(
            page_content=db_data["documents"][i],
            metadata=db_data["metadatas"][i],
        )
    )

log.info(f"Loaded {len(_all_docs)} documents into BM25 index.")
tokenized_corpus = [doc.page_content.split() for doc in _all_docs]
bm25 = BM25Okapi(tokenized_corpus)

# 3. Load NER facts once, keyed by paper_id, for the entity-overlap boost
_facts_by_paper_id = facts_by_paper_id(load_structured_facts())


def hybrid_retrieve(
    query: str,
    k: int = 5,
    use_ner_boost: bool = True,
    use_rerank: bool = False,
    per_paper_cap: int | None = None,
) -> list[Document]:
    """
    Retrieves documents using both Dense (Vector) and BM25 (Keyword) search,
    fuses them via Reciprocal Rank Fusion, and — optionally — boosts papers
    whose NER-extracted materials/devices/experiments are mentioned in the
    query. See ADR-009 for why this is a soft boost rather than a hard
    metadata filter.

    Optional second stage (ADR-019): ``use_rerank`` re-scores the fused pool with a
    cross-encoder, and ``per_paper_cap`` limits chunks per paper in the final k.
    """
    log.info(f"Hybrid retrieval for: {query}")

    pool_k = max(k * POOL_MULTIPLIER, MIN_POOL_SIZE)

    # ── A. Dense Search ──
    bge_query = BGE_QUERY_PREFIX + query
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": pool_k})
    dense_results = dense_retriever.invoke(bge_query)

    # ── B. BM25 Search ──
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:pool_k]
    bm25_results = [_all_docs[i] for i in top_indices]

    # ── C. Fuse (RRF) + NER-aware rerank + dedupe ──
    facts = _facts_by_paper_id if use_ner_boost else {}
    second_stage = use_rerank or per_paper_cap is not None
    final_docs = fuse_and_rerank(
        dense_results, bm25_results, query, facts, pool_k if second_stage else k
    )
    if use_rerank:
        final_docs = cross_encoder_rerank(query, final_docs)
    final_docs = cap_per_paper(final_docs, per_paper_cap)[:k]

    log.info(
        f"Fused {len(dense_results)} dense and {len(bm25_results)} "
        f"BM25 candidates into {len(final_docs)} ranked results "
        f"(ner_boost={use_ner_boost})."
    )
    return final_docs
