import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this question for searching relevant documents: "

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


def hybrid_retrieve(query: str, k: int = 5, use_ner_filter: bool = True) -> list[Document]:
    """
    Retrieves documents using both Dense (Vector) and BM25 (Keyword) search.
    Optionally applies an NER filter to the Dense search to restrict to SOFC domains.
    """
    log.info(f"Hybrid retrieval for: {query}")

    # ── A. Dense Search (with optional NER Metadata Filter) ──
    bge_query = BGE_QUERY_PREFIX + query

    filter_dict = None
    if use_ner_filter:
        # We know from data exploration that true SOFC papers have "fuel" or
        # "oxide" in their NER devices list
        filter_dict = {
            "$or": [{"devices": {"$contains": "fuel"}}, {"devices": {"$contains": "oxide"}}]
        }

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k, "filter": filter_dict})
    dense_results = dense_retriever.invoke(bge_query)

    # ── B. BM25 Search ──
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [_all_docs[i] for i in top_indices]

    # ── C. Combine and Deduplicate ──
    # We use source_id + page_content as a unique key in case chunks have the same source_id
    combined_results = {}
    for doc in dense_results + bm25_results:
        # Create a unique key for the chunk
        key = str(doc.metadata.get("source_id", "")) + doc.page_content[:50]
        combined_results[key] = doc

    final_docs = list(combined_results.values())[:k]
    log.info(
        f"Merged {len(dense_results)} dense and {len(bm25_results)} "
        f"BM25 results into {len(final_docs)} unique docs."
    )
    return final_docs
