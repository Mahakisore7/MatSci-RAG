"""
Phase 4: RAG Indexing — quick_retrieval_test.py

A lightweight interactive test to verify that the ChromaDB index is working.
Run after build_index.py has successfully completed.

Usage:
    uv run python -m src.rag.quick_retrieval_test
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE requires different prefixes for queries vs passages
BGE_QUERY_PREFIX = "Represent this question for searching relevant documents: "

TEST_QUERIES = [
    "What materials are used as electrolytes in solid oxide fuel cells?",
    "What is the effect of temperature on SOFC performance?",
    "Which cathode materials show high oxygen reduction activity?",
]


def run_test():
    log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    log.info(f"Connecting to ChromaDB at {CHROMA_DIR} ...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    count = vectorstore._collection.count()
    log.info(f"ChromaDB connected — {count} vectors in collection '{COLLECTION_NAME}'")

    for query in TEST_QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print("=" * 70)

        # Prepend BGE query instruction prefix manually for accurate retrieval
        prefixed_query = BGE_QUERY_PREFIX + query
        results = vectorstore.similarity_search_with_score(prefixed_query, k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n  Result #{i} | Similarity: {score:.4f}")
            print(f"  Title:     {doc.metadata.get('title', 'N/A')}")
            print(f"  Materials: {doc.metadata.get('materials', 'N/A')}")
            print(f"  Text:      {doc.page_content[:250]}...")


if __name__ == "__main__":
    run_test()
