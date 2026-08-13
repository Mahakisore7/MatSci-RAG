"""
Phase 4: RAG Indexing — build_index.py

This script:
1. Loads the raw SOFC corpus (400 papers) from data/sofc_corpus.jsonl
2. Loads structured NER metadata from data/structured_facts.jsonl
3. Merges them into LangChain Documents with rich metadata
4. Chunks the text using RecursiveCharacterTextSplitter with overlap
5. Embeds each chunk using BAAI/bge-small-en-v1.5 (top MTEB leaderboard model)
6. Persists the vectors + metadata to a local ChromaDB at data/chroma_db/
"""

import json
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

# pyrefly: ignore [missing-import]
from src.rag.ner_utils import clean_entities

log = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
CORPUS_PATH = "data/sofc_corpus.jsonl"
FACTS_PATH = "data/structured_facts.jsonl"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"

# ── Embedding model ──────────────────────────────────────────────────────────
# BGE-small: top-ranked open-source embedding model on MTEB leaderboard.
# Instruction prefix is required by BGE for passage embedding.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_PASSAGE_PREFIX = "Represent this science document for retrieval: "

# ── Chunking config ──────────────────────────────────────────────────────────
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between consecutive chunks to preserve context


def load_corpus(path: str) -> dict[str, dict]:
    """Load sofc_corpus.jsonl into a dict keyed by paper id."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["id"]] = doc
    log.info(f"Loaded {len(corpus)} papers from corpus")
    return corpus


def load_facts(path: str) -> dict[str, dict]:
    """Load structured_facts.jsonl into a dict keyed by source_id."""
    facts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            facts[entry["source_id"]] = entry
    log.info(f"Loaded NER facts for {len(facts)} papers")
    return facts


def build_documents(corpus: dict, facts: dict) -> list[Document]:
    """
    Combine corpus text with NER metadata into LangChain Document objects.
    The NER tags are stored in metadata for later hybrid filtering.
    """
    docs = []
    for paper_id, paper in corpus.items():
        # Facts are keyed by source_id which matches the full URL paper id
        fact = facts.get(paper_id, {})

        # Store clean text only — BGE prefix is only used at embedding time
        text = paper.get("abstract", "")

        materials = clean_entities(fact.get("materials", []))
        devices = clean_entities(fact.get("devices", []))
        experiments = clean_entities(fact.get("experiments", []))

        # Build metadata payload — all values must be strings for ChromaDB
        metadata = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "materials": ", ".join(materials),
            "devices": ", ".join(devices),
            "experiments": ", ".join(experiments),
        }

        docs.append(Document(page_content=text, metadata=metadata))

    log.info(f"Built {len(docs)} LangChain Documents")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_index(chunks: list[Document]) -> Chroma:
    """Embed chunks using BGE and persist to ChromaDB."""
    log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")

    # encode_kwargs normalize_embeddings=True is required for BGE
    # model_kwargs prompt_name="s2p_query" applies the correct BGE prefix at embedding time
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True, "prompt": BGE_PASSAGE_PREFIX},
    )

    log.info(f"Building ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR} ...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    log.info(f"Index built. {vectorstore._collection.count()} vectors stored in ChromaDB.")
    return vectorstore


if __name__ == "__main__":
    corpus = load_corpus(CORPUS_PATH)
    facts = load_facts(FACTS_PATH)
    docs = build_documents(corpus, facts)
    chunks = chunk_documents(docs)
    build_index(chunks)
    log.info("Phase 4 complete. ChromaDB index written to data/chroma_db/")
