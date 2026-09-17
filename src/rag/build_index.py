"""
Phase 4 / Phase 9 Extension: RAG Indexing — build_index.py

This script:
1. Loads the raw SOFC corpus from data/sofc_corpus.jsonl (abstracts)
   OR full-paper sections from data/full_papers.jsonl (--source full_paper)
2. Loads structured NER metadata from data/structured_facts.jsonl
3. Merges them into LangChain Documents with rich metadata
4. Chunks the text using RecursiveCharacterTextSplitter with overlap
5. Embeds each chunk using BAAI/bge-small-en-v1.5
6. Persists the vectors + metadata to a local ChromaDB at data/chroma_db/
   (replaces any existing collection — clean rebuild)
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
FULL_PAPERS_PATH = "data/full_papers.jsonl"
FACTS_PATH = "data/structured_facts.jsonl"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"

# ── Embedding model ──────────────────────────────────────────────────────────
# BGE-small: top-ranked open-source embedding model on MTEB leaderboard.
# Instruction prefix is required by BGE for passage embedding.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_PASSAGE_PREFIX = "Represent this science document for retrieval: "

# ── Chunking config ──────────────────────────────────────────────────────────
CHUNK_SIZE_ABSTRACT = 500  # characters — abstracts are compact
CHUNK_SIZE_FULL = 800  # characters — full-paper prose is denser
CHUNK_OVERLAP = 50  # overlap between consecutive chunks


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


def load_full_papers(path: str) -> dict[str, dict]:
    """Load full_papers.jsonl (output of pdf_parser.py) keyed by paper_id."""
    papers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            papers[doc["paper_id"]] = doc
    log.info(f"Loaded {len(papers)} full-paper records")
    return papers


def build_documents_from_full_papers(full_papers: dict, facts: dict) -> list[Document]:
    """
    Build LangChain Documents from full-paper section records.
    Each section (methods, results, etc.) becomes its own Document so
    the section name is stored as metadata for future filtered retrieval.
    Sections in SECTIONS_TO_SKIP are excluded.
    """
    SECTIONS_TO_SKIP = {"references"}  # boilerplate, no signal
    docs = []
    for paper_id, paper in full_papers.items():
        fact = facts.get(paper_id, {})
        materials = clean_entities(fact.get("materials", []))
        devices = clean_entities(fact.get("devices", []))
        experiments = clean_entities(fact.get("experiments", []))

        base_metadata = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "materials": ", ".join(materials),
            "devices": ", ".join(devices),
            "experiments": ", ".join(experiments),
        }

        for section_name, section_text in paper.get("sections", {}).items():
            if section_name in SECTIONS_TO_SKIP:
                continue
            if not section_text or len(section_text.strip()) < 50:
                continue
            metadata = {**base_metadata, "section": section_name}
            docs.append(Document(page_content=section_text.strip(), metadata=metadata))

    log.info(f"Built {len(docs)} section Documents from {len(full_papers)} papers")
    return docs


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


def chunk_documents(docs: list[Document], chunk_size: int = CHUNK_SIZE_ABSTRACT) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={CHUNK_OVERLAP})")
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

    # Chroma.from_documents() get-or-creates the collection and ADDS to it —
    # it does not clear existing vectors. Delete first so re-running this
    # (e.g. --source full_paper over an existing abstract-based index) is a
    # true replace, not a silent merge of both chunk sets into one collection.
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ).delete_collection()
        log.info(f"Deleted existing '{COLLECTION_NAME}' collection for a clean rebuild.")
    except Exception as e:
        log.debug(f"No existing collection to delete ({e}) — proceeding with fresh build.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    log.info(f"Index built. {vectorstore._collection.count()} vectors stored in ChromaDB.")
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ChromaDB vector index.")
    parser.add_argument(
        "--source",
        choices=["abstract", "full_paper"],
        default="abstract",
        help="'abstract' uses sofc_corpus.jsonl (default). "
        "'full_paper' uses full_papers.jsonl (requires pdf_parser.py output).",
    )
    args = parser.parse_args()

    facts = load_facts(FACTS_PATH)

    if args.source == "full_paper":
        log.info("Building index from FULL PAPER sections...")
        if not os.path.exists(FULL_PAPERS_PATH):
            raise FileNotFoundError(
                f"{FULL_PAPERS_PATH} not found. "
                "Run: uv run python -m src.ingestion.pdf_parser first."
            )
        full_papers = load_full_papers(FULL_PAPERS_PATH)
        docs = build_documents_from_full_papers(full_papers, facts)
        chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE_FULL)
    else:
        log.info("Building index from ABSTRACTS (default)...")
        corpus = load_corpus(CORPUS_PATH)
        docs = build_documents(corpus, facts)
        chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE_ABSTRACT)

    build_index(chunks)
    log.info(f"Phase 4/9 complete. ChromaDB index written to {CHROMA_DIR}/")
