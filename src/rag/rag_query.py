"""
Phase 5: RAG Generation — rag_query.py

The core RAG pipeline: embed a query, retrieve the most relevant passages
from ChromaDB, build a grounding prompt, and generate an answer from ONLY
that context using google/flan-t5-base.
"""

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "sofc_papers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this question for searching relevant documents: "

GEN_MODEL_NAME = "google/flan-t5-base"

# ── 1. Setup Retriever ───────────────────────────────────────────────────────
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# We defer LLM loading to avoid memory issues on import
_gen_model = None
_gen_tokenizer = None
_device = None


def get_llm():
    global _gen_model, _gen_tokenizer, _device
    if _gen_model is not None:
        return _gen_model, _gen_tokenizer, _device

    # In Windows, use "cuda" string instead of int for device placement
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading generator model '{GEN_MODEL_NAME}' on {_device} ...")

    _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    _gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(_device)

    return _gen_model, _gen_tokenizer, _device


def build_prompt(query: str, retrieved_docs: list) -> str:
    # Build context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.metadata.get("title", "Unknown Title")
        text = doc.page_content.strip()
        context_parts.append(f"[{i}] {title}\n{text}")

    context = "\n\n".join(context_parts)

    system_instruction = (
        "You are an expert materials science AI assistant"
        " specializing in Solid Oxide Fuel Cells (SOFCs). "
        "Answer the user's question using ONLY the context"
        " provided below. "
        "If the answer is not contained in the context, or if"
        " the question is completely unrelated to "
        "materials science, reply EXACTLY with:"
        " 'I don't know based on the provided context.' "
        "Do not hallucinate or use outside knowledge.\n\n"
    )
    return system_instruction + f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def answer_query(query: str) -> tuple[str, list]:
    """
    Given a user query, retrieves relevant SOFC documents and generates an answer.
    Returns a tuple of (generated_answer, source_documents).
    """
    log.info(f"Processing query: '{query}'")

    bge_prefixed_query = BGE_QUERY_PREFIX + query
    retrieved_docs = retriever.invoke(bge_prefixed_query)

    prompt = build_prompt(query, retrieved_docs)

    model, tokenizer, device = get_llm()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    log.info(f"Generated Answer: {answer[:80]}...")

    return answer, retrieved_docs


if __name__ == "__main__":
    queries = [
        "What materials are used as electrolytes in solid oxide fuel cells?",
        "What is the effect of temperature on SOFC performance?",
        "Which cathode materials show high oxygen reduction activity?",
        "Who won the Super Bowl last year?",  # Unrelated query test
    ]

    for q in queries:
        print(f"\n{'=' * 80}")
        print(f"QUESTION: {q}")
        print("=" * 80)

        answer, sources = answer_query(q)

        print(f"\nANSWER:\n{answer}\n")
        print("SOURCES:")
        for i, doc in enumerate(sources, 1):
            title = doc.metadata.get("title", "Unknown Title")
            print(f"  [{i}] {title}")
