# ruff: isort: skip_file
"""
Phase 5: RAG Generation — rag_query.py

The core RAG pipeline: embed a query, retrieve the most relevant passages
from ChromaDB, build a grounding prompt, and generate an answer from ONLY
that context using google/flan-t5-base.
"""

# Must import HuggingFaceEmbeddings before torch to avoid Windows DLL crash
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: I001
import torch  # noqa: I001

from langchain_chroma import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

# pyrefly: ignore [missing-import]
from src.rag.ner_utils import detect_ambiguity, facts_by_paper_id, load_structured_facts

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


def build_prompt(query: str, retrieved_docs: list, chat_history: list = None) -> str:
    # Build context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.metadata.get("title", "Unknown Title")
        text = doc.page_content.strip()
        context_parts.append(f"[{i}] {title}\n{text}")

    context = "\n\n".join(context_parts)

    history_str = ""
    if chat_history:
        history_str = "Recent Conversation History:\n"
        # Take last 6 messages (3 turns) to avoid context overflow
        recent_history = chat_history[-6:]
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"

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
    return system_instruction + f"Context:\n{context}\n\n{history_str}Question: {query}\nAnswer:"


def retrieve(query: str, k: int = 5) -> list:
    """Helper for pure retrieval testing."""
    bge_prefixed_query = BGE_QUERY_PREFIX + query
    # Temporarily override k if needed
    retriever.search_kwargs["k"] = k
    docs = retriever.invoke(bge_prefixed_query)
    # Format the return to match test expectations (dicts with 'text' key)
    results = []
    for doc in docs:
        results.append(
            {
                "text": doc.page_content,
                "title": doc.metadata.get("title", ""),
                "source_id": doc.metadata.get("paper_id", ""),
            }
        )
    return results


_structured_facts = facts_by_paper_id(load_structured_facts())


def answer_query(
    query: str, use_hybrid: bool = True, chat_history: list = None, expand_q: bool = False
) -> tuple[str, list]:
    """Retrieves context and generates a grounded answer."""
    log.info(f"Retrieving context for query: {query}")

    search_query = query
    if expand_q:
        from src.rag.query_expansion import expand_query

        search_query = expand_query(query, chat_history)

    if use_hybrid:
        from src.rag.hybrid_retrieval import hybrid_retrieve

        retrieved_docs = hybrid_retrieve(search_query, k=5, use_ner_boost=True)
    else:
        bge_query = BGE_QUERY_PREFIX + search_query
        retrieved_docs = retriever.invoke(bge_query)

    # Check for ambiguity before generating an answer
    ambiguous_materials = detect_ambiguity(retrieved_docs, _structured_facts)
    if ambiguous_materials:
        mat_str = ", ".join(ambiguous_materials[:10])
        msg = (
            f"Multiple candidate materials found across sources: {mat_str}. "
            "Please narrow your question."
        )
        log.info("Ambiguity detected, abstaining from answering.")
        return msg, retrieved_docs

    prompt = build_prompt(query, retrieved_docs, chat_history)

    model, tokenizer, device = get_llm()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    log.info(f"Generated Answer: {answer[:80]}...")

    return answer, retrieved_docs


def verify_answer(query: str, answer: str, retrieved_docs: list) -> str:
    """Asks the same LLM a follow-up question: does the retrieved context
    actually support the answer it just gave? A second, independent check
    before the answer is trusted."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    verify_prompt = (
        f"Context:\n{context}\n\n"
        f'Claim: "{answer}" (answering the question: "{query}")\n\n'
        "Question: Does the context actually support this claim? Answer yes or no, "
        "and quote the exact supporting sentence if yes.\nAnswer:"
    )

    model, tokenizer, device = get_llm()
    inputs = tokenizer(verify_prompt, return_tensors="pt", truncation=True, max_length=1024).to(
        device
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    verification = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return verification


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
