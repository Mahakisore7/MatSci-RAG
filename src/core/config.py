import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")

# Groq retires models without much notice (groq/compound-mini, llama-3.1-8b-instant
# and llama3-8b-8192 have all gone) -- override via env rather than editing code.
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
GROQ_EXPANSION_MODEL = os.getenv("GROQ_EXPANSION_MODEL", "openai/gpt-oss-20b")

# "groq" answers with GROQ_GEN_MODEL and falls back to the local GEN_MODEL_NAME when the
# API is unavailable; "local" always uses GEN_MODEL_NAME. Generator, verifier and judge are
# deliberately three different models so no model grades its own output.
GEN_BACKEND = os.getenv("GEN_BACKEND", "groq")
GROQ_GEN_MODEL = os.getenv("GROQ_GEN_MODEL", "qwen/qwen3.8-27b")
GROQ_VERIFY_MODEL = os.getenv("GROQ_VERIFY_MODEL", "openai/gpt-oss-20b")

# Second-stage ranking (ADR-019): a local cross-encoder re-scores the fused candidates.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANK = os.getenv("USE_RERANK", "1") == "1"  # set USE_RERANK=0 to compare without it


def active_generator() -> str:
    """The model that will answer, given the backend setting and whether a Groq key exists."""
    if GEN_BACKEND == "groq" and os.getenv("GROQ_API_KEY"):
        return GROQ_GEN_MODEL
    return GEN_MODEL_NAME
