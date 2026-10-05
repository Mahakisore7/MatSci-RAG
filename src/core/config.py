import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-large")

# Groq retires models without much notice (groq/compound-mini, llama-3.1-8b-instant
# and llama3-8b-8192 have all gone) -- override via env rather than editing code.
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
GROQ_EXPANSION_MODEL = os.getenv("GROQ_EXPANSION_MODEL", "openai/gpt-oss-20b")
