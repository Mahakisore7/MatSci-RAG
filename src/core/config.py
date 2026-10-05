import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-large")
