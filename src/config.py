# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")  # None if not set -- fine, current models are all public