"""Cloud Data Bootstrapper — ensures runtime knowledge base is present.

If ChromaDB and structured facts are missing (as on a clean Streamlit Cloud clone),
this module automatically downloads and extracts the pre-built knowledge base archive
from GitHub Releases or Hugging Face Datasets.
"""

import os
import shutil
import sys
import tarfile
import urllib.request

from src.core.logging_config import get_logger

log = get_logger(__name__)

# Default public repository release URL for the 120MB knowledge base bundle
DEFAULT_BUNDLE_URL = os.getenv(
    "DATA_BUNDLE_URL",
    "https://github.com/Mahakisore7/MatSci-RAG/releases/download/v1.0.0/data_bundle.tar.gz",
)

REQUIRED_FILES = [
    "data/chroma_db",
    "data/structured_facts.jsonl",
    "data/sofc_corpus.jsonl",
]


def is_data_ready() -> bool:
    """Check if all required data artifacts exist locally."""
    return all(os.path.exists(p) for p in REQUIRED_FILES)


def download_and_extract_bundle(bundle_url: str = DEFAULT_BUNDLE_URL) -> bool:
    """Download the compressed data bundle and extract it into data/."""
    archive_name = "temp_data_bundle.tar.gz"

    log.info(f"Downloading knowledge base archive from: {bundle_url}")
    print(f"Downloading knowledge base from: {bundle_url} ...")

    try:
        # Stream download with progress report
        req = urllib.request.Request(
            bundle_url,
            headers={"User-Agent": "MatSci-RAG-Cloud-Bootstrapper/1.0"},
        )
        with urllib.request.urlopen(req) as response, open(archive_name, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        log.info("Extracting knowledge base archive...")
        print("Extracting knowledge base archive...")
        with tarfile.open(archive_name, "r:gz") as tar:
            tar.extractall(path=".")

        if os.path.exists(archive_name):
            os.remove(archive_name)

        log.info("Knowledge base successfully provisioned!")
        print("Knowledge base successfully provisioned!")
        return True

    except Exception as e:
        log.error(f"Failed to download knowledge base bundle: {e}")
        if os.path.exists(archive_name):
            try:
                os.remove(archive_name)
            except OSError:
                pass
        return False


def ensure_data_ready() -> bool:
    """Ensure data is ready. In Streamlit UI, displays a user-friendly spinner."""
    if is_data_ready():
        return True

    # Check for Streamlit context
    in_streamlit = "streamlit" in sys.modules
    bundle_url = os.getenv("DATA_BUNDLE_URL", DEFAULT_BUNDLE_URL)

    # Attempt to read from st.secrets if available
    try:
        import streamlit as st

        if "DATA_BUNDLE_URL" in st.secrets:
            bundle_url = st.secrets["DATA_BUNDLE_URL"]
    except Exception:
        pass

    if in_streamlit:
        import streamlit as st

        with st.status(
            "Initializing MatSci-RAG knowledge base (~120 MB)...", expanded=True
        ) as status:
            st.write("Downloading pre-indexed ChromaDB and literature facts...")
            success = download_and_extract_bundle(bundle_url)
            if success and is_data_ready():
                status.update(label="Knowledge base ready!", state="complete", expanded=False)
                return True
            else:
                status.update(label="Failed to download knowledge base.", state="error")
                st.error(
                    f"Could not download knowledge base from `{bundle_url}`. "
                    "Please verify `DATA_BUNDLE_URL` in Streamlit secrets or release assets."
                )
                return False
    else:
        return download_and_extract_bundle(bundle_url)
