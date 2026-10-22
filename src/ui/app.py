"""Entry point: uv run streamlit run src/ui/app.py"""

import os
import sys

# Streamlit puts only this file's folder (src/ui) on sys.path, so `import src...`
# fails without the project root. Every page runs inside this process, so this
# one insert covers them all.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st  # noqa: E402

# Bridge Streamlit Secrets (Hugging Face Spaces / Streamlit Cloud) into os.environ
try:
    for key, val in st.secrets.items():
        if isinstance(val, str):
            if key not in os.environ:
                os.environ[key] = val.strip()
            if key.upper() not in os.environ:
                os.environ[key.upper()] = val.strip()
        elif hasattr(val, "items"):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, str):
                    if sub_key not in os.environ:
                        os.environ[sub_key] = sub_val.strip()
                    if sub_key.upper() not in os.environ:
                        os.environ[sub_key.upper()] = sub_val.strip()
except Exception:
    pass

from src.core.data_bootstrapper import ensure_data_ready  # noqa: E402
from src.ui.theme import apply_theme  # noqa: E402

st.set_page_config(
    page_title="MatSci-RAG",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
ensure_data_ready()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logo_path = os.path.join(ROOT_DIR, "docs", "assets", "logo.svg")
if os.path.exists(logo_path):
    st.logo(logo_path, size="large")


UI_DIR = os.path.dirname(__file__)
pages = st.navigation(
    {
        "Research": [
            st.Page(
                os.path.join(UI_DIR, "views", "chat.py"),
                title="Assistant",
                icon=":material/forum:",
                default=True,
            ),
        ],
        "Explore": [
            st.Page(
                os.path.join(UI_DIR, "views", "analytics.py"),
                title="Analytics",
                icon=":material/monitoring:",
            ),
            st.Page(
                os.path.join(UI_DIR, "views", "knowledge_graph.py"),
                title="Knowledge Graph",
                icon=":material/hub:",
            ),
        ],
    }
)
pages.run()
