"""Entry point: uv run streamlit run src/ui/app.py"""

import os
import sys

# Streamlit puts only this file's folder (src/ui) on sys.path, so `import src...`
# fails without the project root. Every page runs inside this process, so this
# one insert covers them all.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st  # noqa: E402

from src.ui.theme import apply_theme  # noqa: E402

st.set_page_config(
    page_title="MatSci-RAG",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.logo("docs/assets/logo.svg", size="large")

pages = st.navigation(
    {
        "Research": [
            st.Page("views/chat.py", title="Assistant", icon=":material/forum:", default=True),
        ],
        "Explore": [
            st.Page("views/analytics.py", title="Analytics", icon=":material/monitoring:"),
            st.Page("views/knowledge_graph.py", title="Knowledge Graph", icon=":material/hub:"),
        ],
    }
)
pages.run()
