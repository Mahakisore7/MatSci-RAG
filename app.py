"""Root entrypoint for MatSci-RAG UI.

Compatible with Hugging Face Spaces, Streamlit Community Cloud, and local execution:
    streamlit run app.py
"""

import os
import runpy
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

target_script = os.path.join(PROJECT_ROOT, "src", "ui", "app.py")
runpy.run_path(target_script, run_name="__main__")
