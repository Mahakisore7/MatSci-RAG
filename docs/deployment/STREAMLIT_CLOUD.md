# Deploying MatSci-RAG to Streamlit Community Cloud

This guide provides step-by-step instructions for deploying **MatSci-RAG** to **Streamlit Community Cloud** ([streamlit.io/cloud](https://streamlit.io/cloud)) for free public access.

---

## Architecture Overview

Streamlit Community Cloud runs directly from your GitHub repository:
- **Compute**: 1 vCPU and ~1 GB of allocated RAM on Debian Linux containers.
- **Dependency Management**: Automatically installs [requirements.txt](../../requirements.txt), using the lightweight CPU-only wheel of PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu torch>=2.2.0`).
- **Knowledge Base Bootstrapping**: The vector database and parsed papers (`data_bundle.tar.gz`, 120 MB) are downloaded once on first container boot via [src/core/data_bootstrapper.py](../../src/core/data_bootstrapper.py) and extracted into `data/`.
- **Generation & Verification**: Powered by Groq API (`qwen/qwen3.8-27b` and `openai/gpt-oss-20b`), keeping container memory below 800 MB (well within limits).

---

## Step 1: Upload the Knowledge Base Bundle to GitHub Releases

Because the vector store is ~202 MB uncompressed (exceeding standard GitHub git limits), we package it as a release asset:

1. The bundle `data_bundle.tar.gz` (120 MB) has already been created in your local repository root. (If you ever need to re-create it in the future, run `uv run python scripts/bundle_data.py`).
2. Go to your GitHub repository in your browser:  
   **[https://github.com/Mahakisore7/MatSci-RAG/releases/new](https://github.com/Mahakisore7/MatSci-RAG/releases/new)**
3. Fill in the release details:
   - **Choose a tag**: Type `v1.0.0` and click **Create new tag**.
   - **Release title**: `v1.0.0 — Knowledge Base Bundle`
   - **Description**: `Knowledge base vectors (ChromaDB), parsed corpus, and extracted MatSciBERT facts.`
4. In the box labeled **"Attach binaries by dropping them here or selecting them"**:
   - Drag and drop `data_bundle.tar.gz` from your project folder:  
     `c:\Users\mahakisore\Academics\SEM 5\DDMMS\Project\MatSci-RAG\data_bundle.tar.gz`
5. Click **Publish release**.

> The direct download URL will automatically match:  
> `https://github.com/Mahakisore7/MatSci-RAG/releases/download/v1.0.0/data_bundle.tar.gz`  
> *(This is already the pre-configured default in `src/core/data_bootstrapper.py`).*

---

## Step 2: Push Your Local Changes to GitHub

Commit the deployment configuration, entrypoint, and documentation to your GitHub repository:

```bash
git add requirements.txt app.py src/ docs/ .gitattributes scripts/
git commit -m "feat(deploy): configure Streamlit Community Cloud deployment and data bootstrapper"
git push origin main
```

---

## Step 3: Deploy on Streamlit Community Cloud

1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** (or [streamlit.io/cloud](https://streamlit.io/cloud)) and sign in with your GitHub account.
2. Click the **Create app** button (or **New app**).
3. Select **"I already have an app"**.
4. Configure the deployment settings:
   - **Repository**: `Mahakisore7/MatSci-RAG`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g., `matsci-rag.streamlit.app`).
5. Click **Advanced settings...** (bottom of the form):
   - **Python version**: Select `3.11` or `3.12`.
   - In the **Secrets** text box, paste your Groq API key:
     ```toml
     GROQ_API_KEY = "gsk_..."
     ```
   - *(Optional)* If you host your data bundle elsewhere, you can also specify:
     ```toml
     DATA_BUNDLE_URL = "https://github.com/Mahakisore7/MatSci-RAG/releases/download/v1.0.0/data_bundle.tar.gz"
     ```
6. Click **Save**, then click **Deploy!**.

---

## Step 4: Verification

1. Streamlit Cloud will open the build console:
   - Installing packages from `requirements.txt` (~1–2 minutes).
   - Launching `app.py`.
2. On first boot, the UI will display a status widget:
   `"Initializing MatSci-RAG knowledge base (~120 MB)..."`
3. In ~10–15 seconds, the archive downloads and extracts into `data/`.
4. Your application will then display:
   - The research assistant chat interface.
   - Dynamic sidebar with model and retrieval stats.
   - Analytics comparison page.
   - Interactive Knowledge Graph visualization.
