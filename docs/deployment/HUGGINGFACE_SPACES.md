# Deploying MatSci-RAG to Hugging Face Spaces

This guide walks you through deploying **MatSci-RAG** to **Hugging Face Spaces** using the Streamlit SDK.

---

## Why Hugging Face Spaces?

As documented in [ADR-020](../ADRs/ADR-020-hugging-face-spaces-deployment.md), Hugging Face Spaces is the ideal hosting platform for MatSci-RAG:

| Feature | Streamlit Community Cloud | Hugging Face Spaces (Free Tier) |
|---|---|---|
| **Memory (RAM)** | 1 GB (hard limit ~3 GB) | **16 GB RAM** |
| **vCPUs** | 1 vCPU | **2 vCPUs** |
| **Disk Space** | Temporary / minimal | **50 GB free disk** |
| **Git LFS** | Not natively supported | **Natively supported (free LFS)** |
| **Risk of OOM** | High (crashes on heavy models/indices) | Zero (plenty of headroom) |
| **Cost** | Free | Free |

---

## Prerequisites

1. A [Hugging Face](https://huggingface.co) account.
2. A Hugging Face User Access Token (with `Write` permissions) from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. A Groq API Key from [console.groq.com](https://console.groq.com).
4. `git` and `git-lfs` installed locally.

---

## Step 1: Create a New Space on Hugging Face

1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in the details:
   - **Space name**: `MatSci-RAG` (or your preferred name)
   - **License**: `Apache 2.0`
   - **Select the Space SDK**: **Streamlit**
   - **Space hardware**: `CPU basic · 2 vCPU · 16 GB · FREE`
   - **Visibility**: `Public` (or `Private`)
3. Click **Create Space**.

---

## Step 2: Configure Space Secrets

Your Space needs your API keys to run generation and verification without committing keys to git:

1. In your newly created Space, click **Settings** (top right tab).
2. Scroll to the **Variables and secrets** section.
3. Click **New secret** and add:
   - **Name**: `GROQ_API_KEY`
   - **Value**: `gsk_...` (your Groq API key)
4. (Optional) If accessing gated models, add:
   - **Name**: `HF_TOKEN`
   - **Value**: `hf_...` (your Hugging Face token)

---

## Step 3: Deploy the Repository to Hugging Face

### Method A: Automated CLI Script (Recommended)

Run the deployment script from the project root:

```bash
uv run python scripts/deploy_to_hf.py --space <your-hf-username>/MatSci-RAG
```

The script will:
1. Verify `git-lfs` is installed and tracking ChromaDB and binary assets.
2. Ensure required knowledge base files (`data/chroma_db`, `data/structured_facts.jsonl`, `data/sofc_corpus.jsonl`) are staged.
3. Configure the Hugging Face Space git remote.
4. Push code and LFS-tracked data to your Space.

---

### Method B: Manual Git Push

If you prefer using standard git commands:

1. Add your Hugging Face Space as a git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/MatSci-RAG
   ```

2. Track data files with Git LFS (already configured in `.gitattributes`):
   ```bash
   git lfs install
   git add .gitattributes requirements.txt app.py
   git add -f data/chroma_db/ data/structured_facts.jsonl data/sofc_corpus.jsonl data/full_papers.jsonl
   git commit -m "chore: prepare deployment for Hugging Face Spaces"
   ```

3. Push to your Hugging Face Space:
   ```bash
   git push space main
   ```

---

## Step 4: Verification

1. Once pushed, navigate to your Space on Hugging Face:
   `https://huggingface.co/spaces/<your-hf-username>/MatSci-RAG`
2. You will see the **Building** status in the header:
   - Hugging Face automatically executes `pip install -r requirements.txt`.
   - Thanks to the CPU-wheel index configuration (`--extra-index-url https://download.pytorch.org/whl/cpu`), the build completes in ~1–2 minutes.
3. Once the build completes, the status changes to **Running** and the full MatSci-RAG UI will be interactive!

---

## Architecture in the Cloud

When running on Hugging Face Spaces:
- **UI Entrypoint**: [app.py](../../app.py) forwards to [src/ui/app.py](../../src/ui/app.py).
- **Secrets**: Streamlit secrets (`st.secrets`) are bridged to `os.environ` on boot.
- **Retrieval**: BGE-small dense retriever + BM25 okapi + MatSciBERT entity boost + cross-encoder re-ranking run inside the 16 GB container.
- **Generation & Verification**: Queries Groq API (`qwen/qwen3.8-27b` and `openai/gpt-oss-20b`) for fast (~1s) grounded responses.
