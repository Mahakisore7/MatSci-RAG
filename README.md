# 🔬 MatSci-RAG: Domain-Specific RAG for Materials Science

This project is a complete, end-to-end Retrieval-Augmented Generation (RAG) system built specifically for Materials Science literature, focusing on Solid Oxide Fuel Cells (SOFCs). 

It demonstrates how to upgrade a "generic" RAG pipeline into a **Domain-Specific NER-RAG** system using a custom-trained MatSciBERT model to extract entities (materials, devices, experiments) and use them to intelligently filter vector searches.

## 🚀 Final Project Status (v1.0.0)
**Status:** COMPLETE (Phases 0-8)

### Key Features Implemented:
1. **Data Ingestion:** 400 SOFC papers scraped from ArXiv.
2. **Custom NER Model:** Fine-tuned MatSciBERT on the Bosch SOFC-Exp corpus to extract domain entities.
3. **Information Extraction:** Processed all 400 papers to extract structured facts (`structured_facts.jsonl`).
4. **Vector Database:** 1,330 text chunks embedded into ChromaDB using `BAAI/bge-small-en-v1.5`.
5. **LLM Generator:** Local inference using `google/flan-t5-base` with strict grounding prompts.
6. **Self-Verification & Abstention:** The model double-checks its own answers and refuses to answer ambiguous queries.
7. **Hybrid Retrieval (NER Filter + BM25 + Dense):** The core innovation. By combining exact keyword matching with NER-based domain filtering (restricting searches to SOFC papers), the system completely eliminates the cross-domain "battery vs fuel cell" confusion common in generic RAG.
8. **Streamlit UI:** A fully functional browser chat interface for easy interaction.

---

## 🛠️ How to Install and Run

### 1. Setup the Environment
This project uses `uv` for reproducible, lightning-fast dependency management.
```bash
# Clone the repository
git clone https://github.com/Mahakisore7/MatSci-RAG.git
cd MatSci-RAG

# Install all dependencies exactly as locked
uv sync
```

### 2. Run the Chat UI
```bash
uv run streamlit run app.py
```
A browser window will open at `http://localhost:8501`. You can toggle between **Dense Only** (generic RAG) and **Hybrid** (our advanced pipeline) in the sidebar to see the difference in performance.

### 3. Run the Test Suite
```bash
uv run pytest tests/ -v
```

---

## 📊 Results & Documentation

- **[RESULTS.md](RESULTS.md):** Detailed benchmark results showing MatSciBERT outperforming generic BERT, 100% retrieval hit-rates, and the final impact of Phase 8 Hybrid Retrieval.
- **[docs/ADRs/](docs/ADRs/):** Architecture Decision Records explaining why every major technical choice was made.
- **[docs/Phases/](docs/Phases/):** The step-by-step build guide.

> **Note on CI Limitations:** The GitHub Actions CI pipeline runs linting (`ruff`) and end-to-end retrieval smoke tests. However, it does not run the NER training tests because the fine-tuned model weights (~400MB) are stored locally in `models/` and are excluded via `.gitignore`.
