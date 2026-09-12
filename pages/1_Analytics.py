"""
Analytics Dashboard Page — pages/1_Analytics.py

Streamlit multi-page app page that shows corpus statistics, NER entity
distributions, and RAG evaluation scores. Runs without loading any heavy
ML models — all data is read from pre-built artifact files on disk.

Run from project root:
    uv run streamlit run app.py
Then click "Analytics" in the sidebar navigation.
"""

import json
import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MatSci-RAG Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Data paths ────────────────────────────────────────────────────────────────
CORPUS_PATH = "data/sofc_corpus.jsonl"
FACTS_PATH = "data/structured_facts.jsonl"
RAGAS_CSV = "docs/results/ragas_evaluation.csv"
CHROMA_DIR = "data/chroma_db"


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_corpus():
    records = []
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


@st.cache_data
def load_facts():
    records = []
    if os.path.exists(FACTS_PATH):
        with open(FACTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


@st.cache_data
def load_ragas():
    if os.path.exists(RAGAS_CSV):
        return pd.read_csv(RAGAS_CSV)
    return None


def is_real_entity(token: str) -> bool:
    """Filter out obvious NER noise tokens (punctuation, subword fragments, etc.)"""
    token = token.strip()
    # too short
    if len(token) <= 1:
        return False
    # subword fragments like ##ite, ##ing
    if token.startswith("##"):
        return False
    # pure punctuation or numbers
    if re.fullmatch(r"[\W\d]+", token):
        return False
    # generic English stopwords that slip through
    noise = {
        "solid",
        "fuel",
        "cell",
        "cells",
        "oxide",
        "sof",
        "or",
        "and",
        "the",
        "of",
        "in",
        "for",
        "to",
        "at",
        "by",
        "is",
        "are",
        "was",
        "with",
        "on",
        "as",
        "from",
        "that",
        "this",
        "an",
        "a",
        "be",
        "not",
        "it",
        "its",
        "can",
        "may",
        "high",
        "low",
        "good",
        "new",
        "current",
        "potential",
        "specific",
        "distinct",
        "limited",
        "order",
        "mix",
        "avoid",
        "leakage",
        "resistance",
        "micron",
        "screened",
        "predicted",
        "cathode",
        "materials",
        "material",
        "applications",
        "temperature",
        "performance",
    }
    if token.lower() in noise:
        return False
    return True


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 MatSci-RAG Analytics Dashboard")
st.markdown(
    "Corpus statistics, NER entity distributions, and RAG evaluation scores — "
    "all from pre-built data files, no model loading required."
)
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
corpus = load_corpus()
facts = load_facts()
ragas_df = load_ragas()

# ── Section 1: Corpus Overview ────────────────────────────────────────────────
st.header("1. Corpus Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Papers", len(corpus))
with col2:
    total_words = sum(len(r.get("abstract", "").split()) for r in corpus)
    st.metric("Total Words (abstracts)", f"{total_words:,}")
with col3:
    avg_words = total_words // max(len(corpus), 1)
    st.metric("Avg Words / Paper", avg_words)
with col4:
    # Estimate chunks (500 char chunks, ~4 chars/word → ~125 words/chunk)
    est_chunks = sum(max(1, len(r.get("abstract", "")) // 450) for r in corpus)
    st.metric("Estimated Chunks (indexed)", est_chunks)

st.markdown("---")

# ── Section 2: NER Entity Analysis ───────────────────────────────────────────
st.header("2. NER Entity Analysis")
st.caption(
    "Entities extracted by the fine-tuned MatSciBERT model. "
    "Noise tokens (punctuation, subword fragments, generic words) are filtered out for display."
)

all_materials: list[str] = []
all_devices: list[str] = []
all_experiments: list[str] = []
all_values: list[str] = []
entity_counts_per_paper: list[dict] = []

papers_with_materials = 0
papers_with_devices = 0

for rec in facts:
    mats = [t for t in rec.get("materials", []) if is_real_entity(t)]
    devs = [t for t in rec.get("devices", []) if is_real_entity(t)]
    exps = [t for t in rec.get("experiments", []) if is_real_entity(t)]
    vals = [t for t in rec.get("values", []) if is_real_entity(t)]

    all_materials.extend(mats)
    all_devices.extend(devs)
    all_experiments.extend(exps)
    all_values.extend(vals)

    if mats:
        papers_with_materials += 1
    if devs:
        papers_with_devices += 1

    entity_counts_per_paper.append(
        {
            "title": rec.get("title", "")[:60] + "...",
            "materials": len(mats),
            "devices": len(devs),
            "experiments": len(exps),
            "values": len(vals),
            "total": len(mats) + len(devs) + len(exps) + len(vals),
        }
    )

# Summary metrics
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Material Entities Extracted", len(all_materials))
with col_b:
    st.metric("Device Entities Extracted", len(all_devices))
with col_c:
    st.metric("Experiment Entities Extracted", len(all_experiments))
with col_d:
    st.metric("Value Entities Extracted", len(all_values))

col_e, col_f = st.columns(2)
with col_e:
    st.metric(
        "Papers With ≥1 Material Entity",
        f"{papers_with_materials} / {len(facts)} ({100*papers_with_materials//max(len(facts),1)}%)",
    )
with col_f:
    st.metric(
        "Papers With ≥1 Device Entity",
        f"{papers_with_devices} / {len(facts)} ({100*papers_with_devices//max(len(facts),1)}%)",
    )

# Entity type distribution pie chart
st.subheader("Entity Type Distribution")
pie_data = {
    "Type": ["Materials", "Devices", "Experiments", "Values"],
    "Count": [len(all_materials), len(all_devices), len(all_experiments), len(all_values)],
}
fig_pie = px.pie(
    pie_data,
    names="Type",
    values="Count",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Extracted NER entities by type (noise-filtered)",
)
st.plotly_chart(fig_pie, width="stretch")

# Top material tokens
if all_materials:
    st.subheader("Top 20 Most Common Material Tokens")
    mat_counter = Counter(t.lower() for t in all_materials)
    top_mats = mat_counter.most_common(20)
    fig_bar = px.bar(
        x=[t[0] for t in top_mats],
        y=[t[1] for t in top_mats],
        labels={"x": "Token", "y": "Count"},
        color_discrete_sequence=["#6366f1"],
        title="Top 20 material tokens (after noise filtering)",
    )
    st.plotly_chart(fig_bar, width="stretch")

# Top device tokens
if all_devices:
    st.subheader("Top 15 Most Common Device Tokens")
    dev_counter = Counter(t.lower() for t in all_devices)
    top_devs = dev_counter.most_common(15)
    fig_dev = px.bar(
        x=[t[0] for t in top_devs],
        y=[t[1] for t in top_devs],
        labels={"x": "Token", "y": "Count"},
        color_discrete_sequence=["#10b981"],
        title="Top 15 device tokens",
    )
    st.plotly_chart(fig_dev, width="stretch")

st.markdown("---")

# ── Section 3: RAG Evaluation Results ────────────────────────────────────────
st.header("3. RAG Quality Evaluation")

if ragas_df is not None:
    # Drop the AVERAGE row for per-question chart
    q_df = ragas_df[ragas_df["question"] != "AVERAGE"].copy()
    avg_row = ragas_df[ragas_df["question"] == "AVERAGE"]

    # Aggregate metrics
    if not avg_row.empty:
        a = avg_row.iloc[0]
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric(
                "Faithfulness",
                f"{float(a['faithfulness']):.3f}",
                help="1.0 = model never hallucinates; 0.0 = pure hallucination",
            )
        with col_r2:
            st.metric(
                "Answer Relevancy",
                f"{float(a['answer_relevancy']):.3f}",
                help="1.0 = answer fully addresses the question",
            )
        with col_r3:
            st.metric(
                "Context Precision",
                f"{float(a['context_precision']):.3f}",
                help="1.0 = all retrieved chunks are relevant",
            )

    # Per-question chart
    if not q_df.empty:
        st.subheader("Per-Question Scores")
        fig_scores = go.Figure()
        short_q = [q[:50] + "..." if len(q) > 50 else q for q in q_df["question"]]
        for metric, color in [
            ("faithfulness", "#6366f1"),
            ("answer_relevancy", "#f59e0b"),
            ("context_precision", "#10b981"),
        ]:
            fig_scores.add_trace(
                go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=short_q,
                    y=q_df[metric].astype(float),
                    marker_color=color,
                )
            )
        fig_scores.update_layout(
            barmode="group",
            xaxis_tickangle=-30,
            yaxis=dict(range=[0, 1.1], title="Score"),
            title="Faithfulness / Relevancy / Precision per benchmark question",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig_scores, width="stretch")

    # Detailed table
    st.subheader("Detailed Results Table")
    display_df = q_df[
        ["question", "answer", "faithfulness", "answer_relevancy", "context_precision"]
    ].copy()
    display_df.columns = ["Question", "Answer", "Faithfulness", "Relevancy", "Precision"]
    display_df["Answer"] = display_df["Answer"].str[:80] + "..."
    st.dataframe(display_df, width="stretch")

    st.info(
        "**Faithfulness = 1.0** is the core design goal — the model never fabricates. "
        "Low Relevancy/Precision scores are expected for an abstracts-only corpus and "
        "motivate full-paper ingestion (planned extension)."
    )
else:
    st.warning(
        "No evaluation results found. Run `uv run python scripts/evaluate_ragas.py` "
        "to generate scores."
    )

st.markdown("---")

# ── Section 4: System Info ────────────────────────────────────────────────────
st.header("4. System Configuration")

sys_col1, sys_col2 = st.columns(2)
with sys_col1:
    st.markdown(
        """
**Retrieval Stack**
| Component | Value |
|---|---|
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Embedding Dimensions | 384 |
| Vector Database | ChromaDB |
| Chunking | 500 chars, 50 overlap |
| BM25 Library | `rank-bm25` |
| Fusion | Reciprocal Rank Fusion |
"""
    )
with sys_col2:
    st.markdown(
        """
**Generation Stack**
| Component | Value |
|---|---|
| Generator | `google/flan-t5-base` |
| Parameters | 250M |
| NER Model | `m3rg-iitd/matscibert` |
| NER Test F1 | 0.0915 |
| Verification | 2nd Flan-T5 call (self-check) |
| Judge LLM (eval) | `groq/compound-mini` |
"""
    )
