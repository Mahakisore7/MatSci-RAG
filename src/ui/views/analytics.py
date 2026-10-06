"""Analytics page: corpus statistics, NER entity distributions and evaluation scores.

Reads only pre-built artifacts from disk, so it never loads a model.
"""

import glob
import json
import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.config import GEN_MODEL_NAME, GROQ_EXPANSION_MODEL, GROQ_JUDGE_MODEL
from src.ui.theme import CHART_COLORS, DEVICE, EXPERIMENT, MATERIAL, PAPER, hero, section

CORPUS_PATH = "data/sofc_corpus.jsonl"
FULL_PAPERS_PATH = "data/full_papers.jsonl"
FACTS_PATH = "data/structured_facts.jsonl"
EVAL_GLOB = "docs/results/v2/*.csv"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
}


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_corpus() -> list[dict]:
    return _read_jsonl(CORPUS_PATH)


@st.cache_data
def load_facts() -> list[dict]:
    return _read_jsonl(FACTS_PATH)


@st.cache_data
def count_full_papers() -> int:
    return len(_read_jsonl(FULL_PAPERS_PATH))


@st.cache_data
def load_evaluations() -> dict[str, pd.DataFrame]:
    """One DataFrame per eval run, keyed by a readable configuration name.

    Expects files named <abstracts|fullpaper>_<generator>_exp-<on|off>.csv.
    """
    runs = {}
    for path in sorted(glob.glob(EVAL_GLOB)):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.fullmatch(r"(abstracts|fullpaper)_(\w+)_exp-(on|off)", stem)
        if not m:
            continue
        corpus_name, gen, exp = m.groups()
        name = (
            f"{'Full text' if corpus_name == 'fullpaper' else 'Abstracts'} · flan-t5-{gen}"
            f"{' · expansion' if exp == 'on' else ''}"
        )
        runs[name] = pd.read_csv(path)
    return runs


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
        title_font=dict(size=15, color="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def pct(n: int, d: int) -> str:
    return f"{100 * n // max(d, 1)}%"


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


# ── Data ─────────────────────────────────────────────────────────────────────
corpus = load_corpus()
facts = load_facts()
evaluations = load_evaluations()
n_full = count_full_papers()

hero(
    "Explore",
    "Analytics",
    "What is in the corpus, what the NER model extracted from it, and how the full "
    "pipeline scores under an LLM judge.",
    chips=[
        ("papers", str(len(corpus))),
        ("with full text", str(n_full)),
        ("papers tagged", str(len(facts))),
        ("eval runs", str(len(evaluations))),
    ],
)

# ── 1. Corpus ────────────────────────────────────────────────────────────────
section("01", "Corpus")
total_words = sum(len(r.get("abstract", "").split()) for r in corpus)
on_topic = sum(
    1
    for r in corpus
    if re.search(r"fuel cell|SOFC|solid oxide", r.get("title", "") + r.get("abstract", ""), re.I)
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Papers", len(corpus), border=True)
c2.metric("Full text parsed", f"{n_full} ({pct(n_full, len(corpus))})", border=True)
c3.metric("Abstract words", f"{total_words:,}", border=True)
c4.metric(
    "Explicitly about fuel cells",
    f"{on_topic} ({pct(on_topic, len(corpus))})",
    border=True,
    help="Title or abstract mentions 'fuel cell', 'SOFC' or 'solid oxide'. The arXiv keyword "
    "search also returned batteries, catalysis and some off-topic work.",
)

# ── 2. NER ───────────────────────────────────────────────────────────────────
section(
    "02",
    "Extracted entities",
    "From the fine-tuned MatSciBERT model. Punctuation, sub-word fragments and generic words "
    "are filtered out for display.",
)

all_materials: list[str] = []
all_devices: list[str] = []
all_experiments: list[str] = []
all_values: list[str] = []
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
    papers_with_materials += bool(mats)
    papers_with_devices += bool(devs)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Materials", f"{len(all_materials):,}", border=True)
m2.metric("Devices", f"{len(all_devices):,}", border=True)
m3.metric("Experiments", f"{len(all_experiments):,}", border=True)
m4.metric("Values", f"{len(all_values):,}", border=True)

n1, n2 = st.columns(2)
n1.metric(
    "Papers with a material entity",
    f"{papers_with_materials} / {len(facts)} ({pct(papers_with_materials, len(facts))})",
    border=True,
)
n2.metric(
    "Papers with a device entity",
    f"{papers_with_devices} / {len(facts)} ({pct(papers_with_devices, len(facts))})",
    border=True,
    help="Most papers have no device entity, which is why retrieval uses entities as a soft "
    "boost rather than a filter (ADR-009).",
)

left, right = st.columns([2, 3])
with left:
    fig_pie = px.pie(
        names=["Materials", "Devices", "Experiments", "Values"],
        values=[len(all_materials), len(all_devices), len(all_experiments), len(all_values)],
        color=["Materials", "Devices", "Experiments", "Values"],
        color_discrete_map={
            "Materials": MATERIAL,
            "Devices": DEVICE,
            "Experiments": EXPERIMENT,
            "Values": PAPER,
        },
        hole=0.55,
        title="Entities by type",
    )
    st.plotly_chart(style_fig(fig_pie), width="stretch")
with right:
    if all_materials:
        top = Counter(t.lower() for t in all_materials).most_common(15)
        fig_mat = px.bar(
            x=[c for _, c in top][::-1],
            y=[t for t, _ in top][::-1],
            orientation="h",
            labels={"x": "Mentions", "y": ""},
            color_discrete_sequence=[MATERIAL],
            title="Most frequent materials",
        )
        st.plotly_chart(style_fig(fig_mat), width="stretch")

if all_devices:
    top = Counter(t.lower() for t in all_devices).most_common(15)
    fig_dev = px.bar(
        x=[t for t, _ in top],
        y=[c for _, c in top],
        labels={"x": "", "y": "Mentions"},
        color_discrete_sequence=[DEVICE],
        title="Most frequent devices",
    )
    st.plotly_chart(style_fig(fig_dev, height=340), width="stretch")

# ── 3. Evaluation ────────────────────────────────────────────────────────────
section(
    "03",
    "End-to-end quality",
    f"7 benchmark questions scored by an LLM judge ({GROQ_JUDGE_MODEL}). "
    "Scores are only comparable between runs that used the same judge.",
)

if evaluations:
    summary = pd.DataFrame(
        [
            {"Configuration": name, **df[df["question"] == "AVERAGE"][METRICS].iloc[0].to_dict()}
            for name, df in evaluations.items()
        ]
    ).rename(columns=METRIC_LABELS)

    fig_cmp = go.Figure()
    for color, label in zip(CHART_COLORS, METRIC_LABELS.values(), strict=False):
        fig_cmp.add_trace(
            go.Bar(name=label, x=summary["Configuration"], y=summary[label], marker_color=color)
        )
    fig_cmp.update_layout(
        title="Average score by configuration",
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Score"),
    )
    st.plotly_chart(style_fig(fig_cmp, height=420), width="stretch")

    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            label: st.column_config.ProgressColumn(label, min_value=0, max_value=1, format="%.3f")
            for label in METRIC_LABELS.values()
        },
    )

    chosen = st.selectbox("Inspect a run question by question", list(evaluations))
    per_q = evaluations[chosen]
    per_q = per_q[per_q["question"] != "AVERAGE"]
    st.dataframe(
        per_q.rename(columns={"question": "Question", "answer": "Answer", **METRIC_LABELS}),
        hide_index=True,
        width="stretch",
        column_config={
            "Question": st.column_config.TextColumn(width="medium"),
            "Answer": st.column_config.TextColumn(width="medium"),
            **{
                label: st.column_config.NumberColumn(width="small", format="%.2f")
                for label in METRIC_LABELS.values()
            },
        },
    )
else:
    st.info(
        "No evaluation results yet. Run `uv run python scripts/evaluate_ragas.py "
        "--output docs/results/v2/abstracts_large_exp-off.csv`.",
        icon=":material/info:",
    )

# ── 4. Configuration ─────────────────────────────────────────────────────────
section("04", "System configuration")
cfg_left, cfg_right = st.columns(2)
with cfg_left:
    st.markdown(
        """
| Retrieval | |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) |
| Vector store | ChromaDB |
| Chunking | 500 chars (abstracts) · 800 chars (full text), 50 overlap |
| Keyword search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion + NER-overlap boost |
"""
    )
with cfg_right:
    st.markdown(
        f"""
| Generation | |
|---|---|
| Generator | `{GEN_MODEL_NAME}` |
| NER model | `m3rg-iitd/matscibert` (test F1 0.0915) |
| Verification | second generator call (self-check) |
| Query expansion | `{GROQ_EXPANSION_MODEL}` |
| Evaluation judge | `{GROQ_JUDGE_MODEL}` |
"""
    )
