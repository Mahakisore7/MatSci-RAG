import json
import os
import tempfile
from html import escape

import networkx as nx
import streamlit as st
from pyvis.network import Network

# pyrefly: ignore [missing-import]
from src.rag.ner_utils import clean_entities
from src.ui.theme import DEVICE, EXPERIMENT, MATERIAL, PAPER, hero, legend

FACTS_PATH = "data/structured_facts.jsonl"


@st.cache_data
def load_and_clean_graph_data():
    """Loads facts and cleans entities for plotting."""
    if not os.path.exists(FACTS_PATH):
        return []

    data = []
    with open(FACTS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # Clean entities using our existing NER cleaning logic
            record["materials"] = clean_entities(record.get("materials", []))
            record["devices"] = clean_entities(record.get("devices", []))
            record["experiments"] = clean_entities(record.get("experiments", []))
            data.append(record)
    return data


def build_networkx_graph(data, max_papers=None, target_paper=None, target_entity=None):
    """Builds a NetworkX graph based on the filter criteria."""
    G = nx.Graph()

    # Filter data based on mode
    if target_paper:
        data = [d for d in data if d["title"] == target_paper]
    elif target_entity:
        data = [
            d
            for d in data
            if target_entity in d["materials"]
            or target_entity in d["devices"]
            or target_entity in d["experiments"]
        ]
    elif max_papers:
        data = data[:max_papers]

    for record in data:
        paper_id = record["source_id"].split("/")[-1]
        title = record["title"]

        # Truncate title for node label
        short_title = title[:40] + "..." if len(title) > 40 else title

        # Add Paper Node (Blue, larger)
        G.add_node(
            paper_id, label=short_title, title=escape(title), kind="paper", size=25, color=PAPER
        )

        # Add Material Nodes (Red)
        for mat in record["materials"]:
            G.add_node(
                mat,
                label=mat,
                title=f"Material: {escape(mat)}",
                kind="material",
                size=15,
                color=MATERIAL,
            )
            G.add_edge(paper_id, mat, color="#475569")

        # Add Device Nodes (Green)
        for dev in record["devices"]:
            G.add_node(
                dev,
                label=dev,
                title=f"Device: {escape(dev)}",
                kind="device",
                size=15,
                color=DEVICE,
            )
            G.add_edge(paper_id, dev, color="#475569")

        # Add Experiment Nodes (Orange)
        for exp in record["experiments"]:
            G.add_node(
                exp,
                label=exp,
                title=f"Experiment: {escape(exp)}",
                kind="experiment",
                size=15,
                color=EXPERIMENT,
            )
            G.add_edge(paper_id, exp, color="#475569")

    return G


def render_pyvis(G, physics=True):
    """Converts NetworkX to PyVis and renders HTML.

    Uses cdn_resources='remote' so pyvis does NOT generate a local lib/
    folder. Uses Python's tempfile so no workspace artifact is left behind.
    """
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0f172a",  # matches the sidebar navy
        font_color="white",
        notebook=False,
        cdn_resources="remote",  # Load vis.js from CDN, no local lib/
    )

    # Configure physics for better layout, or turn it off for massive graphs
    if physics:
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)
    else:
        net.toggle_physics(False)

    net.from_nx(G)

    # Write to a proper OS temp file, read it back, then delete immediately
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        tmp_path = f.name
    net.write_html(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        html_string = f.read()
    os.unlink(tmp_path)  # Clean up immediately — no leftover artifacts

    return html_string


# ── UI Layout ────────────────────────────────────────────────────────────────

data = load_and_clean_graph_data()

hero(
    "Explore",
    "Knowledge graph",
    "Papers linked to the materials, devices and experiments MatSciBERT extracted from them. "
    "Hover a node for its full name; drag to rearrange.",
)

if not data:
    st.error(f"Could not load data from `{FACTS_PATH}`. Has the ingestion pipeline run?")
    st.stop()

# Build lists for dropdowns
all_titles = sorted({d["title"] for d in data})
all_entities = set()
for d in data:
    all_entities.update(d["materials"])
    all_entities.update(d["devices"])
    all_entities.update(d["experiments"])
all_entities = sorted(all_entities)

with st.sidebar:
    st.subheader("View")
    view_mode = (
        st.segmented_control(
            "View mode",
            ["All", "Paper", "Entity"],
            default="All",
            label_visibility="collapsed",
        )
        or "All"
    )

    max_papers = None
    target_paper = None
    target_entity = None

    if view_mode == "All":
        max_papers = st.slider(
            "Number of papers to plot", min_value=10, max_value=200, value=50, step=10
        )
        st.caption(
            "Rendering more than 100 papers at once may cause browser "
            "lag due to physics simulation."
        )
    elif view_mode == "Paper":
        target_paper = st.selectbox("Select a Paper", all_titles)
    else:
        target_entity = st.selectbox("Select an Entity", all_entities)

    enable_physics = st.toggle("Physics simulation", value=True)

    st.divider()
    st.subheader("Legend")
    legend(
        [("Paper", PAPER), ("Material", MATERIAL), ("Device", DEVICE), ("Experiment", EXPERIMENT)]
    )


# Build and render graph
with st.spinner("Building graph…"):
    G = build_networkx_graph(
        data, max_papers=max_papers, target_paper=target_paper, target_entity=target_entity
    )

    if G.number_of_nodes() == 0:
        st.warning("No nodes matched your filter criteria.")
    else:
        # Render HTML
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", G.number_of_nodes(), border=True)
        c2.metric("Edges", G.number_of_edges(), border=True)
        c3.metric(
            "Papers shown",
            sum(1 for _, a in G.nodes(data=True) if a.get("kind") == "paper"),
            border=True,
        )
        html = render_pyvis(G, physics=enable_physics)
        st.iframe(html, height=715)
