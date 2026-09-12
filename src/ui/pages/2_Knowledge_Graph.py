import json
import os
import tempfile

import networkx as nx
import streamlit as st
from pyvis.network import Network

# pyrefly: ignore [missing-import]
from src.rag.ner_utils import clean_entities

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")

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
            paper_id, label=short_title, title=title, group="paper", size=25, color="#3b82f6"
        )

        # Add Material Nodes (Red)
        for mat in record["materials"]:
            G.add_node(
                mat, label=mat, title=f"Material: {mat}", group="material", size=15, color="#ef4444"
            )
            G.add_edge(paper_id, mat, color="#9ca3af")

        # Add Device Nodes (Green)
        for dev in record["devices"]:
            G.add_node(
                dev, label=dev, title=f"Device: {dev}", group="device", size=15, color="#10b981"
            )
            G.add_edge(paper_id, dev, color="#9ca3af")

        # Add Experiment Nodes (Orange)
        for exp in record["experiments"]:
            G.add_node(
                exp,
                label=exp,
                title=f"Experiment: {exp}",
                group="experiment",
                size=15,
                color="#f59e0b",
            )
            G.add_edge(paper_id, exp, color="#9ca3af")

    return G


def render_pyvis(G, physics=True):
    """Converts NetworkX to PyVis and renders HTML.

    Uses cdn_resources='remote' so pyvis does NOT generate a local lib/
    folder. Uses Python's tempfile so no workspace artifact is left behind.
    """
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0e1117",  # Match Streamlit dark mode
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
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        tmp_path = f.name
    net.write_html(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        html_string = f.read()
    os.unlink(tmp_path)  # Clean up immediately — no leftover artifacts

    return html_string


# ── UI Layout ────────────────────────────────────────────────────────────────

st.title("🕸️ SOFC Knowledge Graph")
st.markdown(
    "Explore the relationships between papers and their extracted Materials, "
    "Devices, and Experiments. \n"
    "Nodes are sized and colored by type. Hover over nodes to see full titles."
)

data = load_and_clean_graph_data()

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
    st.header("Graph Filters")
    view_mode = st.radio(
        "View Mode", ["Global View (Top N Papers)", "Paper-Centric", "Entity-Centric"]
    )

    max_papers = None
    target_paper = None
    target_entity = None

    if view_mode == "Global View (Top N Papers)":
        max_papers = st.slider(
            "Number of papers to plot", min_value=10, max_value=200, value=50, step=10
        )
        st.caption(
            "Rendering more than 100 papers at once may cause browser "
            "lag due to physics simulation."
        )
    elif view_mode == "Paper-Centric":
        target_paper = st.selectbox("Select a Paper", all_titles)
    else:
        target_entity = st.selectbox("Select an Entity", all_entities)

    st.markdown("---")
    enable_physics = st.checkbox("Enable Physics Simulation", value=True)

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown("🔵 **Paper** (Center node)")
    st.markdown("🔴 **Material**")
    st.markdown("🟢 **Device**")
    st.markdown("🟠 **Experiment**")


# Build and render graph
with st.spinner("Building Graph..."):
    G = build_networkx_graph(
        data, max_papers=max_papers, target_paper=target_paper, target_entity=target_entity
    )

    if G.number_of_nodes() == 0:
        st.warning("No nodes matched your filter criteria.")
    else:
        # Render HTML
        st.markdown(f"**Nodes:** {G.number_of_nodes()} | **Edges:** {G.number_of_edges()}")
        html = render_pyvis(G, physics=enable_physics)
        st.iframe(html, height=715, scrolling=False)
