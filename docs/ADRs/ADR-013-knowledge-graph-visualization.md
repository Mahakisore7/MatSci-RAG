# ADR-013: Knowledge Graph Visualization

**Status:** Accepted  
**Date:** 2026-09-12  
**Phase:** Phase 9 Extension  
**Branch:** `feature/knowledge-graph`

---

## Context

The pipeline uses the MatSciBERT NER model to extract specific entities (`materials`, `devices`, `experiments`) from the SOFC papers. These entities are currently saved in `data/structured_facts.jsonl` and used for NER-aware retrieval (ADR-009). 

However, visualizing this data as a Knowledge Graph allows users to intuitively explore the literature: seeing how papers cluster around specific materials, or identifying which experiments are commonly run on specific solid oxide fuel cell architectures. We need a way to render this interactively within the existing Streamlit UI.

---

## Decision

### Graph Library: `pyvis` + `networkx`

**Chosen:** `networkx` for graph construction, `pyvis` for HTML rendering.

**Alternatives considered:**
| Library | Pros | Why Rejected |
|---|---|---|
| `streamlit-agraph` | Native Streamlit component | Slower updates; complex callback model; less smooth physics. |
| `plotly` (Network) | Already installed in project | No physics simulation; requires manually calculating node (x,y) positions using `networkx` layout algorithms first, which looks static and cluttered. |
| `pyvis` ✅ | Excellent physics simulation, highly interactive, easy HTML export | Requires adding a new dependency; writes temp HTML files. |

**Rationale:**
`pyvis` produces dynamic, physics-based network graphs (using vis.js under the hood) that automatically untangle themselves. It is highly interactive (zoom, pan, hover tooltips) and can be easily embedded into Streamlit using `streamlit.components.v1.html(..., scrolling=False)`. `networkx` is used to build the intermediate graph logic (nodes, edges, attributes) efficiently before passing it to `pyvis`.

---

### UI Design & Performance Strategy

Rendering a full graph of 400 papers + ~2,000 entities in the browser with active physics simulation will cause the browser to freeze or lag heavily.

To mitigate this, the UI enforces a filtering strategy:
1. **Global View:** By default, only the top N papers (default 50) are plotted. The user can increase this via a slider, but they do so with a warning about browser lag.
2. **Paper-Centric View:** The user selects a single paper from a dropdown, and the graph renders only that paper and its direct entity connections.
3. **Entity-Centric View:** The user selects a specific material/device, and the graph renders all papers that mention it.

Additionally, a **"Enable Physics Simulation"** toggle allows the user to turn off the expensive physics calculations once the graph has settled into a pleasing layout.

---

### Data Cleaning

The `structured_facts.jsonl` data often contains noisy sub-word tokens from the NER model (e.g. `##sr0`, commas, punctuation). The visualization pipeline intercepts this data and applies the `clean_entities()` function from `src.rag.ner_utils.py` *before* plotting. This ensures the graph only displays clean, meaningful entities and prevents the graph from being cluttered with hundreds of floating punctuation nodes.

---

## Consequences

- Added `pyvis` and `networkx` to project dependencies (`pyproject.toml`).
- Added a new Streamlit tab: `pages/2_Knowledge_Graph.py` (now `src/ui/views/knowledge_graph.py`).
- Enhances user capability to perform "exploratory" literature reviews without issuing explicit text queries.
