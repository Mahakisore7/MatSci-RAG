# Sample Questions for MatSci-RAG UI

Use these questions in the Streamlit UI (`uv run streamlit run app.py`) to test the different capabilities of the system.

### 1. Specific Fact Extraction (Standard RAG)
These questions test the system's ability to find specific technical facts and synthesize them into an answer.
- *"What is considered a major reason for anode degradation in Ni-YSZ solid oxide fuel cells?"*
- *"What are the advantages of using proton-conducting electrolytes in SOFCs?"*
- *"What is the typical operating temperature range for intermediate-temperature SOFCs (IT-SOFCs)?"*

### 2. Testing Hybrid Retrieval vs Dense (Phase 8 Upgrade)
To see the true power of Phase 8, toggle the sidebar to **"Dense Only"**, ask this question, and see it answer with a battery material (e.g. FePO4). Then toggle to **"Hybrid"** and ask it again to see it correctly use NER filtering to find a SOFC material.
- *"Which cathode materials show high oxygen reduction activity?"*

### 3. Testing Uncertainty Abstention (Phase 8 Upgrade)
This question is intentionally too vague. The system will retrieve papers mentioning many different materials (LSCF, YSZ, etc.). Instead of guessing, it should trigger the ambiguity warning: *"Multiple candidate materials found across sources... Please narrow your question."*
- *"What material was used in the experiment?"*

### 4. Testing Guardrails (Anti-Hallucination)
This tests the system prompt's strict grounding instruction. Since this information is not in the SOFC literature corpus, the model should refuse to answer.
- *"Who won the Super Bowl last year?"*
- *"What is the capital of France?"*
