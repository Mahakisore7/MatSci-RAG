# Sample Questions for MatSci-RAG UI

Use these questions in the Streamlit UI (`uv run streamlit run app.py`) to test the different capabilities of the system.

### 1. Specific Fact Extraction (Standard RAG)
These questions test the system's ability to find specific technical facts and synthesize them into an answer.
- *"What is considered a major reason for anode degradation in Ni-YSZ solid oxide fuel cells?"*
- *"What are the advantages of using proton-conducting electrolytes in SOFCs?"*
- *"What is the typical operating temperature range for intermediate-temperature SOFCs (IT-SOFCs)?"*

### 2. Testing Hybrid Retrieval vs Dense (Phase 8 Upgrade)
Toggle the sidebar to **"Dense Only"**, ask this question, and see it lean towards battery cathode papers (e.g. FePO4). Then toggle to **"Hybrid"** and ask again — the genuinely relevant SOFC paper ranks noticeably higher (it's boosted, not filtered-in, so this is a ranking shift, not a guarantee of an all-SOFC result set — see [ADR-009](ADRs/ADR-009-ner-aware-retrieval-reranking.md)).
- *"Which cathode materials show high oxygen reduction activity?"*

### 3. Uncertainty Abstention (Phase 8 Upgrade — currently rarely triggers)
This was intended to demo the system refusing to answer an overly vague question by detecting many candidate materials in the retrieved papers. In practice, the underlying NER model's `materials` extraction is too noisy to reliably power this (see ADR-009) — the abstention threshold is deliberately set high so it fails safe instead of misfiring on ordinary questions, which means it will rarely fire at all until NER precision improves. Not recommended as a live demo right now.
- *"What material was used in the experiment?"*

### 4. Testing Guardrails (Anti-Hallucination)
This tests the system prompt's strict grounding instruction. Since this information is not in the SOFC literature corpus, the model should refuse to answer.
- *"Who won the Super Bowl last year?"*
- *"What is the capital of France?"*
