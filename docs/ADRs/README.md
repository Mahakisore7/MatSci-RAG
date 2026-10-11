# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the MatSci-RAG project.

## What is an ADR?

An ADR is a short document that captures an important technical or design decision, along with the context and reasoning that led to it. The goal is to create a historical record so that anyone (including your future self) can understand *why* the project was built the way it was — not just *how*.

## Format

Each ADR follows this standard structure:

- **Status:** `Accepted` | `Deprecated` | `Superseded by ADR-XXX`
- **Context:** What was the situation or problem being solved?
- **Decision:** What was chosen?
- **Reasoning:** Why was this chosen over the alternatives?
- **Consequences:** What are the positive and negative trade-offs?

## Index

| ADR | Title | Status |
|---|---|---|
| [ADR-001](ADR-001-use-sofc-exp-dataset.md) | Use SOFC-Exp Dataset for NER Training | Accepted |
| [ADR-002](ADR-002-matscibert-vs-bert-baseline.md) | MatSciBERT vs BERT-base as Fine-Tuning Backbone | Accepted (F1 numbers superseded by ADR-004) |
| [ADR-003](ADR-003-sofc-corpus-for-inference.md) | Fetch SOFC-Specific arXiv Corpus for Inference | Accepted |
| [ADR-004](ADR-004-fix-character-offset-bug.md) | Fix Windows `\r\n` Character-Offset Bug in Parser | Accepted |
| [ADR-005](ADR-005-advanced-rag-stack.md) | Advanced RAG Stack — ChromaDB + BGE over FAISS + MiniLM | Accepted |
| [ADR-006](ADR-006-raw-model-generate-over-langchain-pipeline.md) | Use Raw `model.generate()` over LangChain Pipeline for Generation | Accepted |
| [ADR-007](ADR-007-streamlit-ui-filewatcher-fix.md) | Streamlit for UI and `fileWatcherType=none` for Windows Compatibility | Accepted |
| [ADR-008](ADR-008-phase8-bugfixes-import-order-double-import.md) | Phase 8 Bug Fixes — Import Order, Double Import | Accepted (NER filter section superseded by ADR-009) |
| [ADR-009](ADR-009-ner-aware-retrieval-reranking.md) | NER-Aware Retrieval Reranking (Replacing the Broken Metadata Filter) | Accepted |
| [ADR-010](ADR-010-multi-turn-memory-and-llm-judge-evaluation.md) | Multi-Turn Conversation Memory + LLM-Judge RAG Evaluation | Accepted |
| [ADR-011](ADR-011-llm-query-expansion.md) | LLM-Based Query Expansion | Accepted |
| [ADR-012](ADR-012-full-paper-ingestion.md) | Full Paper Ingestion via PDF Download + Section Parsing | Accepted |
| [ADR-013](ADR-013-knowledge-graph-visualization.md) | Knowledge Graph Visualization | Accepted |
| [ADR-014](ADR-014-rest-api.md) | A REST API over the RAG Pipeline (Docker considered and dropped) | Accepted |
| [ADR-015](ADR-015-generator-size-and-full-corpus-evaluation.md) | Generator Size — flan-t5-large Evaluated and Not Adopted | Accepted |
| [ADR-016](ADR-016-groq-generator-with-local-fallback.md) | Groq-Hosted Generator with Local Fallback, and a Separate Verifier | Accepted |
| [ADR-017](ADR-017-fuel-cell-corpus-filter.md) | Fuel-Cell Corpus Filter — Built, Measured, Not the Default | Accepted (optional) |

## Naming Convention

`ADR-{number}-{short-slug}.md`

Numbers are sequential and never reused. If a decision is reversed, the old ADR is marked `Superseded` and a new ADR is created.
