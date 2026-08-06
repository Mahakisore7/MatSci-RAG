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
| [ADR-002](ADR-002-matscibert-vs-bert-baseline.md) | MatSciBERT vs BERT-base as Fine-Tuning Backbone | Accepted |
| [ADR-003](ADR-003-sofc-corpus-for-inference.md) | Fetch SOFC-Specific arXiv Corpus for Inference | Accepted |
| [ADR-004](ADR-004-fix-character-offset-bug.md) | Fix Windows `\r\n` Character-Offset Bug in Parser | Accepted |

## Naming Convention

`ADR-{number}-{short-slug}.md`

Numbers are sequential and never reused. If a decision is reversed, the old ADR is marked `Superseded` and a new ADR is created.
