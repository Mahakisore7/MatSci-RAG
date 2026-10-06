# ADR-014: A REST API over the RAG Pipeline

**Status:** Accepted
**Date:** 2026-10-05
**Phase:** Phase 10 — Productionization

---

## Context

Through Phase 9 the pipeline was reachable only through the Streamlit UI. Other programs — a
lab notebook, a batch evaluation, another service — had no way to call it, and the UI's
behaviour (which options exist, what a "source" is) was defined only implicitly by Streamlit code.

---

## Decision

Add a thin FastAPI layer, `src/api/main.py`:

- `POST /query` → `{answer, abstained, verification, sources[]}`
- `GET /health` → `{status, generator}`
- Interactive Swagger docs at `/docs`, generated from the Pydantic schemas.

Run it with:

```bash
uv run uvicorn src.api.main:app --port 8000
```

It **reuses** `answer_query()` / `verify_answer()` unchanged — no retrieval or generation logic is
duplicated, so the UI and API cannot drift apart. Input is validated at this boundary: question
length 1–1000 chars, chat-history roles restricted to `user`/`assistant`, history capped at 20
messages; anything else is a `422`.

Two design details:
- **Lazy pipeline import** (`get_pipeline()`): importing the module does not load ChromaDB or
  any model, so `tests/test_api.py` mocks the pipeline and runs in CI in under a second.
- **A lock around generation**: the generator is one shared in-process model and FastAPI runs
  sync endpoints in a thread pool, so concurrent requests are serialized rather than interleaving
  `generate()` calls on the same weights — throughput traded for safety, the right call for a
  single-process service.

| Alternative | Why rejected |
|---|---|
| Flask | No built-in request validation or OpenAPI docs |
| Streamlit only | Not callable by other programs |
| `async def` endpoint | `generate()` is CPU/GPU-bound and blocking; it would stall the event loop |

### Containerization — considered and dropped

A CPU-only Docker image + `docker-compose.yml` was built and verified (2026-10-05), but removed
at the project owner's request: the image was ~3 GB and its build cache ~20 GB of local disk,
a poor trade for a project that runs on one machine. If deployment is needed later, the main
lesson to reuse is that `uv.lock` pins CUDA torch, so a CPU image must install
`torch==2.6.0` from the PyTorch CPU index **before** the other locked packages; otherwise
`accelerate`/`sentence-transformers` pull the ~4 GB CUDA stack back in from PyPI.

---

## Consequences

**Positive**
- The pipeline is callable over HTTP with a documented, validated contract.
- 9 new contract tests run in CI without models or data.

**Negative / accepted trade-offs**
- The API and the Streamlit UI each load their own copy of the models if both run at once.
- Requests are processed one at a time.
