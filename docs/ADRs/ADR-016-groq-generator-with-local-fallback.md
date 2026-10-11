# ADR-016: Groq-Hosted Generator with Local Fallback, and a Separate Verifier

**Status:** Accepted
**Date:** 2026-10-11
**Phase:** Phase 10
**Supersedes:** the generator decision in ADR-015 (its measurements stand)

---

## Context

ADR-015 measured that generation, not retrieval, was the bottleneck: Answer Relevancy stayed at or
near zero for every Flan-T5 configuration, even when the retrieved passages were rated highly
relevant, and `flan-t5-large` didn't help. The self-check was also weak, because the same small
model checked its own answer and approved a wrong one.

Two ways to get a stronger generator:

| Option | For | Against |
|---|---|---|
| **Groq-hosted model** | Much larger models than a laptop GPU can run; free tier; ~1–2 s per answer; no downloads | Answers depend on an external API |
| Local 7–8B model | Fully offline | ~5 GB download, tight GPU memory, slower, and still far smaller |

## Decision

- **Generator:** `qwen/qwen3.8-27b` via Groq (`GROQ_GEN_MODEL`).
- **Fallback:** if Groq is unavailable (no key, 5xx, rate limit, timeout, empty reply), the same
  prompt goes to local `flan-t5-base` (`GEN_MODEL_NAME`) and a warning is logged. Setting
  `GEN_BACKEND=local` forces fully offline operation.
- **Verifier:** a different model, `openai/gpt-oss-20b` (`GROQ_VERIFY_MODEL`), which must quote
  its supporting sentence.
- **Judge** unchanged: `openai/gpt-oss-120b`. Generator, verifier and judge are three different
  models, so no model grades its own output.
- **Query expansion on by default** in the app (when a Groq key is set), because with this
  generator it measurably raises Answer Relevancy.

Generation lives in `src/rag/generation.py`, which imports torch and transformers lazily, so the
routing logic is unit-tested without models or network access (`tests/test_generation.py`).

## Measurement

Full-text index, the same 7 questions, same judge (`docs/results/v2/fullpaper_qwen_*.csv`):

| Generator | Expansion | Faithfulness | Answer Relevancy | Context Precision |
|---|:---:|---:|---:|---:|
| flan-t5-base | | 0.857 | 0.071 | 0.586 |
| flan-t5-base | on | 0.857 | 0.000 | 0.571 |
| **qwen3.8-27b** | | **1.000** | **0.429** | 0.543 |
| **qwen3.8-27b** | **on** | 0.857 | **0.714** | 0.543 |

Neither run fell back to the local model. Retrieval is unchanged, so the Context Precision
differences are judge noise (about ±0.03–0.04).

On the anode-degradation question, Flan-T5 had answered from the wrong passage while the right
one sat at rank 2. Qwen names Ni coarsening and explains the Gibbs–Thomson mechanism, and the
verifier quotes the source sentence. The remaining "I don't know" answers (3 of 7 without
expansion) are on questions where the retrieved passages genuinely don't contain the answer.

## Consequences

- Answer quality improves roughly 6×, or 10× with expansion, on Answer Relevancy, with no loss
  of faithfulness.
- The answer path depends on Groq by default. The fallback keeps the app working, but answers
  quietly get worse when it triggers; the warning in the log is the signal.
- Groq retires models (ADR-010 update). Because model names are configuration, swapping one is
  an env-var change plus one eval run.
