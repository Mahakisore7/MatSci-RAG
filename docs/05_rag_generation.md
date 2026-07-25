# Phase 5 — RAG: Generation with Grounded Answers

> **Where this file lives:** `docs/build-guide/05_rag_generation.md`
> **Diagram referenced below lives at:** `docs/build-guide/diagrams/phase5_flow.png`

---

## 1. Overview — What This Phase Is and Why It Exists

Every phase so far has been building toward this one. Phase 1 gave you a corpus. Phases 2-3 gave you structured facts. Phase 4 gave you fast, meaning-based search over that corpus. This phase is where those pieces actually combine into the thing the whole project is named after: **Retrieval-Augmented Generation** — an LLM that answers questions using *only* text it was just handed, not whatever it happened to memorize during its own training.

This is also the phase where the project's central motivation (reducing hallucination) either actually holds up or doesn't. Everything from here on is testable in a very direct way: ask a question, read the answer, check whether it's actually grounded in the printed sources. There's no abstraction left to hide behind — this is the real behavior of your system.

By the end of this phase you will have:

- `src/rag_query.py` with a working `answer_query()` function
- The ability to ask a real question and get back both an answer *and* the exact passages it was generated from
- Personal confirmation, across several different questions, that the answers are actually using the retrieved context rather than ignoring it

---

## 2. Phase Data Flow

![Phase 5 flow](diagrams/phase5_flow.png)

Notice where `passages` branches into two arrows: one into the prompt (so the LLM can use the content), and one out as `sources` (so a human can verify it later, in Phase 7's UI). This is the entire mechanism that makes an answer *trustworthy* rather than merely *plausible* — the same object serves both roles, so there's no way for the displayed source to be disconnected from what the model actually saw.

---

## 3. File Structure — Before and After This Phase

### Before (end of Phase 4):
```
MatSci-RAG/
├── src/
│   └── ... (fetch_corpus.py, ..., build_index.py, quick_retrieval_test.py)
├── data/
│   ├── faiss.index
│   └── passages.json
└── ...
```

### After this phase:
```
MatSci-RAG/
├── src/
│   ├── ... (unchanged)
│   └── rag_query.py                <- NEW
├── tests/
│   └── test_rag_generation.py      <- NEW
└── ...
```

No new data files this phase — everything needed already exists from Phase 4; this phase is pure logic sitting on top of it.

---

## 4. Git Versioning for This Phase

```bash
git checkout main
git pull
git checkout -b phase-5-rag-generation
```

**Commit type:** `feat:`
**Tag on completion:** `v0.6.0`

---

## 5. Step-by-Step Implementation

### 5.1 A quick concept check — why Flan-T5-base, specifically

Plain T5 is trained on a generic text-to-text objective (translation, summarization, etc.) but was never explicitly trained to *follow instructions*. **Flan-T5** is the same architecture, further fine-tuned specifically to follow natural-language instructions well. That distinction matters directly for this phase: our prompt template's core instruction — *"answer using ONLY the context below, say you don't know if it's not there"* — is exactly the kind of instruction Flan-T5 was trained to actually obey, whereas a raw T5 model would be far less reliable at respecting it. This isn't a minor pick; it's the reason the hallucination-reduction strategy in Section 5.2 below has a reasonable chance of working at all.

### 5.2 A quick concept check — why the prompt is phrased exactly this way

```
"Answer the question using ONLY the context below. If the answer is not
contained in the context, say you don't know."
```

This single sentence is doing real work, not just being polite. It's a direct, deliberate mitigation against hallucination — explicitly telling the model it's allowed to say "I don't know" removes the implicit pressure a model otherwise has to always produce *some* confident-sounding answer, even when it shouldn't. This exact technique traces back to the literature you reviewed earlier in this project (both the base paper and ChatExtract lean on variations of this idea) — it's not a Claude invention, it's standard practice in the field.

### 5.3 Create `src/rag_query.py`

*(Full code — no ambiguity here, both `sentence-transformers` and `transformers`' T5 interfaces are small and stable.)*

```python
# src/rag_query.py
"""The core RAG pipeline: embed a query, retrieve the most relevant passages,
build a grounding prompt, and generate an answer from ONLY that context.
See Sections 5.1-5.2 above for why Flan-T5 and this specific prompt phrasing."""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from logging_config import get_logger

log = get_logger(__name__)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss.index")
passages = json.load(open("data/passages.json"))

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

log.info(f"RAG generation model loaded on device: {device}")

def retrieve(query: str, k: int = 5) -> list[dict]:
    qvec = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(qvec, dtype="float32"), k)
    return [passages[i] for i in indices[0]]

def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(f"[{i+1}] {p['title']}: {p['text']}" for i, p in enumerate(retrieved))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

def answer_query(query: str, k: int = 5) -> tuple[str, list[dict]]:
    retrieved = retrieve(query, k)
    prompt = build_prompt(query, retrieved)
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = gen_model.generate(**inputs, max_new_tokens=200)
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    log.info(f"Q: {query} -> A: {answer[:80]}...")
    return answer, retrieved

if __name__ == "__main__":
    q = "What hardness values have been reported for high entropy alloys?"
    answer, sources = answer_query(q)
    print("ANSWER:", answer)
    print("\nSOURCES:")
    for s in sources:
        print("-", s["title"])
```

**Two implementation details worth understanding, not just copying:**
- `.to(device)` appears on the model *and* on the tokenized inputs — both need to live on the same device (GPU or CPU) or PyTorch will throw a mismatch error. This is the actual mechanism that makes your 4050 get used instead of silently falling back to CPU.
- `truncation=True, max_length=1024` means if your retrieved context (all `k` passages combined) is too long, it gets **silently cut off** at the end — not an error, just quietly missing information. This is directly relevant if you increase `k` or work with longer passages later; see Section 8's troubleshooting entry on this exact failure mode.

### 5.4 Run it

```bash
uv run src/rag_query.py
```

### 5.5 Verify — this is the real test of the whole project so far

Don't just run the one hardcoded question. Open a Python shell or add more lines to the `if __name__` block, and try **3-4 different questions** covering different parts of your corpus:

```python
answer, sources = answer_query("Your second test question here")
print(answer)
```

For each one, read the answer *and* the sources side by side. Ask yourself: is this answer actually supported by what's in those sources, or does it sound like it's drifting into the model's own general knowledge? This distinction — grounded vs. merely plausible — is the entire point of the project, and this is the first phase where you can actually observe it directly.

### 5.6 Write the test

```python
# tests/test_rag_generation.py
import sys
sys.path.append("src")

def test_answer_query_returns_answer_and_sources():
    from rag_query import answer_query
    answer, sources = answer_query("hardness of high entropy alloys", k=3)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(sources) == 3
```

```bash
uv run pytest tests/test_rag_generation.py -v
```

---

## 6. Git: Commit and Merge

```bash
git add src/rag_query.py tests/test_rag_generation.py
git commit -m "feat: implement RAG query pipeline with retrieval, prompt construction, and grounded generation"

git checkout main
git merge phase-5-rag-generation
git tag v0.6.0
```

---

## 7. ✅ Phase 5 Checkpoint — Do Not Proceed to Phase 6 Until All Checked

- [ ] `answer_query()` runs without errors and returns both an answer and a list of sources
- [ ] You tested at least 3-4 different questions, not just the one hardcoded example
- [ ] For each, you personally confirmed the answer is actually grounded in the printed sources
- [ ] Tests pass
- [ ] Merged to `main`, tagged `v0.6.0`

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Answer is generic / ignores context entirely | Print the full `prompt` string (add a `log.info(prompt)` line) and check the context actually got filled in correctly before blaming the model |
| Answer is truncated or seems to ignore later passages | `max_length=1024` may be cutting off context — reduce `k` (fewer passages) or check total prompt length |
| Answer says "I don't know" even for questions you know are answerable | Check Phase 4's retrieval is actually returning the right passages for that specific query — the generator can only work with what it's given |
| Model runs on CPU despite having a GPU | Confirm `torch.cuda.is_available()` still returns `True` (re-run Phase 0's `verify_gpu.py` if unsure) |

---

## 9. On Handing This to an IDE Agent

Nothing here needs one for correctness — the Hugging Face T5 interface used is small, stable, and fully covered above. If you want to experiment with a different, larger generation model later (a stretch idea, not required), that's a reasonable thing to discuss with an agent — but the core retrieve-then-generate logic in this file should stay exactly this simple; resist any temptation (from an agent or otherwise) to add complexity here before Phase 6 has actually measured whether this simple version works well enough.

---

**Next file:** `06_evaluation.md` — request it once every box in Section 7 is checked.
