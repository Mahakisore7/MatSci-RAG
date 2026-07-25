# Phase 8 — Stretch Improvements (Optional)

> **Where this file lives:** `docs/build-guide/08_stretch_improvements.md`
> **Diagram referenced below lives at:** `docs/build-guide/diagrams/phase8_flow.png`

---

## 1. Overview — What This Phase Is and Why It's Different From the Others

Every phase up to this point was required — skip one and the next phase's code won't run. This phase is genuinely optional, and that changes how to approach it: **pick what's valuable given the time you actually have left, don't feel obligated to do all three.** A project with Phases 0-7 fully solid and zero stretch items is a complete, defensible project. A project that rushed through Phases 0-7 to leave time for stretch items is a worse project, not a better one.

Each of the three upgrades below traces directly back to a specific, documented limitation identified in this project's literature review (the base paper's own reported failure modes) — so even if you only *describe* these in a final report as "future work, motivated by X," that's a completely legitimate, honest thing to write. Actually implementing one or two is a bonus on top of an already-complete project, not a requirement to make it complete.

By the end of this phase (to whatever extent you choose to pursue it) you will have:

- Optionally, one or more of: a self-verification step, uncertainty-aware abstention, or hybrid retrieval
- A final, clean project state: up-to-date `README.md`, complete `RESULTS.md`, a readable `git log`, tagged `v1.0.0`

---

## 2. Where Each Upgrade Slots Into What You Already Built

![Phase 8 flow](diagrams/phase8_flow.png)

The blue/navy boxes are everything you already built in Phases 4-7 — completely unchanged. The copper/red boxes are the three optional additions below, shown exactly where each one would slot into that existing flow. Notice none of them replace anything — they're genuinely additive, which is why you can pick any subset (including none) without breaking what already works.

---

## 3. File Structure — What Changes (Depends on What You Pick)

```
MatSci-RAG/
├── src/
│   ├── rag_query.py           <- MODIFIED (if doing 8.1 or 8.2)
│   └── hybrid_retrieval.py    <- NEW (only if doing 8.3)
├── README.md                   <- UPDATED (final housekeeping, Section 6)
├── RESULTS.md                  <- FINALIZED (Section 6)
└── ...
```

---

## 4. Git Versioning for This Phase

The same pattern as every previous phase, but **one branch per individual upgrade**, not one branch for all of Phase 8 — this keeps each optional addition independently reviewable and revertable:

```bash
git checkout main
git pull
git checkout -b phase-8-<short-name>
# ... implement, test, commit ...
git checkout main
git merge phase-8-<short-name>
git tag v0.8.<n>
```

Don't skip the branch-per-feature habit just because you're near the end — this is exactly when a broken experiment could otherwise take down a working `main`.

---

## 5. The Three Upgrades, in Priority Order (Cheapest First)

### 5.1 Self-verification step

**Traces back to:** the base paper's lack of any automatic answer-checking step, and directly mirrors the technique ChatExtract (Polak & Morgan) uses to reach ~90% precision through follow-up verification prompts.

*(Full code — this is a genuinely complete, ready-to-use addition.)*

Add to `src/rag_query.py`:

```python
def verify_answer(query: str, answer: str, retrieved: list[dict]) -> str:
    """Asks the same LLM a follow-up question: does the retrieved context
    actually support the answer it just gave? A second, independent check
    before the answer is trusted."""
    context = "\n\n".join(p["text"] for p in retrieved)
    verify_prompt = (
        f"Context: {context}\n\n"
        f"Claim: \"{answer}\" (answering the question: \"{query}\")\n\n"
        "Does the context actually support this claim? Answer yes or no, "
        "and quote the exact supporting sentence if yes."
    )
    inputs = gen_tokenizer(verify_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = gen_model.generate(**inputs, max_new_tokens=100)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

Branch: `phase-8-self-verification`. Test it the same way as Phase 5: run it on a few real questions and read whether the verification step's yes/no judgment actually matches your own reading of the source.

### 5.2 Uncertainty-aware abstention

**Traces back to:** the exact multi-entity confusion failure mode reported by *both* the base paper (Maharana et al.) and ChatExtract — when a passage mentions several materials, the system risks silently picking the wrong one.

*(Structure given in full; the "how many is too many" threshold is a judgment call for you to make based on what you observe in your own corpus — noted honestly, not a gap in the code itself.)*

```python
def detect_ambiguity(retrieved: list[dict], structured_facts: list[dict]) -> list[str] | None:
    """Checks whether the retrieved passages' source abstracts mention more
    than one distinct material in the structured facts table (Phase 3).
    Returns a list of candidate materials if ambiguous, None if not."""
    source_ids = {p["source_id"] for p in retrieved}
    candidates = set()
    for fact in structured_facts:
        if fact["source_id"] in source_ids:
            candidates.update(fact["materials"])

    # Judgment call: what counts as "too many" for your specific corpus and
    # question style? Start with a simple threshold and adjust based on what
    # you actually observe when testing against your Phase 6 questions.
    if len(candidates) > 1:
        return list(candidates)
    return None
```

Wire this in before `build_prompt()`: if `detect_ambiguity()` returns candidates, show the user all of them with their sources instead of letting the generator silently pick one. Branch: `phase-8-uncertainty-abstention`.

### 5.3 Hybrid retrieval (BM25 + dense)

**Traces back to:** a general, well-documented weakness of pure dense (embedding) search — it can miss an exact string match (a specific alloy name or chemical formula) because embeddings prioritize semantic meaning over exact wording.

```bash
uv add rank-bm25
```

*(Structure given in full — combining scores is simple; production systems often use a proper re-ranker model instead of the equal-weighted average shown here, noted honestly as the natural next step beyond this version.)*

```python
# src/hybrid_retrieval.py
from rank_bm25 import BM25Okapi
import json

passages = json.load(open("data/passages.json"))
tokenized_corpus = [p["text"].split() for p in passages]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_retrieve(query: str, k: int = 5) -> list[dict]:
    scores = bm25.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [passages[i] for i in top_indices]

def hybrid_retrieve(query: str, k: int = 5) -> list[dict]:
    """Combines dense and keyword retrieval by a simple equal-weighted score
    average. A more sophisticated version would use a trained cross-encoder
    re-ranker instead -- a reasonable next step beyond this, not included
    here to keep this addition genuinely simple to reason about."""
    dense_results = retrieve(query, k=k)  # from rag_query.py
    bm25_results = bm25_retrieve(query, k=k)
    combined = {p["source_id"]: p for p in dense_results + bm25_results}
    return list(combined.values())[:k]
```

Branch: `phase-8-hybrid-retrieval`.

---

## 6. Final Housekeeping — Do This Regardless of Which (If Any) Stretch Items You Pursue

- [ ] `README.md` at the project root is up to date: what the project does, how to set it up (`uv sync`), how to run each script, links to `RESULTS.md`, and the "CI Limitations" note from Phase 7
- [ ] `RESULTS.md` contains both the NER F1 comparison (Phase 2) and the RAG hit-rate report (Phase 6), plus notes on any Phase 8 items you implemented
- [ ] `git log --oneline --all` reads as a clean, understandable history — skim it yourself and confirm a stranger could follow what happened and why
- [ ] Final tag marking project completion for this review cycle:

```bash
git tag v1.0.0
git push origin main --tags
```

---

## 7. ✅ Final Project Checkpoint

- [ ] Every phase's individual checkpoint (Phases 0-7) was genuinely met, not skipped
- [ ] `RESULTS.md` has real, honestly-reported numbers — not best-case cherry-picks
- [ ] The Streamlit app runs and the CI badge is green
- [ ] You can explain, in your own words, every box in every diagram across all 8 phases — if any box still feels like a black box, that's worth revisiting before calling this done

---

## 8. On Handing Any of This to an IDE Agent

All three upgrades above are complete enough to implement directly. The one genuinely open judgment call — the ambiguity threshold in Section 5.2 — is a good, narrow thing to discuss with an agent once you've looked at real examples from your own corpus: "here are 5 cases where my corpus had 2 vs. 5 vs. 10 candidate materials for a query — help me think through a sensible threshold" is a much better prompt than asking it to invent a number with no data behind it.

---

*This closes the build guide. Every phase file from `00` through `08` is now complete — the project this guide describes is the same one whose objective, base paper, and novelty argument were established back in the Zeroth Review document. Everything since has been building the thing that document proposed.*
