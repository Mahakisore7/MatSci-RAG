# Phase 3 — NER: Extract Structured Facts from Our Own Corpus

> **Where this file lives:** `docs/build-guide/03_ner_inference.md`
> **Diagram referenced below lives at:** `docs/build-guide/diagrams/phase3_flow.png`

---

## 1. Overview — What This Phase Is and Why It Exists

Phase 2 proved MatSciBERT is a good entity extractor, but it was only ever run against the Matscholar benchmark — a labeled dataset that exists purely for evaluation, not the actual corpus this project cares about. This phase is short but important: it's the moment your fine-tuned model finally does real, useful work on *your* data — the arXiv abstracts from Phase 1 — producing something that didn't exist before: a structured, queryable table of (material, property) mentions pulled automatically out of unstructured prose.

This structured table matters beyond just "being a nice output." It's what will let Phase 5's RAG system optionally enrich its answers with pre-extracted facts rather than relying purely on retrieved free text, and it's a concrete, demoable artifact on its own — "here's a table of materials and properties mined automatically from 400 papers" is a genuinely impressive thing to show, independent of the chatbot built in later phases.

By the end of this phase you will have:

- `data/structured_facts.jsonl` — one row per abstract, each with extracted materials and properties
- Personal, manual confirmation (same discipline as Phase 1) that the extractions actually look correct
- A small analysis showing which materials/properties appear most often in your corpus — a first genuine piece of insight your project has produced

---

## 2. Phase Data Flow

![Phase 3 flow](diagrams/phase3_flow.png)

Two independent phases' outputs — your corpus (Phase 1) and your trained model (Phase 2) — combine for the first time here. Everything to the left of this phase was preparation; this is the first moment the project produces a genuinely new kind of artifact (structured data, not just text or a trained model file).

---

## 3. File Structure — Before and After This Phase

### Before (end of Phase 2):
```
MatSci-RAG/
├── src/
│   ├── config.py
│   ├── logging_config.py
│   ├── fetch_corpus.py
│   ├── prepare_ner_data.py
│   └── train_ner.py
├── data/
│   ├── corpus.jsonl
│   └── matscholar_processed/
├── models/
│   ├── matscibert_ner/
│   └── bert_baseline_ner/
└── ...
```

### After this phase:
```
MatSci-RAG/
├── src/
│   ├── ...(unchanged)
│   └── run_ner_inference.py         <- NEW
├── tests/
│   └── test_ner_inference.py        <- NEW
├── data/
│   ├── corpus.jsonl
│   ├── matscholar_processed/
│   └── structured_facts.jsonl       <- NEW (generated, gitignored)
└── ...
```

One new script, one new test, one new generated file — this phase stays deliberately small.

---

## 4. Git Versioning for This Phase

```bash
git checkout main
git pull
git checkout -b phase-3-ner-inference
```

**Commit type:** `feat:`
**Tag on completion:** `v0.4.0`

---

## 5. Step-by-Step Implementation

### 5.1 Create `src/run_ner_inference.py`

*(Full code, no gaps — this reuses the already-trained, already-verified model from Phase 2.)*

```python
# src/run_ner_inference.py
"""Applies the fine-tuned MatSciBERT NER model to every abstract in our own
corpus (not the Matscholar benchmark) and produces a structured facts table."""

from transformers import pipeline
import json
from logging_config import get_logger

log = get_logger(__name__)

def run_inference(corpus_path: str, model_dir: str, output_path: str) -> int:
    ner_pipeline = pipeline(
        "token-classification",
        model=model_dir,
        aggregation_strategy="simple",   # merges B-/I- tokens into full entity spans automatically
        device=0,   # use the GPU
    )

    results = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            entities = ner_pipeline(paper["abstract"])
            materials = [e["word"] for e in entities if "MAT" in e["entity_group"]]
            properties = [e["word"] for e in entities if "PRO" in e["entity_group"]]
            results.append({
                "source_id": paper["id"],
                "title": paper["title"],
                "materials": materials,
                "properties": properties,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info(f"Wrote structured facts for {len(results)} abstracts to {output_path}")
    return len(results)

if __name__ == "__main__":
    run_inference("data/corpus.jsonl", "models/matscibert_ner", "data/structured_facts.jsonl")
```

**One dependency worth double-checking before you run this:** the `"MAT" in e["entity_group"]` and `"PRO" in e["entity_group"]` filters assume the label names from Phase 2's `LABELS` list (`B-MAT`, `I-MAT`, `B-PRO`, `I-PRO`, etc.). If your Phase 2 parser (Section 5.2 of that file) ended up producing differently-named tags, update these filter strings to match — this is a small, five-second check, not a rewrite.

### 5.2 Run it

```bash
uv run src/run_ner_inference.py
```

### 5.3 Verify — same discipline as Phase 1, this can't be automated away

```bash
uv run python -c "import json; [print(json.loads(l)) for l in open('data/structured_facts.jsonl').readlines()[:5]]"
```

Read the first 5 results by hand. Do the extracted materials and properties actually correspond to what those specific abstracts are about? A model that's 90% accurate on Matscholar can still produce occasional nonsense on your specific corpus — this is exactly the check that would catch it early.

### 5.4 Bonus: a first piece of real insight from your structured data

*(Optional, but genuinely worth 2 minutes — this is the first moment your project produces something you couldn't have gotten by just reading papers one at a time.)*

```python
# quick_facts_analysis.py
import json
from collections import Counter

materials = Counter()
properties = Counter()

with open("data/structured_facts.jsonl") as f:
    for line in f:
        record = json.loads(line)
        materials.update(record["materials"])
        properties.update(record["properties"])

print("Top 10 most-mentioned materials:")
for mat, count in materials.most_common(10):
    print(f"  {mat}: {count}")

print("\nTop 10 most-mentioned properties:")
for prop, count in properties.most_common(10):
    print(f"  {prop}: {count}")
```

```bash
uv run quick_facts_analysis.py
```

If this list looks sensible for your topic (e.g., real alloy names and real property words appearing at the top), that's a strong, independent confirmation that both Phase 2's training and this phase's inference actually worked correctly together.

### 5.5 Write the test

```python
# tests/test_ner_inference.py
import json
import os

def test_structured_facts_file_exists():
    assert os.path.exists("data/structured_facts.jsonl")

def test_structured_facts_have_required_fields():
    with open("data/structured_facts.jsonl") as f:
        record = json.loads(f.readline())
    for field in ["source_id", "title", "materials", "properties"]:
        assert field in record
```

```bash
uv run pytest tests/test_ner_inference.py -v
```

---

## 6. Git: Commit and Merge

```bash
git add src/run_ner_inference.py tests/test_ner_inference.py
git commit -m "feat: apply fine-tuned MatSciBERT NER to full corpus, build structured facts table"

git checkout main
git merge phase-3-ner-inference
git tag v0.4.0
```

---

## 7. ✅ Phase 3 Checkpoint — Do Not Proceed to Phase 4 Until All Checked

- [ ] `data/structured_facts.jsonl` exists, one row per abstract
- [ ] You have personally read 5 rows and confirmed the extractions look correct
- [ ] The top-10 materials/properties list (Section 5.4) looks sensible for your topic
- [ ] Tests pass
- [ ] Merged to `main`, tagged `v0.4.0`

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `materials` / `properties` lists are always empty | Check the entity_group filter strings match your actual Phase 2 label names (see the note in 5.1) |
| Extractions look nonsensical | Re-check Phase 2's checkpoint was genuinely met (real F1 numbers, not just "code ran") before trusting this phase's output |
| `RuntimeError` about device / CUDA in the pipeline | Confirm `device=0` is valid — if you don't have a GPU available in this exact terminal session, temporarily change to `device=-1` (CPU) to unblock, then investigate the GPU issue separately |

---

## 9. On Handing This to an IDE Agent

Nothing here needs one — this phase applies already-verified components (Phase 1's corpus, Phase 2's model) with a small, fully-specified script. If you do want to extend Section 5.4's analysis (e.g., a bar chart instead of printed text), that's a reasonable, low-risk thing to ask Claude/Gemini for — it's presentation, not correctness-critical logic.

---

**Next file:** `04_rag_indexing.md` — request it once every box in Section 7 is checked.
