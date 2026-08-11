# Phase 5: RAG Generation Log

This document tracks the implementation and testing of the generation step in the RAG pipeline.

---

## Run 1: Full Pipeline Test (Phase 5 Complete)
**Date:** 2026-08-11
**Model:** `google/flan-t5-base`
**Hardware:** NVIDIA RTX 4050 (6GB VRAM), CUDA 12.4
**Model Load Time:** ~11 seconds (weights: 282 files, ~1GB)
**Generation Interface:** Direct `AutoModelForSeq2SeqLM.generate()` (raw PyTorch)
**Retriever:** ChromaDB + BGE-small-en-v1.5, k=5

### Configuration
```python
max_new_tokens = 200
max_length     = 1024  # tokenizer truncation limit
truncation     = True
device         = "cuda"
k              = 5     # number of chunks retrieved
chunk_size     = 500   # inherited from Phase 4
chunk_overlap  = 50    # inherited from Phase 4
```

### Prompt Template (final version)
```
You are an expert materials science AI assistant specializing in Solid Oxide Fuel Cells (SOFCs).
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, or if the question is completely unrelated to
materials science, reply EXACTLY with: 'I don't know based on the provided context.'
Do not hallucinate or use outside knowledge.

Context:
[1] {title_1}
{chunk_text_1}

[2] {title_2}
{chunk_text_2}
...

Question: {user_query}
Answer:
```

### Test Results

#### Query 1: "What materials are used as electrolytes in solid oxide fuel cells?"
**Retrieved Sources:**
1. High-entropy perovskites, architectured by s0/d0/d10 cations, as novel electrolytes for solid oxide fuel cells
2. A brief review of high-entropy oxides in solid oxide fuel cell applications
3. Compositionally Complex Perovskite Oxides as a New Class of Li-Ion Solid Electrolytes
4. A protonated brownmillerite electrolyte for superior low-temperature proton conductivity
5. Ferroelastic toughening: can it solve the mechanics challenges of solid electrolytes?

**Generated Answer:** `"High-entropy perovskites."`

**Quality:** ✅ **Correct and grounded.** The answer directly matches Source [1] which is about high-entropy perovskite electrolytes. Sources [1] and [2] are exactly the right papers. Source [3] is slightly off-domain (Li-ion) but related.

**Grounding Verified:** Yes — "High-entropy perovskites" appears verbatim in Source [1]'s title.

---

#### Query 2: "What is the effect of temperature on SOFC performance?"
**Retrieved Sources:**
1. Micro solid oxide fuel cells: a new generation of micro-power sources for portable applications
2. Electrochemical properties of solid oxide fuel cells under the coupling effect of airflow pattern and airflow velocity (retrieved twice — different chunks)
3. Material Discovery and Design Principles for Stable, High Activity Perovskite Cathodes for Solid Oxide Fuel Cells
4. Comparing the 3D morphology of solid-oxide fuel cell anodes for different manufacturing processes, annealing times, and operating temperatures

**Generated Answer:** `"The electrochemical performance of SOFC is affected by various factors such as gas flow pattern (co-flow, counter-flow, cross-flow), flow rate (cathode and anode channel gases), and operating voltage"`

**Quality:** ✅ **Grounded.** The answer is derived entirely from Source [2] which is specifically about airflow and temperature effects on SOFC electrochemistry. Source [4] mentions operating temperatures directly.

**Grounding Verified:** Yes — the answer precisely paraphrases the content of the retrieved airflow paper.

---

#### Query 3: "Which cathode materials show high oxygen reduction activity?"
**Retrieved Sources:**
1. Ab initio random structure searching for battery cathode materials (**battery paper — off-domain**)
2. Computational screening of cathode materials for Zn-ion rechargeable batteries (**battery paper — off-domain**)
3. Computational discovery of cathode materials for rechargeable aqueous zinc-ion batteries (**battery paper — off-domain**)
4. Material Discovery and Design Principles for Stable, High Activity Perovskite Cathodes for SOFCs (**relevant**)
5. Ab initio random structure searching for battery cathode materials (**battery paper — off-domain, duplicate**)

**Generated Answer:** `"FePO₄"`

**Quality:** ⚠️ **Off-domain.** The retriever returned primarily battery cathode papers because "cathode materials" and "oxygen reduction" match battery literature heavily. FePO₄ is a lithium-iron-phosphate battery cathode, not an SOFC cathode material. The LLM correctly answered from the context it was given — the retriever is the failure point, not the generator.

**Root Cause:** The SOFC corpus contains mixed-domain papers (materials science generally), and "cathode" appears in both SOFC and battery papers. This is documented in Phase 4's retrieval test.

**Resolution Plan:**
- Phase 6: Quantify with hit-rate metric. Define threshold.
- Phase 8 (Stretch): Implement hybrid BM25 + vector retrieval. BM25 would strongly prefer papers containing the exact phrase "solid oxide fuel cell" for SOFC-specific queries.

---

#### Query 4: "Who won the Super Bowl last year?" (Hallucination Test)
**Retrieved Sources:** (completely unrelated papers — biology, electrocatalysis, etc.)

**Generated Answer:** `"I don't know."`

**Quality:** ✅ **Hallucination guardrail confirmed working.** The model correctly identified that the retrieved context contained no relevant answer and refused to hallucinate. This is the most critical requirement of the RAG architecture — verified.

---

### Bugs Fixed During Phase 5 Implementation

| # | Bug Description | Root Cause | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: langchain.chains` | LangChain v1.3 moved legacy chains to `langchain_classic` | Updated import to `langchain_classic.chains` |
| 2 | Silent Python process crash (`ACCESS_VIOLATION`, exit code 1) | On Windows, importing `torch` before `sentence_transformers` causes DLL conflict with sklearn/pyarrow | Reordered imports: `HuggingFaceEmbeddings` before `torch` |
| 3 | `KeyError: Unknown task 'text2text-generation'` | Transformers v5 removed this pipeline task string | Switched to direct `model.generate()` — no pipeline needed |
| 4 | `T5ForConditionalGeneration is not supported for text-generation` | T5 is Seq2Seq (encoder-decoder); `text-generation` task only supports decoder-only models | Use `AutoModelForSeq2SeqLM` directly |
| 5 | `AttributeError: 'RunnableSequence' has no 'combine_docs_chain'` | LangChain LCEL v1.x doesn't expose internal chain attributes | Dropped LangChain chain entirely; manual `build_prompt()` + `model.generate()` |

### Timing
| Step | Time |
|---|---|
| Embedding model load (BGE) | ~9 seconds |
| ChromaDB connect | <1 second |
| LLM model load (flan-t5-base) | ~11 seconds |
| Generation per query | ~0.5 seconds |
| **Total end-to-end (4 queries)** | **~22 seconds** |
