# ADR-006: Use Raw `model.generate()` over LangChain Pipeline for Generation

**Date:** 2026-08-11
**Status:** Accepted
**Phase:** 5 — RAG Generation

---

## Context

Phase 5 requires connecting the ChromaDB retriever (built in Phase 4) to a generative LLM to produce answers grounded in retrieved context. The original project plan specified using `google/flan-t5-base` loaded through a `transformers` pipeline, then wrapped in a LangChain `HuggingFacePipeline` and connected to a `create_stuff_documents_chain`.

We are running **Transformers v5.14.1** and **LangChain v1.3.14** (2026 releases), which diverged significantly from the 2023 API surface described in most tutorials and the project guide.

---

## Options Evaluated

### Option A: LangChain `HuggingFacePipeline` + `create_stuff_documents_chain` (Original Plan)
This is the documented approach from LangChain's own tutorials: wrap the HuggingFace model in a `pipeline()`, then pass it to `HuggingFacePipeline`, then use `create_stuff_documents_chain` with a `ChatPromptTemplate`.

**Problems encountered (all fatal):**
1. `from langchain.chains import create_retrieval_chain` — `ModuleNotFoundError`. LangChain v1.3 moved these to `langchain_classic`, which is a separate backcompat package.
2. After switching to `langchain_classic.chains`, importing it triggered a **Windows silent fatal crash** (`ACCESS_VIOLATION`). Root cause: `langchain_classic` imports `langchain_text_splitters`, which imports `sentence_transformers`, which imports `pyarrow` and `sklearn` — and on Windows, loading `sklearn` *after* `torch` is already in memory causes a DLL conflict that instantly kills the interpreter with no error message.
3. Even after fixing the import order, `pipeline("text2text-generation", ...)` raised `KeyError: Unknown task text2text-generation`. Transformers v5 removed this task name from its registry. T5-based Seq2Seq models must now use `"text-generation"` or be called directly.
4. After switching to `"text-generation"`, the model itself (`T5ForConditionalGeneration`) raised: `T5ForConditionalGeneration is not supported for text-generation`. T5 is an **encoder-decoder** (Seq2Seq) model; the `text-generation` pipeline is designed only for **decoder-only** models (like GPT, Llama, etc.).
5. Even if the pipeline succeeded, accessing `chain.combine_docs_chain` raised `AttributeError: 'RunnableSequence' object has no attribute 'combine_docs_chain'`. LangChain's LCEL (v1.x) does not expose this internal attribute on composed chains.

Every path through LangChain's wrapper layers was blocked by a different version incompatibility.

### Option B: Raw `model.generate()` via PyTorch (CHOSEN)
Call the HuggingFace model directly using the standard PyTorch API (`AutoTokenizer` + `AutoModelForSeq2SeqLM.generate()`). This is the **stable, low-level API** that has not changed since HuggingFace 4.x, is guaranteed to work with T5, and requires no integration wrappers.

The LangChain retriever (ChromaDB) is still used for document retrieval. Only the **generation step** bypasses LangChain.

---

## Decision

Use **Option B**: direct `model.generate()` for LLM inference. Keep LangChain only for the ChromaDB retrieval layer.

---

## Detailed Technical Explanation of the Implementation

### Step 1: Retrieval (unchanged from Phase 4)
The user's query is prefixed with the BGE instruction `"Represent this question for searching relevant documents: "` and passed to `retriever.invoke()`. ChromaDB returns the top-5 most semantically similar chunks.

### Step 2: Prompt Construction
We build the full text prompt from scratch using a Python function `build_prompt()`:

```python
def build_prompt(query: str, retrieved_docs: list) -> str:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.metadata.get("title", "Unknown Title")
        text = doc.page_content.strip()
        context_parts.append(f"[{i}] {title}\n{text}")
    
    context = "\n\n".join(context_parts)
    
    return (
        "You are an expert materials science AI assistant specializing in SOFCs. "
        "Answer using ONLY the context below. "
        "If not in context, say 'I don't know based on the provided context.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
```

**Why this exact phrasing matters:**
- `"ONLY the context below"` — The word "ONLY" is critical. Without it, Flan-T5 defaults to mixing its pre-trained knowledge with the context, producing answers that *sound* grounded but aren't.
- `"If not in context, say 'I don't know'"` — This removes the implicit pressure to always produce an answer, which is the primary source of LLM hallucination. Flan-T5 was specifically trained on instruction-following tasks like this.
- The domain scoping `"specializing in SOFCs"` narrows the domain so that non-SOFC questions are more likely to trigger the refusal.

### Step 3: Generation (the key change)
```python
# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

# Run inference (no gradient needed — saves GPU memory)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=200)

# Decode back to English
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
```

`truncation=True, max_length=1024` — If the combined prompt (system instruction + 5 retrieved chunks) exceeds 1024 tokens, the tokenizer silently cuts off the end. This is a known risk when using `k=5` chunks, each potentially 500 characters. If answers degrade in quality, reducing `k` to 3 is the first fix.

### Step 4: Lazy LLM Loading
The embedding model loads at module import time (needed for ChromaDB). The LLM model is loaded lazily on the first call to `get_llm()`. This avoids loading 1GB of model weights if the module is imported for testing without being called.

---

## Import Order Fix (Windows-Specific)

On Windows, the following import order causes a fatal `ACCESS_VIOLATION` crash:

```python
import torch  # ❌ Loads CUDA DLLs into memory
from langchain_huggingface import HuggingFaceEmbeddings  # ❌ Then loads sklearn/pyarrow → DLL conflict
```

The correct order on Windows:

```python
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Load sklearn/pyarrow FIRST
import torch  # ✅ CUDA DLLs load after, no conflict
```

This is documented because it produces a **completely silent crash** — no Python traceback, no error message, just exit code 1. It will be needed again if `rag_query.py` is refactored.

---

## Consequences

- **Positive:** Zero dependency on LangChain version for the generation step. The code will work with any future Transformers version because `AutoModelForSeq2SeqLM.generate()` is part of the stable, backward-compatible HuggingFace API.
- **Positive:** The `build_prompt()` function is fully readable plain Python — no hidden template logic inside LangChain chains.
- **Negative:** We lose LangChain's structured output parsing. The answer is a raw decoded string; post-processing is manual.
- **Negative:** If we want to swap the LLM to a larger model (e.g., Llama-3-8B), we need to update the `AutoModelForSeq2SeqLM` to `AutoModelForCausalLM` and adjust the prompt format, because decoder-only models use a different generation pattern.
- **Known Limitation:** Query 3 ("Which cathode materials show high oxygen reduction activity?") still retrieves battery papers. This is a retrieval problem, not a generation problem, and will be addressed in Phase 6 evaluation or Phase 8 hybrid retrieval.
