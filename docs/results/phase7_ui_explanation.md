# Phase 7: Streamlit UI + GitHub Actions CI — Complete Explanation

> Written on: 2026-08-12

This document explains Phase 7 in complete detail: what was done, why it was done, what problems we hit, and how they were solved.

---

## What is Phase 7 About?

By the end of Phase 6, our entire AI system was working correctly. We had:
- 400 papers stored as mathematical vectors in ChromaDB
- A system that finds the top 5 relevant papers for any question
- An LLM (Flan-T5) that reads those papers and generates a grounded answer
- A 100% hit rate on 20 specific test questions

**The problem:** None of this was usable by a non-programmer. To use the system, you had to open a terminal and run Python scripts. A mentor or classmate couldn't use it.

**Phase 7 solves two things:**
1. **The UI Problem:** Wrap the pipeline in a website so anyone can use it from a browser.
2. **The CI Problem:** Replace the fake test pipeline we created in Phase 0 (which just printed "Tests will go here") with a real automated test system that runs on GitHub every time code is pushed.

---

## Part 1: The Streamlit Web App (`app.py`)

### What is Streamlit?
Streamlit is a Python library that turns a plain `.py` file into a working website. You don't write any HTML, CSS, or JavaScript. You just write Python and Streamlit creates a web page for you.

### What does our `app.py` do?
When you run `uv run streamlit run app.py`, a website opens at `http://localhost:8501`. The page looks like a ChatGPT-style interface. Here is exactly what happens step by step:

**Step 1: The user opens the browser**
They see:
- A sidebar on the left showing the system architecture (Corpus: 400 SOFC papers, Embedding: BGE-small, Generator: flan-t5-base, Database: ChromaDB)
- A title: "Materials Science Literature Assistant"
- A description explaining what the assistant can answer
- A text input at the bottom: "Ask a question about SOFCs..."

**Step 2: The user types a question and presses Enter**
For example: *"What is considered a major reason for anode degradation in Ni-yttria stabilized zirconia solid oxide fuel cell anodes?"*

**Step 3: A spinner appears**
The user sees a spinning loading icon that says "Searching literature and generating answer..." This takes about 15-20 seconds for the first question (because the Flan-T5 model needs to load into GPU memory). After that, it is much faster.

**Step 4: The answer appears**
The answer is displayed in bold: **"Ni coarsening"**

**Step 5: Source papers appear below the answer**
Below the answer, expandable dropdowns appear. Each one says:
`[1] Ni coarsening in the three-phase solid oxide fuel cell anode - a phase-field simulation study`

When you click it, you can read the full paragraph from the paper that was used to generate the answer, and see the source URL (e.g., `http://arxiv.org/abs/1201.1567v1`).

This is the "verify" step — the user doesn't just have to trust the AI's answer. They can click the source, read the original text, and check for themselves. This is the whole point of building a RAG system instead of just using a regular chatbot.

### The Chat History Feature
We used Streamlit's `st.session_state` to store the conversation history. This means if you ask multiple questions in one session, all the previous questions and answers stay visible on the screen, just like a real chat.

---

## Part 2: The Bug — `torchvision` Crash

### What happened?
When we first ran `uv run streamlit run app.py`, the app crashed immediately with this error:
```
ModuleNotFoundError: No module named 'torchvision'
```

### Why did it crash?
This is a complex reason. When Streamlit starts, it tries to be very smart. It scans every Python library that our code imports, looking for file changes. This feature is called the **File Watcher**. It is used for live-reload: if you edit `app.py`, Streamlit automatically refreshes the browser without you having to restart the server.

Our code imports the `transformers` library. The `transformers` library has hundreds of sub-modules — not just for text models (which is what we use), but also for image models, video models, audio models, etc. These image model sub-modules (like `detr` and `deepseek_vl`) try to import a library called `torchvision` to process images.

We never installed `torchvision` because we are only building a text-based system. So when Streamlit's file watcher tried to scan the image model sub-modules, they crashed because `torchvision` was missing.

**The key insight:** This was not a bug in our code. Our code was never going to use those image model sub-modules. Streamlit just happened to trigger them while scanning.

### How it was fixed
We created a configuration file at `.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "none"
```

This one line tells Streamlit: "Don't scan any files at all. Skip the file watcher entirely." The live-reload feature is disabled, but that is fine because:
- We don't need live-reload in a deployed/demo app
- A manual server restart (`Ctrl+C` and then `uv run streamlit run app.py`) is easy enough

This file is committed to the git repository so that every person who clones the project gets this fix automatically. They just run `uv run streamlit run app.py` and it works.

---

## Part 3: The CI Pipeline (`.github/workflows/ci.yml`)

### What is CI?
CI stands for "Continuous Integration." It means: every time you push code to GitHub, an automated system runs your tests on GitHub's own servers. 

**Why does this matter?** Without CI, you can only say "it worked on my machine." With CI, GitHub runs a green or red badge that tells everyone else "this code was independently tested and it passed."

### What did we replace?
In Phase 0, we created a placeholder CI file that looked like this:
```yaml
- run: echo "Tests will go here"
```
This was completely fake — it just printed a message. It always passed, even if the code was completely broken.

### What does the new CI do?
The new `.github/workflows/ci.yml` does 4 real things every time you push to the `main` branch:

**Step 1: Check out the repository**
GitHub's servers download your code.

**Step 2: Install `uv`**
GitHub installs the `uv` package manager (using the official `astral-sh/setup-uv` action).

**Step 3: Install all project dependencies**
`uv sync --locked` installs the exact same library versions that are in `uv.lock`. This is important — it guarantees that GitHub is testing with the exact same environment you used locally. There is no risk of "it worked locally but GitHub installed a different version."

**Step 4: Lint with `ruff`**
`uv run ruff check .` scans all Python files for style errors and bad code patterns.

**Step 5: Run all tests**
`uv run pytest tests/ -v` runs all 10 tests (from Phases 1-7).

If any of these steps fail, the push shows a red "X" on GitHub. If they all pass, it shows a green checkmark.

### The CI Limitation
Our NER model weights (`models/matscibert_ner/`) are stored in a local folder. They are too large for git, so they are in `.gitignore`. This means GitHub's servers don't have access to the model.

For this reason, the NER-specific tests (Phase 2 tests that require the model file to exist) pass locally but will fail in CI. We added a note to `README.md` under "CI Limitations" to be transparent about this — not hiding it. This is a real production practice: a green CI badge that secretly means less than it looks is worse than an honest note explaining what is actually verified.

---

## Part 4: The New `retrieve()` Function in `rag_query.py`

### Why was it added?
Our new smoke test (`tests/test_pipeline.py`) needed to verify that the retriever works end-to-end. But the existing `answer_query()` function also loads the Flan-T5 LLM model into memory, which takes about 30 seconds. Running 30 seconds of model loading just to verify the database connection would make the test suite too slow.

### What it does
The new `retrieve(query, k)` function is a clean, fast helper that only does the vector search part:
1. Prefixes the query with the BGE query prefix
2. Calls the ChromaDB retriever
3. Returns results as a list of plain Python dictionaries with `text`, `title`, and `source_id` keys

This way, `test_pipeline.py` can run in about 55 seconds (dominated by loading the embedding model, not the LLM) and verify that the retrieval system is working correctly.

---

## Summary Table

| What | File | Why |
|---|---|---|
| The web UI | `app.py` | Make the system usable by non-programmers |
| Fix the torchvision crash | `.streamlit/config.toml` | Disable file watcher — Streamlit was scanning vision model sub-modules |
| Smoke test | `tests/test_pipeline.py` | Verify retrieval works end-to-end without loading the LLM |
| Real CI pipeline | `.github/workflows/ci.yml` | Replace the Phase 0 echo placeholder with actual lint + test |
| Transparent CI docs | `README.md` | Document that NER model is not tested in CI (too large for git) |
| ADR | `docs/ADRs/ADR-007-streamlit-ui-filewatcher-fix.md` | Record why Streamlit was chosen and how the crash was fixed |

## Files Updated

| File | Change |
|---|---|
| `RESULTS.md` | Added Phase 7 live test result, bug log, and test summary |
| `.gitignore` | Added `crash.log` and `.streamlit/secrets.toml` entries |
| `docs/ADRs/ADR-007...` | Created — documents all three decisions made in Phase 7 |
| `src/rag/rag_query.py` | Added `retrieve()` helper function |
