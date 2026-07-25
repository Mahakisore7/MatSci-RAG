# Phase 0 — Project Foundation & Environment Setup

> **Where this file lives:** `docs/build-guide/00_environment_setup.md`
> **Diagrams referenced below live at:** `docs/build-guide/diagrams/` (lowercase — this exact spelling matters for the image links to render)

---

## 1. Overview — What This Phase Is and Why It Exists

Before any data, any model, any line of ML code — every real production project starts by answering one question: **"If someone else (or future-you) cloned this repo tomorrow, could they get an identical, working environment in under 5 minutes?"** If the answer is no, everything built afterward inherits that fragility. Phase 0 exists purely to make the answer **yes**.

By the end of this phase you will have:

- A dependency system (`uv`) where the *exact* library versions are locked and reproducible — not "whatever pip decided to install today"
- A linter (`ruff`) that enforces one consistent code style automatically
- A pre-commit hook that runs that linter *before* code is even allowed into git history
- A logging system used by every script from Phase 1 onward — no `print()` debugging in this project
- A `.env`-based config pattern — so no secret or API key is ever hardcoded or accidentally committed
- A CI pipeline placeholder (finished properly in Phase 7) that will eventually run automatically on every push
- A **git repository with a real commit history**, tagged at meaningful checkpoints

This is not ceremony. Every one of these exists because it prevents a specific, common failure mode you will otherwise hit later (inconsistent environments, silently broken formatting, leaked secrets, unreadable debug output, "it works on my machine").

### Reconciling with what you've already done

Your screenshot shows you've already run `uv init` and have: `main.py`, `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `.venv/`, plus you manually created `.env`, `.env.example`, `.gitignore`, and a `docs/` folder. **That's a good start — it's the skeleton `uv init` gives you, plus some manual additions.** This phase file gives you the *delta*: everything still missing to call Phase 0 actually complete.

---

## 2. The Workflow This Phase Sets Up

![Tool workflow](diagrams/tool_workflow.png)

Three loops running at three different speeds — and this phase is what wires all three up:
- **Inner loop** (every few minutes): you edit code, run it with `uv run`, repeat.
- **Commit-time loop** (automatic, every `git commit`): pre-commit runs `ruff` and blocks the commit if there's an issue.
- **Push-time loop** (automatic, every `git push`): GitHub Actions CI runs the full check remotely, independent of your machine.

You set all three up once, here, and never think about them again for the rest of the project.

---

## 3. File Structure — Before and After This Phase

### What you have right now (from `uv init` + your manual additions):

```
MatSci-RAG/
├── .venv/
├── docs/
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

### What Phase 0 adds (everything below is new):

```
MatSci-RAG/
├── .venv/                          (already exists)
├── docs/
│   └── build-guide/                <- this guide lives here
│       ├── 00_environment_setup.md
│       └── diagrams/
├── .github/
│   └── workflows/
│       └── ci.yml                  <- NEW: CI placeholder
├── src/                             <- NEW: all your source code goes here
│   ├── config.py                   <- NEW
│   └── logging_config.py           <- NEW
├── tests/                          <- NEW: empty for now, used from Phase 1
├── data/                           <- NEW: gitignored, populated from Phase 1
├── models/                         <- NEW: gitignored, populated from Phase 2
├── .env                            (already exists)
├── .env.example                    (already exists)
├── .gitignore                      (already exists — we'll add lines to it)
├── .pre-commit-config.yaml         <- NEW
├── .python-version                 (already exists)
├── LICENSE                         (already exists)
├── main.py                         (already exists)
├── pyproject.toml                  (already exists — we'll add config blocks to it)
├── README.md                       (already exists)
└── uv.lock                         (already exists — will update as we add packages)
```

**Why `src/` and not just loose files at the root:** this is the standard Python project layout used industry-wide (called a "src layout"). It cleanly separates "code that ships/matters" from config files and docs sitting at the root, and prevents a whole category of import bugs where Python accidentally imports the wrong thing from your project root instead of an installed package.

---

## 4. Git Versioning for This Phase

### Branch naming convention (used for every phase, not just this one)

```bash
git checkout -b phase-0-setup
```

One short-lived branch per phase. Merged back to `main` only when that phase's checkpoint (Section 7 below) passes. Never commit half-working code directly to `main`.

### Commit message convention: Conventional Commits

```
feat: ...      new functionality
fix: ...       bug fix
docs: ...      documentation only
test: ...      adding/updating tests
chore: ...     tooling, config, dependencies — no source logic change
refactor: ...  code restructuring, no behavior change
```

Phase 0's commit will be a `chore:` commit — we're building tooling, not features yet.

### Tagging convention

```bash
git tag v0.1.0    # after Phase 0
git tag v0.2.0    # after Phase 1
...
git tag v1.0.0    # after everything, end of Phase 8
```

Each tag is a restorable checkpoint. If something breaks badly three phases from now, you (or your mentor) can check out an earlier tag and confirm exactly when things were last known-good.

---

## 5. Step-by-Step Implementation

### 5.1 Confirm your current state first (don't skip this)

```bash
cd path/to/MatSci-RAG
cat pyproject.toml
uv tree
git status
git branch
```

If `git status` says "not a git repository," run `git init` and `git branch -M main` before continuing.

### 5.2 Start the feature branch

```bash
git checkout -b phase-0-setup
```

### 5.3 Check your GPU driver

```bash
nvidia-smi
```
Note the **CUDA Version** shown top-right — needed in 5.5.

### 5.4 Create the missing folders

```bash
mkdir -p src tests data models .github/workflows docs/build-guide/diagrams
```

### 5.5 Add the CUDA index config to `pyproject.toml`

Open the file `uv init` already created — **append** this, don't replace anything already there:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cuda" },
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```
*(swap `cu121` for whatever `nvidia-smi` reported)*

### 5.6 Add the `ruff` config to the same `pyproject.toml`

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]   # pycodestyle, pyflakes, isort, pyupgrade, bugbear
```

### 5.7 Install all project dependencies

```bash
uv add torch
uv add transformers datasets seqeval accelerate
uv add sentence-transformers faiss-cpu
uv add arxiv streamlit python-dotenv
uv add scikit-learn pandas matplotlib

uv add --dev ruff pytest pre-commit
```

Each command updates `pyproject.toml` and `uv.lock` together — this pairing is what makes the environment reproducible; never hand-edit `uv.lock`.

### 5.8 Create `src/logging_config.py`

*(Full code — this one's simple enough to give you complete, no gaps.)*

```python
# src/logging_config.py
"""Centralized logging setup. Every script in this project imports get_logger
from here instead of using print() -- gives consistent, timestamped, leveled
output across the whole pipeline."""

import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### 5.9 Create `src/config.py`

*(Full code — wires up the `.env` files you already created.)*

```python
# src/config.py
"""Central place for config/secrets. Nothing else in the project should call
os.getenv() directly -- import from here instead, so there's exactly one
place that knows where config comes from."""

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")  # None if not set -- fine, all models used are public
```

Update your existing `.env.example` to document this:
```
# HF_TOKEN=            # only needed if you later use a gated/private Hugging Face model
```

### 5.10 Update `.gitignore`

Add these lines if not already present:
```
.venv/
__pycache__/
*.pyc
.env
data/*.jsonl
data/matscholar_processed/
models/
*.index
.pytest_cache/
.ruff_cache/
```

### 5.11 Create `.pre-commit-config.yaml`

*(Full code.)*

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
uv run pre-commit install
```

### 5.12 Create the CI placeholder: `.github/workflows/ci.yml`

*(Full code — this gets finished properly in Phase 7; for now it just confirms the pipeline triggers.)*

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
      - run: uv sync --locked --all-extras --dev
      - run: echo "Tests will go here in Phase 7"
```

### 5.13 Verify GPU access end-to-end

*(Full code.)*

```python
# verify_gpu.py
import torch
from src.logging_config import get_logger

log = get_logger(__name__)
log.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log.info(f"Device: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM (GB): {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}")
```

```bash
uv run verify_gpu.py
```

---

## 6. Git: Commit and Merge

![Git branching model](diagrams/git_branching.png)

```bash
git add -A
git commit -m "chore: complete Phase 0 - folders, ruff, pre-commit, logging, config, CI placeholder"

git checkout main
git merge phase-0-setup
git tag v0.1.0
```

---

## 7. ✅ Phase 0 Checkpoint — Do Not Proceed to Phase 1 Until All Checked

- [ ] `src/`, `tests/`, `data/`, `models/`, `.github/workflows/` all exist
- [ ] `uv run verify_gpu.py` logs `CUDA available: True` and your GPU name
- [ ] `uv run ruff check .` runs with no errors
- [ ] `uv run pre-commit run --all-files` passes cleanly
- [ ] `git log` shows your Phase 0 commit on `main`, tagged `v0.1.0`

---

## 8. Disk Budget Reference (for your 40 GB free)

| Item | Approx. size |
|---|---|
| PyTorch + CUDA libraries | ~3 GB |
| MatSciBERT + generic BERT (cached, Phase 2) | ~900 MB |
| MiniLM embedding model (Phase 4) | ~90 MB |
| Flan-T5-base (Phase 5) | ~1 GB |
| Corpus + Matscholar dataset | a few MB |
| **Total** | **~6-9 GB — comfortable** |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Double-check the CUDA index URL in 5.5 matches `nvidia-smi` exactly |
| `uv add torch` pulls a CPU-only wheel | Confirm the `[tool.uv.sources]` block was added *before* running `uv add torch` |
| `pre-commit` blocks every commit | That's correct behavior — run `uv run ruff check . --fix`, then commit again |
| Diagrams not rendering in preview | Folder must be named exactly `diagrams` (lowercase) — case-sensitive path match required |

---

## 10. If You Want to Hand Any of This to an IDE Agent (Antigravity / Claude / Gemini)

Everything above is complete enough to implement directly — there's no "gist only" section in this phase, since environment setup has no ambiguity to leave for a model to fill in. From Phase 2 onward, you'll hit at least one step (parsing the Matscholar dataset's exact file format) where I'll give you the structure and logic, and explicitly flag where you'll need an agent (or your own inspection of the raw files) to fill in a format-specific detail — that's a "give the gist" moment done for a real reason, not a shortcut.

---

**Next file:** `01_data_collection.md` — request it once every box in Section 7 is checked.
