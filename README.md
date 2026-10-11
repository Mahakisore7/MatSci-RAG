<div align="center">

<img src="docs/assets/banner.svg" alt="MatSci-RAG — NER-aware question answering over Solid Oxide Fuel Cell literature" width="100%">

<br/>

<p>
Ask a question about SOFC materials and get an answer grounded <b>only</b> in 400 arXiv papers —<br/>
with the passages it came from, a self-verification check, and an explicit <i>"I don't know"</i> when the literature is silent.
</p>

<p>
<a href="https://github.com/Mahakisore7/MatSci-RAG/actions/workflows/ci.yml"><img src="https://github.com/Mahakisore7/MatSci-RAG/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<img src="https://img.shields.io/badge/tests-60%20passing-16a34a?style=flat-square&logo=pytest&logoColor=white" alt="60 tests">
<img src="https://img.shields.io/badge/ADRs-19-6366f1?style=flat-square" alt="19 ADRs">
<img src="https://img.shields.io/badge/python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.13">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-475569?style=flat-square" alt="Apache 2.0"></a>
</p>

<p>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/ChromaDB-FF6446?style=for-the-badge" alt="ChromaDB">
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge" alt="Groq">
<img src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="uv">
</p>

<p>
<a href="#overview"><b>Overview</b></a> &nbsp;·&nbsp;
<a href="#architecture"><b>Architecture</b></a> &nbsp;·&nbsp;
<a href="#quickstart"><b>Quickstart</b></a> &nbsp;·&nbsp;
<a href="#rest-api"><b>REST API</b></a> &nbsp;·&nbsp;
<a href="#results"><b>Results</b></a> &nbsp;·&nbsp;
<a href="#findings"><b>Findings</b></a> &nbsp;·&nbsp;
<a href="#roadmap"><b>Roadmap</b></a> &nbsp;·&nbsp;
<a href="#documentation"><b>Docs</b></a>
</p>

</div>

---

<a name="overview"></a>

## <img src="docs/assets/icons/overview.svg" width="26" height="26" align="top" alt=""> Overview

General-purpose RAG treats a materials-science paper like any other text: it embeds chunks and
retrieves by similarity. That breaks on this domain — *"cathode materials"* pulls in battery
papers, and chemical formulas like `La0.6Sr0.4Co0.2Fe0.8O3` don't embed meaningfully.

MatSci-RAG adds a **domain layer**: a MatSciBERT model fine-tuned for materials NER extracts
materials, devices and experiments from every paper, and retrieval uses those entities to
re-rank results — without ever filtering a paper out.

| Capability | How it works |
|---|---|
| **Domain NER** | MatSciBERT fine-tuned on SOFC-Exp; beats a BERT-base baseline (F1 0.0915 vs 0.0816) |
| **Hybrid retrieval** | Dense (BGE-small) + BM25, fused with Reciprocal Rank Fusion, plus an NER-overlap boost |
| **Grounded generation** | `qwen3.8-27b` via Groq, with automatic fallback to local Flan-T5; refuses unsupported questions |
| **Self-verification** | A *different* model (`gpt-oss-20b`) checks the answer against the passages and quotes its evidence |
| **Ambiguity abstention** | Asks the user to narrow the question when results span too many materials |
| **Multi-turn memory** | Follow-ups resolve pronouns from the last 3 turns |
| **Query expansion** | Optional LLM rewrite into a keyword-rich search query (retrieval only) |
| **Full-paper ingestion** | PDF download → section parsing → section-aware chunking |
| **Knowledge graph** | Interactive paper ↔ entity graph |
| **LLM-as-judge evaluation** | Faithfulness / Answer Relevancy / Context Precision, reproducible from one script |
| **REST API** | FastAPI with validated schemas and Swagger docs |

<p align="center">
  <img src="docs/assets/ui-assistant.png" alt="MatSci-RAG assistant" width="92%">
</p>

---

<a name="architecture"></a>

## <img src="docs/assets/icons/architecture.svg" width="26" height="26" align="top" alt=""> Architecture

### System overview

Two halves run at different times: an **offline pipeline** that builds the knowledge base once,
and an **online pipeline** that answers every question. Both share the same NER facts.

```mermaid
flowchart TB
    subgraph SRC["SOURCES"]
        direction LR
        ARX[("arXiv API<br/>400 SOFC papers")]
        EXP[("SOFC-Exp<br/>annotated NER corpus")]
        ARX ~~~ EXP
    end

    subgraph OFF["OFFLINE PIPELINE · run once"]
        direction LR
        ING["Ingestion<br/>abstracts + full-text PDFs"] --> NER["MatSciBERT NER<br/>fine-tune → inference"] --> IDX["Chunk + embed<br/>BGE-small-en-v1.5"]
    end

    subgraph KB["KNOWLEDGE BASE"]
        direction LR
        CH[("ChromaDB<br/>vectors + NER metadata")]
        FX[("Structured facts<br/>materials · devices · experiments")]
        CH ~~~ FX
    end

    subgraph ON["ONLINE PIPELINE · every question"]
        direction LR
        RET["Hybrid retrieval<br/>dense + BM25 + NER boost<br/>+ cross-encoder re-rank"] --> GEN["Grounded generation<br/>Qwen3 · Flan-T5 fallback"] --> VER["Self-verification<br/>+ abstention"]
    end

    subgraph IF["INTERFACES"]
        direction LR
        UI["Streamlit app<br/>chat · analytics · graph"]
        API["REST API<br/>FastAPI · /query"]
        UI ~~~ API
    end

    GROQ{{"Groq API<br/>query expansion · eval judge"}}

    SRC ==> OFF ==> KB ==> ON ==> IF
    GROQ -. optional .-> ON

    classDef src fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef off fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    classDef kb fill:#ecfeff,stroke:#0891b2,color:#083344
    classDef on fill:#fff7ed,stroke:#ea580c,color:#431407
    classDef ifc fill:#f0fdf4,stroke:#16a34a,color:#052e16
    classDef ext fill:#faf5ff,stroke:#9333ea,color:#3b0764
    class ARX,EXP src
    class NER,ING,IDX off
    class CH,FX kb
    class RET,GEN,VER on
    class UI,API ifc
    class GROQ ext
    style SRC fill:transparent,stroke:#94a3b8,stroke-dasharray:4 4
    style OFF fill:transparent,stroke:#6366f1,stroke-dasharray:4 4
    style KB fill:transparent,stroke:#0891b2,stroke-dasharray:4 4
    style ON fill:transparent,stroke:#ea580c,stroke-dasharray:4 4
    style IF fill:transparent,stroke:#16a34a,stroke-dasharray:4 4
```

### Offline: building the knowledge base

```mermaid
flowchart TB
    A[("arXiv search<br/>400 papers")] --> B["fetch_corpus.py<br/>titles + abstracts"]
    A --> P["pdf_downloader.py<br/>399 / 400 PDFs"]
    P --> Q["pdf_parser.py<br/>pypdf → section detection<br/>intro · methods · results · …"]

    S[("SOFC-Exp<br/>BIO-tagged sentences")] --> T["train.py<br/>MatSciBERT vs BERT-base"]
    T -->|"best model"| N["inference.py<br/>run NER on every abstract"]
    B --> N
    N --> F[("structured_facts.jsonl")]

    B --> C1["Abstract chunks<br/>500 chars · 50 overlap"]
    Q --> C2["Section chunks<br/>800 chars · 50 overlap<br/>references dropped"]
    B -. "fallback for papers<br/>without a parsed PDF" .-> C2
    F -. "joined as chunk metadata" .-> C1
    F -.-> C2

    C1 -->|"--source abstract"| E["build_index.py<br/>BGE-small embeddings"]
    C2 -->|"--source full_paper"| E
    E --> DB[("ChromaDB<br/>collection: sofc_papers")]

    classDef store fill:#ecfeff,stroke:#0891b2,color:#083344
    classDef ner fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
    classDef full fill:#fff7ed,stroke:#ea580c,color:#431407
    class A,S,F,DB store
    class T,N ner
    class P,Q,C2 full
```

The index is built from **either** abstracts **or** full-text sections. Rebuilding always
replaces the collection — it never merges the two.

### Online: answering a question

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant I as Streamlit UI / FastAPI
    participant X as Query expansion<br/>(Groq gpt-oss-20b)
    participant R as hybrid_retrieve()
    participant K as ChromaDB + BM25
    participant A as detect_ambiguity()
    participant G as Generator<br/>(qwen3.8-27b)
    participant V as Verifier<br/>(gpt-oss-20b)

    U->>I: question (+ last 3 turns)
    opt expansion enabled
        I->>X: rewrite for search
        X-->>I: keyword-rich query<br/>(raw query on any error)
    end
    I->>R: search query
    R->>K: dense top-20 + BM25 top-20
    K-->>R: candidates
    R->>R: RRF fusion + NER-overlap boost
    R->>R: cross-encoder re-ranks the top 20
    R-->>I: top-5 passages
    I->>A: entities in the top-5
    alt too many distinct materials
        A-->>U: abstain — "please narrow your question"
    else normal case
        I->>G: context-only prompt + ORIGINAL question
        G-->>I: answer
        I->>V: "does the context support this?"
        V-->>I: yes / no + quoted evidence
        I-->>U: answer · verification badge · cited sources
    end
```

Expansion changes only **what is searched for**; the generator always sees the user's original
question, so a bad rewrite can't put words in the answer.

### Zoom-in: why retrieval is NER-*aware*, not NER-*gated*

An early version hard-filtered results by NER tags. The data showed why that fails: **68% of
papers have no extracted `devices` at all**, so a filter silently discarded most of the corpus.
The current design uses entities only as a **soft boost**:

```mermaid
flowchart LR
    Q["Query"] --> D["Dense search<br/>BGE-small · pool = max(4k, 20)"]
    Q --> B["BM25 search<br/>keyword · pool = max(4k, 20)"]
    Q --> E["Entities mentioned<br/>in the query"]
    D --> RRF["Reciprocal Rank Fusion<br/>Σ 1 / (60 + rank)"]
    B --> RRF
    E -->|"+0.02 per matching<br/>paper entity"| RRF
    RRF --> T["Dedupe → top 20"]
    T --> X["Cross-encoder re-rank<br/>MiniLM-L6 · local"]
    X --> K["Top-k passages"]

    style RRF fill:#fff7ed,stroke:#ea580c,color:#431407
    style X fill:#ecfdf5,stroke:#10b981,color:#052e16
    style E fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
```

A paper with no extracted entities can still rank first on text alone; entities can only lift
relevant papers, never remove one. Derivation in [ADR-009](docs/ADRs/ADR-009-ner-aware-retrieval-reranking.md).

A cross-encoder then re-scores the fused top 20 by reading each query–passage pair together, which
the first stage cannot do ([ADR-019](docs/ADRs/ADR-019-cross-encoder-reranking.md)). On 21 questions
with known answering papers it raised hit@5 from 0.714 to 0.952 and MRR from 0.448 to 0.849, for
about +100 ms per query.

### Evaluation harness

```mermaid
flowchart LR
    QS["7 benchmark<br/>SOFC questions"] --> P["answer_query()<br/>config under test"]
    P --> AN["answer + top-5 contexts"]
    AN --> J{{"Judge LLM<br/>Groq gpt-oss-120b"}}
    J --> M1["Faithfulness<br/>answer ⊆ context?"]
    J --> M2["Answer Relevancy<br/>answers the question?"]
    J --> M3["Context Precision<br/>are the passages relevant?"]
    M1 & M2 & M3 --> CSV[("docs/results/v2/*.csv")]

    style J fill:#faf5ff,stroke:#9333ea,color:#3b0764
```

Transient API errors are retried with backoff; any other judge failure **aborts the run**
instead of being recorded as a score (see [Findings](#findings) for why).

---

<a name="quickstart"></a>

## <img src="docs/assets/icons/quickstart.svg" width="26" height="26" align="top" alt=""> Quickstart

**Prerequisites:** Python 3.13, [`uv`](https://docs.astral.sh/uv/), and optionally a free
[Groq API key](https://console.groq.com/) for query expansion and evaluation.

```bash
git clone https://github.com/Mahakisore7/MatSci-RAG.git
cd MatSci-RAG
uv sync
cp .env.example .env          # add GROQ_API_KEY (optional)
```

**Build the knowledge base** (one-time; `data/` and `models/` are gitignored):

```bash
# 1. Corpus: 400 SOFC abstracts from arXiv
uv run python -m src.ingestion.fetch_corpus \
    --query "solid oxide fuel cell SOFC electrolyte anode cathode material" \
    --output data/sofc_corpus.jsonl

# 2. NER training data: Bosch's SOFC-Exp corpus (follow its README to fetch the texts)
git clone https://github.com/boschresearch/sofc-exp_textmining_resources.git data/sofc_exp_repo
uv run python -m src.ner.prepare_data                 # → data/ner_processed

# 3. Fine-tune MatSciBERT + BERT-base baseline, then tag every paper
uv run python -m src.ner.train                        # GPU recommended
uv run python -m src.ner.inference                    # → data/structured_facts.jsonl

# 4. Vector index — abstracts (default) or full text
uv run python -m src.rag.build_index
uv run python scripts/ingest_full_papers.py           # optional: download, parse, re-index
```

**Run it:**

```bash
uv run streamlit run src/ui/app.py                    # chat UI → http://localhost:8501
uv run uvicorn src.api.main:app --port 8000           # REST API → http://localhost:8000/docs
uv run pytest tests/ -v                               # test suite
uv run python scripts/evaluate_ragas.py --expand      # LLM-judge evaluation
```

### Configuration

All settings live in [`src/core/config.py`](src/core/config.py) and can be overridden with
environment variables (or `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `GEN_BACKEND` | `groq` | `groq` = hosted model with local fallback; `local` = fully offline |
| `GROQ_GEN_MODEL` | `qwen/qwen3.8-27b` | Answer generator |
| `GROQ_VERIFY_MODEL` | `openai/gpt-oss-20b` | Self-verification (deliberately a different model) |
| `GEN_MODEL_NAME` | `google/flan-t5-base` | Local generator, and the fallback when Groq is unavailable |
| `USE_RERANK` | `1` | Cross-encoder re-ranking of retrieved passages; `0` disables it |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder used for re-ranking |
| `GROQ_API_KEY` | — | Enables query expansion and evaluation |
| `GROQ_EXPANSION_MODEL` | `openai/gpt-oss-20b` | Query rewriting (fast; on the live path) |
| `GROQ_JUDGE_MODEL` | `openai/gpt-oss-120b` | Evaluation judge |
| `HF_TOKEN` | — | Hugging Face downloads (optional) |

Generator, verifier and judge are three different models, so no model grades its own output.
Groq has retired several models this project used, so model names are configuration, not code.

---

<a name="rest-api"></a>

## <img src="docs/assets/icons/api.svg" width="26" height="26" align="top" alt=""> REST API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes anode degradation in Ni-YSZ SOFCs?", "expand_query": true}'
```

```jsonc
{
  "answer": "Based on the provided context, anode degradation in Ni-YSZ SOFCs is primarily caused by **Ni coarsening**. This is a capillarity-driven phenomenon where materials are transported from regions of high curvature to regions of lower curvature (in accordance with the Gibbs-Thomson effect). This process increases the resistance in SOFCs and results in cell degradation. Additionally, high operating temperatures contribute to rapid component degradation.",
  "abstained": false,
  "verification": "Yes.  \n“Ni coarsening in the SOFC anode is a capillarity‑driven phenomenon. Regions with high curvatures have higher chemical potentials than those with lower curvatures in accordance with the Gibbs‑Thomson effect. Materials will therefore be transported from higher‑to‑lower‑curvature regions when mobility is sufficiently large for the time scale of interest.” (Context [4])",
  "sources": [
    // 5 passages; the one the verifier cites is shown
    {
      "rank": 4,
      "title": "Ni coarsening in the three-phase solid oxide fuel cell anode - a phase-field simulation study",
      "paper_id": "http://arxiv.org/abs/1201.1567v1",
      "section": "introduction",
      "text": "major mechanisms responsible for microstructural change i n SOFC anodes [10]. …"
    }
  ]
}
```

This is real output from `qwen3.8-27b`: the answer is grounded in the retrieved passages, and
the verifier (a different model) quotes the exact supporting sentence.

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | string | — | 1–1000 characters |
| `use_hybrid` | bool | `true` | `false` = dense-only retrieval |
| `expand_query` | bool | `false` | LLM rewrite before retrieval |
| `verify` | bool | `true` | Run the self-verification check |
| `chat_history` | list | `[]` | `{role: "user" \| "assistant", content}`, max 20 |

`GET /health` returns the active generator. Interactive docs are at `/docs`. Design notes in
[ADR-014](docs/ADRs/ADR-014-rest-api.md).

---

<a name="results"></a>

## <img src="docs/assets/icons/results.svg" width="26" height="26" align="top" alt=""> Results

### NER: domain pretraining helps

| Model | Test F1 |
|---|---|
| BERT-base | 0.0816 |
| **MatSciBERT** | **0.0915** |

Absolute F1 is low because the training set is small (SOFC-Exp) and most tokens are non-entities;
the comparison between the two models is the point. Details in [RESULTS.md](RESULTS.md).

### Retrieval

| Test | Result |
|---|---|
| Dense retrieval hit-rate, 20 specific questions | **100% (20/20)** correct paper in top-5 |
| *"Which cathode materials show high ORR activity?"* | Dense-only returns a battery paper; hybrid returns the SOFC paper ([screenshots](#screenshots)) |

### End-to-end quality (LLM judge, 7 questions)

Ten configurations, the same 7 benchmark questions, judged by `openai/gpt-oss-120b`
([raw CSVs](docs/results/v2/); analysis in [ADR-015](docs/ADRs/ADR-015-generator-size-and-full-corpus-evaluation.md)
and [ADR-016](docs/ADRs/ADR-016-groq-generator-with-local-fallback.md)):

| Index | Generator | Expansion | Faithfulness | Answer Relevancy | Context Precision |
|---|---|:---:|---:|---:|---:|
| Abstracts | flan-t5-base | | 0.857 | 0.000 | 0.314 |
| Abstracts | flan-t5-base | on | 0.857 | **0.143** | 0.486 |
| Abstracts | flan-t5-large | | 0.857 | 0.000 | 0.343 |
| Abstracts | flan-t5-large | on | 0.857 | 0.071 | 0.457 |
| Full text | flan-t5-base | | 0.857 | 0.071 | **0.586** |
| Full text | flan-t5-base | on | 0.857 | 0.000 | 0.571 |
| Full text | flan-t5-large | | 0.714 | 0.000 | **0.586** |
| Full text | flan-t5-large | on | 0.857 | 0.000 | 0.543 |
| Full text | qwen3.8-27b | | **1.000** | 0.429 | 0.543 |
| **Full text** | **qwen3.8-27b** | **on** | 0.857 | **0.714** | 0.543 |

**Bold row = the configuration the app serves by default.** Context Precision depends only on
retrieval, so the spread between generators on the same index (0.314 vs 0.343) shows the judge's
noise: about ±0.03.

<p align="center">
  <img src="docs/assets/ui-analytics-eval.png" alt="Evaluation results in the Analytics page" width="88%">
</p>

---

<a name="findings"></a>

## <img src="docs/assets/icons/findings.svg" width="26" height="26" align="top" alt=""> Findings & Limitations

These are measured, not assumed — each has evidence in [RESULTS.md](RESULTS.md).

**1 · Generation was the bottleneck — and a stronger model fixed it.** Full text nearly doubled
Context Precision (0.314 → 0.586), yet with Flan-T5 Answer Relevancy stayed near zero in every
configuration; `flan-t5-large` only became more cautious. Swapping in `qwen3.8-27b` on the same
retrieval raised Answer Relevancy **0.071 → 0.429**, and to **0.714** with query expansion,
with Faithfulness unchanged or better. Retrieval had been finding the right passages all along.

<table>
<tr>
<td width="50%" valign="top"><b>Before</b> — flan-t5-base: the answer is in passage 2, the model answers from passage 1, and the same-model verifier approves it.<br/><br/><img src="docs/assets/ui-answer-flan-t5.png" alt="Flan-T5 answer taken from the wrong passage"></td>
<td width="50%" valign="top"><b>After</b> — qwen3.8-27b: a correct, grounded answer; a different verifier model quotes the supporting sentence.<br/><br/><img src="docs/assets/ui-answer.png" alt="Qwen answer naming Ni coarsening with quoted evidence"></td>
</tr>
</table>

**2 · A verifier is only as good as its model.** With Flan-T5 checking its own output, the
self-check approved a wrong answer. It now runs on a separate model (`gpt-oss-20b`) and must quote
its evidence, so a "supported" badge can be checked by a human in seconds.

**3 · Query expansion's value depends on the generator.** It raises Context Precision on abstracts
(0.314 → 0.486) but not on full text. With Qwen it still lifts Answer Relevancy (0.429 → 0.714),
because the rewritten query pulls in passages a capable model can synthesize — so it is now on by
default in the app.

**4 · Silent failures were the biggest engineering risk.** Each of these looked healthy from
the outside: PDF downloads failed for 100% of papers (wrong `arxiv` API); the retired judge
model's 404s were recorded as 0.0 scores; query expansion silently fell back to the raw query
for the same reason; and the chat UI could not import its own package after a restructure.
All are fixed, and the evaluator now **aborts** rather than invent a score.

**5 · The corpus is noisy, but noise isn't what limits it.** Only 145 of the 400 "SOFC" papers
mention a fuel-cell technology. Indexing just those 145 (3× fewer vectors) gave no measurable gain
and was slightly worse with expansion ([ADR-017](docs/ADRs/ADR-017-fuel-cell-corpus-filter.md)):
the questions that fail, fail on both indexes. The filter remains available as
`build_index --fuel-cell-only`.

**6 · NER is useful but imprecise.** Test F1 is 0.09, and most papers have no extracted device
entity (68% raw, 78% after noise filtering) — which is exactly why entities re-rank results
instead of filtering them (ADR-009).

**7 · The benchmark is small.** Seven questions detect large effects (like finding 1), not
differences of a few hundredths. Three of the seven still get *"I don't know"*. A 28-question follow-up
([ADR-018](docs/ADRs/ADR-018-larger-benchmark-and-retrieval-diagnosis.md)) found 11 abstentions
and, in 4 of 5 cases checked, an on-topic paper already in the corpus that retrieval ranked
outside the top 5. The limit is ranking, not coverage, and cross-encoder
re-ranking ([ADR-019](docs/ADRs/ADR-019-cross-encoder-reranking.md)) now closes most of that gap.

**8 · The answer path now depends on an API.** If Groq is down or rate-limited, answers fall
back to local Flan-T5 automatically, so the app keeps working, but with the weaker model.

---

<a name="roadmap"></a>

## <img src="docs/assets/icons/roadmap.svg" width="26" height="26" align="top" alt=""> Roadmap

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "14px", "cScale0": "#e0e7ff", "cScaleLabel0": "#1e1b4b", "cScale1": "#cffafe", "cScaleLabel1": "#083344", "cScale2": "#ffedd5", "cScaleLabel2": "#431407", "cScale3": "#dcfce7", "cScaleLabel3": "#052e16", "cScale4": "#f3e8ff", "cScaleLabel4": "#3b0764"}}}%%
timeline
    title MatSci-RAG build history
    Phases 0–3 : Tooling, CI, logging
               : arXiv corpus (400 papers)
               : MatSciBERT NER fine-tune + inference
    Phases 4–6 : ChromaDB index (BGE-small)
               : Grounded Flan-T5 generation
               : Retrieval evaluation (20/20)
    Phases 7–8 : Streamlit UI + GitHub Actions
               : Hybrid BM25 + NER-aware RRF
               : Self-verification + abstention
    Phase 9    : Multi-turn memory · query expansion
               : Full-paper ingestion · knowledge graph
               : LLM-judge evaluation · analytics
    Phase 10   : REST API · redesigned UI · configurable models
               : 399/400 full papers · 10-config re-evaluation
               : Qwen3 generator · separate verifier
               : corpus filter measured (not adopted)
               : 28-question benchmark · retrieval diagnosis
               : cross-encoder re-ranking
```

**Next steps**, in order of expected impact:

1. **Judged end-to-end run with re-ranking.** Re-ranking is built and lifts retrieval hit@5 from
   0.714 to 0.952 (ADR-019), but its effect on Answer Relevancy has not been judged yet: a
   28-question run needs most of Groq's 200,000-token daily judge budget. Run it with `--resume`.
   Also try a larger candidate pool and a larger cross-encoder.
2. **Held-out benchmark** — the 28-question set (`--benchmark v2`) was written by the system's
   author; questions from someone else would be a stronger test.
3. **Better PDF extraction** (layout-aware parsing) — `pypdf` text carries broken words and
   equation fragments into full-paper chunks.
4. **A local strong generator** (e.g. a 7–8B model) as an offline-capable alternative to Groq.

---

<a name="screenshots"></a>

## <img src="docs/assets/icons/screenshots.svg" width="26" height="26" align="top" alt=""> Screenshots

**Assistant** — example questions to start, retrieval and verification controls in the sidebar:

![MatSci-RAG assistant page](docs/assets/ui-assistant.png)

**Analytics** — corpus coverage, extracted entities and every evaluation run:

![Analytics page](docs/assets/ui-analytics.png)

**Knowledge graph** — papers linked to the materials, devices and experiments extracted from them:

![Knowledge graph page](docs/assets/ui-knowledge-graph.png)

<details>
<summary><b>Dense vs hybrid retrieval on the same question</b> (captured with the Phase 8 UI)</summary>
<br/>

*Dense only* pulls an off-domain battery paper and answers FePO₄:

![Dense-only retrieval answering FePO4](docs/assets/screenshot-dense-only-cathode-query.png)

*Hybrid (dense + BM25 + NER)* surfaces the SOFC paper instead:

![Hybrid retrieval answering Perovskite](docs/assets/screenshot-hybrid-cathode-query.png)

</details>

---

<a name="tech-stack"></a>

## <img src="docs/assets/icons/stack.svg" width="26" height="26" align="top" alt=""> Tech Stack

| Layer | Stack |
|---|---|
| **NER** | <img src="https://img.shields.io/badge/MatSciBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="MatSciBERT"> <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers"> <img src="https://img.shields.io/badge/seqeval-475569?style=flat-square" alt="seqeval"> |
| **Retrieval** | <img src="https://img.shields.io/badge/BGE--small--en-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="BGE--small--en"> <img src="https://img.shields.io/badge/ChromaDB-FF6446?style=flat-square" alt="ChromaDB"> <img src="https://img.shields.io/badge/BM25-475569?style=flat-square" alt="BM25"> <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"> |
| **Generation** | <img src="https://img.shields.io/badge/Flan--T5-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Flan--T5"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"> |
| **LLM services** | <img src="https://img.shields.io/badge/Groq%20%C2%B7%20gpt--oss--20b-F55036?style=flat-square" alt="Groq%20%C2%B7%20gpt--oss--20b"> <img src="https://img.shields.io/badge/Groq%20%C2%B7%20qwen3.8--27b-F55036?style=flat-square" alt="Groq qwen3.8-27b"> <img src="https://img.shields.io/badge/Groq%20%C2%B7%20gpt--oss--120b-F55036?style=flat-square" alt="Groq%20%C2%B7%20gpt--oss--120b"> |
| **Ingestion** | <img src="https://img.shields.io/badge/arXiv%20API-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv API"> <img src="https://img.shields.io/badge/pypdf-475569?style=flat-square" alt="pypdf"> <img src="https://img.shields.io/badge/Requests-475569?style=flat-square" alt="Requests"> |
| **Interfaces** | <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"> <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic"> <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"> <img src="https://img.shields.io/badge/pyvis%20%C2%B7%20networkx-475569?style=flat-square" alt="pyvis%20%C2%B7%20networkx"> |
| **Engineering** | <img src="https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=uv&logoColor=white" alt="uv"> <img src="https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black" alt="Ruff"> <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest"> <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions"> <img src="https://img.shields.io/badge/pre--commit-FAB040?style=flat-square" alt="pre--commit"> |

---

<a name="project-structure"></a>

## <img src="docs/assets/icons/structure.svg" width="26" height="26" align="top" alt=""> Project Structure

```
MatSci-RAG/
├── src/
│   ├── core/                 config.py (all tunable settings) · logging_config.py
│   ├── ingestion/            fetch_corpus.py · pdf_downloader.py · pdf_parser.py
│   ├── ner/                  prepare_data.py · train.py · inference.py
│   ├── rag/
│   │   ├── build_index.py        abstract or full-paper index (with abstract fallback)
│   │   ├── hybrid_retrieval.py   dense + BM25 candidate pools
│   │   ├── rerank.py             cross-encoder re-ranking · per-paper cap
│   │   ├── ner_utils.py          entity cleaning · RRF + NER boost · ambiguity check
│   │   ├── query_expansion.py    Groq query rewriting
│   │   └── rag_query.py          answer_query() · verify_answer()
│   ├── api/main.py           FastAPI: POST /query, GET /health
│   └── ui/                   app.py (router) · theme.py · views/{chat, analytics, knowledge_graph}.py
├── scripts/                  evaluate_ragas.py · evaluate_rag.py · ingest_full_papers.py
├── tests/                    60 tests — 51 run everywhere (incl. CI); 9 that need local
│                             data/models skip cleanly on a fresh checkout
├── docs/
│   ├── ADRs/                 19 Architecture Decision Records
│   ├── Phases/               step-by-step build guide, Phases 0–8
│   ├── results/              evaluation CSVs (v2/ = current judge) and logs
│   ├── Complete-Guide.md     the whole project in one document, all diagrams
│   ├── PPT-Source.md         slide-by-slide source for a presentation
│   ├── MatSci-RAG-Review.md  formal project review
│   └── Project-Explainer.md  glossary, full flow, mentor Q&A
├── data/  models/            gitignored — generated by the offline pipeline
└── RESULTS.md                all benchmark evidence
```

---

<a name="documentation"></a>

## <img src="docs/assets/icons/docs.svg" width="26" height="26" align="top" alt=""> Documentation

| Document | What's in it |
|---|---|
| [RESULTS.md](RESULTS.md) | Every measurement: NER F1, retrieval hit-rate, eval tables, bugs found |
| [docs/MatSci-RAG-Review.md](docs/MatSci-RAG-Review.md) | Problem statement, literature review, methodology, results |
| [docs/Complete-Guide.md](docs/Complete-Guide.md) | The whole project in one document: every architecture diagram, all results, the investigation story, limits and what is pending |
| [docs/PPT-Source.md](docs/PPT-Source.md) | Slide-by-slide source for a presentation: on-slide text, visuals, speaker notes, backup slides |
| [docs/Project-Explainer.md](docs/Project-Explainer.md) | Glossary of every term, a fully traced example, mentor Q&A |
| [docs/ADRs/](docs/ADRs/) | 19 decision records — why each technical choice was made |
| [docs/sample_questions.md](docs/sample_questions.md) | Ready-to-paste demo questions |
| [docs/archive/](docs/archive/) | Base paper (Maharana et al., 2025) and the Zeroth Review |

---

<div align="center">
<sub>Licensed under [Apache 2.0](LICENSE) · Built with `uv`, tested with `pytest`, checked by GitHub Actions</sub>
</div>
