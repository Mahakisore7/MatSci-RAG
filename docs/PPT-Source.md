# MatSci-RAG — Slide Source for the Presentation

*Built from [`Complete-Guide.md`](Complete-Guide.md). Every number here is copied from that file or
from `docs/results/`. Paste a slide's **On slide** text into your deck, add its **Visual**, and use
the **Speaker notes** as your script.*

**Target length:** about 20 minutes, 22 main slides + 4 backup slides.
**Audience assumed:** a mentor / faculty panel that knows ML basics but not this project.

---

## How to use this file

1. **Diagrams.** The Mermaid code lives in `Complete-Guide.md`, section numbers are given per slide
   (e.g. "Guide §4.1"). To turn one into an image: paste the code into <https://mermaid.live> and
   export PNG/SVG, or run `npx -p @mermaid-js/mermaid-cli mmdc -i diagram.mmd -o diagram.png -b transparent -s 3`.
   Alternatively open the README on GitHub, where every diagram already renders, and screenshot it.
2. **Screenshots** already exist in `docs/assets/`; the file names are given per slide.
3. **Charts.** Data for each chart is in a small table in the slide; build them as simple bar charts.
4. **Style.** One idea per slide, ≤ 5 bullets, ≥ 24 pt text, diagrams large. Suggested palette
   (matches the app): indigo `#4f46e5` primary, navy `#0f172a` background for title/section slides,
   orange `#ea580c` for the online pipeline, cyan `#0891b2` for the knowledge base, green `#16a34a`
   for "good", red `#dc2626` only for failures. Font: Inter.
5. **Honesty rule.** Slides 17, 20 and 21 show what failed and what is not done. Keep them: they are
   the strongest part of the story.

---

## Slide 1 — Title

**On slide**
- **MatSci-RAG**
- NER-aware question answering over Solid Oxide Fuel Cell literature
- *Your name · course (DDMMS, Sem 5) · date*

**Visual:** `docs/assets/banner.svg` full-bleed, or the logo `docs/assets/logo.svg`.

**Speaker notes:** "This project answers questions about fuel-cell materials using only 400 research
papers as its source, shows where each answer came from, and says 'I don't know' when the papers are
silent. I'll show how it works, and just as importantly what I measured, what failed, and what's left."

---

## Slide 2 — The problem

**On slide**
- Chatbots **hallucinate** technical facts and can't cite sources
- Generic RAG still fails in this field:
  - *"cathode materials"* returns **battery** papers
  - Formulas like `La0.6Sr0.4Co0.2Fe0.8O3` don't embed meaningfully
- Researchers need answers they can **trace and trust**

**Visual:** two-column "ask → get" mock: query "Which cathode materials show high oxygen reduction
activity?" → dense-only answer **FePO₄** (a battery material). Use
`docs/assets/screenshot-dense-only-cathode-query.png`.

**Speaker notes:** "This is a real failure from my own system before the fix. Plain semantic search
saw the word 'cathode' and returned lithium-battery papers, so the answer was FePO₄, a battery
material, fluently stated and wrong for fuel cells."

---

## Slide 3 — The idea

**On slide**
- Add a **materials-science layer** to RAG:
  1. **NER** (MatSciBERT) tags materials, devices, experiments in every paper
  2. **Hybrid retrieval** — meaning + keyword, boosted by those tags
  3. **Cross-encoder** re-reads the top candidates
  4. **Strong LLM answers only from the passages**; a *different* LLM checks it
- Refuses when the literature is silent

**Visual:** four numbered boxes in a row (tag → search → re-rank → answer + check), using your slide tool's built-in icons.

**Speaker notes:** "Four layers, each addressing a specific failure. The NER knows the domain, the
hybrid retrieval stops semantic search drifting to the wrong field, the cross-encoder fixes ranking,
and a separate model checks the answer so no model grades its own work."

---

## Slide 4 — Background in 30 seconds

**On slide**
- **RAG** = retrieve text, then answer *from that text*
- **Dense** search: similar meaning · **BM25**: exact words · **RRF** merges both
- **NER**: label spans as MATERIAL / DEVICE / EXPERIMENT
- **Cross-encoder**: scores query + passage together (accurate, slower)
- **LLM-as-judge**: a model scores answers on a rubric

**Visual:** simple glossary strip; no diagram.

**Speaker notes:** "Only five ideas are needed. Dense search finds similar meaning but misses exact
formulas; BM25 is the reverse; fusing them covers both. A cross-encoder is a more careful second
reader used only on the best 20 candidates."

---

## Slide 5 — Data and models

**On slide**
- **400** arXiv SOFC papers · full text for **399** · **22,659** searchable chunks
- NER training: **SOFC-Exp**, 876 human-labelled sentences
- Local: MatSciBERT, BGE-small, cross-encoder MiniLM, Flan-T5 (fallback)
- Groq API: **qwen3.8-27b** (answers) · gpt-oss-20b (checks) · gpt-oss-120b (evaluation)

**Visual:** two-column table "Data" / "Models" (Guide §3).

**Speaker notes:** "Three different LLMs for three jobs on purpose: the one that answers is never the
one that checks it or the one that grades it. Everything is configurable by environment variable
because the API provider retired models three times during this project."

---

## Slide 6 — System architecture

**On slide**
- Offline: build the knowledge base once
- Online: answer every question
- Shared **NER facts** link the two

**Visual:** **Guide §4.1 — System overview** (full slide, largest diagram).

**Speaker notes:** "Left to right: sources, an offline pipeline that runs once, the knowledge base,
the online pipeline that runs per question, and two interfaces, a Streamlit app and a REST API. The
dotted line is Groq: optional for query rewriting, used for evaluation."

---

## Slide 7 — Offline pipeline

**On slide**
- PDFs → section-aware parsing → **800-char chunks**
- Abstracts → **500-char chunks**
- NER facts attached to every chunk as metadata
- BGE embeddings → **ChromaDB**
- Papers with no PDF fall back to their abstract (none silently dropped)

**Visual:** **Guide §4.2 — Offline pipeline.**

**Speaker notes:** "Two text sources, one index. The fallback matters: without it, full-text
mode would drop any paper whose PDF failed to download, and it would vanish from the index without any error."

---

## Slide 8 — NER: does domain pretraining help?

**On slide**
- Controlled experiment: same data, same hyper-parameters, only pretraining differs

| Model | Test F1 |
|---|---:|
| BERT-base | 0.0816 |
| **MatSciBERT** | **0.0915** |

- Absolute F1 is low (876 sentences, mostly non-entity tokens) — **reported openly**
- Consequence: NER is used as a **soft boost**, never a filter

**Visual:** two-bar chart (0.0816 vs 0.0915), y-axis from 0.

**Speaker notes:** "MatSciBERT, pretrained on 2.4 million materials papers, beats generic BERT on
identical training, which replicates a published finding on my own data. The score is low in absolute
terms and I say so. That low precision directly shaped the design on slide 10."

---

## Slide 9 — Online flow: one question, step by step

**On slide**
- Optional rewrite → hybrid search → re-rank → ambiguity check → answer → verify
- Generator sees the **original** question (a bad rewrite can't change the answer)
- Output: answer + verification badge + cited sources

**Visual:** **Guide §4.3 — Sequence diagram.**

**Speaker notes:** "Walk the numbered arrows. The key safety property: query expansion only changes
what is searched, not what the model is asked. And the system can stop early and ask the user to narrow
an ambiguous question, though honestly that rarely fires because NER is noisy."

---

## Slide 10 — Retrieval: why a boost, not a filter

**On slide**
- Dense + BM25 → **RRF** → +0.02 per matching NER entity → **cross-encoder** → top 5
- First idea: filter by NER tags. Data said no: **68% of papers have no extracted device**
- A filter would silently delete most of the corpus
- So entities only **lift** relevant papers (ADR-009)

**Visual:** **Guide §4.4 — Retrieval zoom-in.**

**Speaker notes:** "This is a decision driven by data. I inspected NER coverage and found a hard
filter would hide two-thirds of the papers. The soft boost keeps recall and still pulls fuel-cell
papers up."

---

## Slide 11 — Demo: same question, two retrievals

**On slide**
- *"Which cathode materials show high oxygen reduction activity?"*
- Dense only → **FePO₄** (battery, wrong domain)
- Hybrid → **Perovskite** (correct SOFC cathode class)

**Visual:** side by side `docs/assets/screenshot-dense-only-cathode-query.png` and
`docs/assets/screenshot-hybrid-cathode-query.png`.

**Speaker notes:** "Same question, only the retrieval setting changed. Note the verifier said 'yes'
to the wrong answer in the dense case: verification checks consistency with the retrieved text, not
truth, which is why retrieval quality matters so much."

*(Live demo option: run the app and ask: "What causes anode degradation in Ni-YSZ SOFCs?" → Ni
coarsening, Gibbs–Thomson, verifier quote. Guardrail: "Who won the Super Bowl last year?" → refused.)*

---

## Slide 12 — Generation: fallback and a separate verifier

**On slide**
- Answer: **qwen3.8-27b** via Groq → automatic **Flan-T5** fallback on any failure
- `GEN_BACKEND=local` → fully offline
- Verifier is a **different** model and must **quote its evidence**
- Generator, verifier, judge = 3 different models

**Visual:** **Guide §4.5 — Generation with fallback.**

**Speaker notes:** "A warning is logged whenever the fallback triggers, because answers quietly get
worse then. I tested the fallback by forcing a bad model name. The quoting requirement turns a
'verified' badge into something a human can check in seconds."

---

## Slide 13 — The product: app and API

**On slide**
- **Streamlit**: Assistant · Analytics · Knowledge graph
- **FastAPI**: `POST /query`, `GET /health`, Swagger docs
- Sources and verification persist in the chat history

**Visual:** `docs/assets/ui-assistant.png` (large) with small insets of `ui-analytics.png` and
`ui-knowledge-graph.png` (if present in `docs/assets/`; otherwise capture them from the running app).
Optionally **Guide §4.6**.

**Speaker notes:** "Two ways in: a human-friendly app and a programmatic API with validated schemas.
The knowledge graph and analytics pages are tools around the system, not part of answering."

---

## Slide 14 — How we measure it

**On slide**
- LLM judge scores **Faithfulness · Answer Relevancy · Context Precision**
- Benchmarks: 7 questions (v1) → **28** (v2)
- **Judge-free** retrieval check: is the right paper in the top 5?
- Judge errors **abort** the run; they are never scored as 0

**Visual:** **Guide §4.7 — Evaluation harness.**

**Speaker notes:** "Faithfulness asks if the answer sticks to the sources; relevancy asks if it
answers the question; precision asks if the passages were relevant. The abort-on-error rule exists
because an earlier bug recorded API failures as zero scores."

---

## Slide 15 — Result 1: generation was the bottleneck

**On slide**
- Same retrieval, same questions, same judge — **only the generator changed**
- Answer Relevancy **0.071 → 0.429 → 0.714** (Flan-T5 → Qwen → Qwen + expansion)
- Faithfulness stays high (0.857–1.000)

**Visual:** bar chart:

| Config | Answer Relevancy |
|---|---:|
| flan-t5-base | 0.071 |
| qwen3.8-27b | 0.429 |
| qwen3.8-27b + expansion | 0.714 |

plus the before/after screenshots `docs/assets/ui-answer-flan-t5.png` | `docs/assets/ui-answer.png`.

**Speaker notes:** "This is the cleanest experiment in the project. Retrieval had been finding the
right passages all along; the small model couldn't use them. A bigger Flan-T5 didn't help, so size
within that family wasn't the answer. Caveat: 7 questions detect large effects like this one, not
small ones."

---

## Slide 16 — Result 2: what did *not* work

**On slide**
- `flan-t5-large`: more cautious, not better ("I don't know" on 6 of 7)
- Query expansion + full text with Flan-T5: relevancy collapsed to **0.000**
- Index only the 145 fuel-cell papers: **no measurable gain**
- Same-model self-check: approved a wrong answer

**Visual:** four "❌ → what we learned" cards (use plain icons, not emoji, if matching the app style).

**Speaker notes:** "Negative results are recorded as decision records, not hidden. Each closed off an
easy explanation and pointed at the real bottleneck."

---

## Slide 17 — Result 3: an explanation I got wrong

**On slide**
- I assumed: "unanswered questions → the corpus lacks those papers"
- Built a 28-question benchmark and **checked**: 11 of 28 abstain
- In 4 of 5 cases checked, the answering paper **was in the corpus** but ranked outside the top 5
- Corrected the ADRs and README the same day

**Visual:** small table (Guide ADR-018): topic · paper in corpus? · in top 5? with 4 "no" and 1 "yes, but model declined".

**Speaker notes:** "I'd written 'coverage is the limit' in two decision records without checking.
The larger benchmark let me test it, and it was wrong: the papers were there, ranked too low. I
corrected the records rather than leaving them. Five cases is evidence, not proof across all eleven."

---

## Slide 18 — Result 4: cross-encoder re-ranking

**On slide**
- Local MiniLM cross-encoder re-scores the fused top 20
- Answering paper in top 5: **15 → 20 of 21** questions
- hit@5 **0.714 → 0.952** · MRR **0.448 → 0.849**
- Cost: about **+100 ms** per query
- **Judged on 28 questions:** "I don't know" **11 → 5** · Answer Relevancy **0.607 → 0.821**
- Cost: faithfulness dipped **0.982 → 0.964** (one answer added a number not in the sources)
- Caveat: gold papers chosen by me; single run per configuration

**Visual:** grouped bar chart:

| Configuration | hit@5 | MRR |
|---|---:|---:|
| Baseline | 0.714 | 0.448 |
| Cross-encoder | 0.952 | 0.849 |

Second chart (judged, 28 questions): Answer Relevancy 0.607 → 0.821; Context Precision 0.582 → 0.689;
abstentions 11 → 5. Adding query expansion on top: relevancy 0.821 (unchanged), faithfulness 0.929, 4 of 28 unanswered: no net gain.

**Speaker notes:** "Measured without any LLM judge, so it costs no API budget and can be re-run any
time. One miss remains: its target paper isn't even in the 20 candidates, so the next try is a larger
candidate pool. I then confirmed it with the judge on all 28 questions: 'I don't know' answers fell
from 11 to 5 and relevancy rose from 0.607 to 0.821. It isn't free: faithfulness dipped slightly, to 0.964,
because one answer quoted a figure that isn't in the retrieved text. I haven't yet judged re-ranking
combined with query expansion: it adds nothing, relevancy stays at 0.821 and faithfulness falls to 0.929."

---

## Slide 19 — Engineering quality

**On slide**
- **60 tests** (51 run anywhere incl. CI; 9 need local data and skip cleanly)
- GitHub Actions CI · ruff · pre-commit
- **19 decision records**, including reversed ones
- Bugs that looked healthy: 100% PDF download failure · judge 404s scored as 0.0 · expansion silently off

**Visual:** a 4-box row (Tests · CI · ADRs · Bug table) or the table from Guide §10 trimmed to 3 rows.

**Speaker notes:** "The biggest engineering lesson was silent failure: three separate bugs produced
plausible-looking numbers while being broken. The fixes make failures loud: the evaluator aborts, the
expansion step preflights its model."

---

## Slide 20 — Limitations

**On slide**
- NER F1 ≈ 0.09 → boost, not filter; ambiguity abstention rarely fires
- Answers depend on an external API (fallback exists, answers get weaker)
- Verification checks consistency, not truth
- Small, author-written benchmarks (7 and 28 questions; judge noise ±0.03)
- One narrow domain, 400 papers

**Visual:** plain list, no diagram.

**Speaker notes:** "I'd rather state these than be caught on them. None invalidates the findings, but
they bound them: the effects I report are large ones, in one domain."

---

## Slide 21 — What is pending

**On slide**
- Decide the query-expansion default: it adds nothing after re-ranking (repeat the run to confirm)
- Look into the one faithfulness regression (grain-boundary answer)
- Larger candidate pool; larger cross-encoder
- Held-out benchmark written by someone else
- Better PDF extraction; local 7–8B generator for offline use

**Visual:** roadmap list with "blocked by quota" tag on the first item.

**Speaker notes:** "The judged re-ranking run is done. The next concrete action is one command on a
day the quota resets. Everything else is incremental."

---

## Slide 22 — Takeaways

**On slide**
1. Domain layer + hybrid retrieval fixes real domain-drift failures
2. Measure, then decide: **10 hypotheses tested, several disproved**
3. Generation, then ranking, were the real bottlenecks — not the corpus
4. Reliability = separate checker + abstention + loud failures
5. Everything reproducible: scripts, 19 ADRs, 60 tests

**Visual:** big numbers strip: **0.071 → 0.714** (generator) · **0.714 → 0.952** (retrieval hit@5) · **11 → 5** abstentions · **19 ADRs** · **60 tests**.

**Speaker notes:** "If you remember one thing: the system improved because each change was tested
against a number, and when a number contradicted me I changed my mind in writing."

---

# Backup slides (for Q&A)

## B1 — Full results table (10 configurations)
Use Guide §7.3 as a table. Note: last row is the default; judge noise ≈ ±0.03.

## B2 — Decision records
Use Guide §9 (19 rows). Group: NER (001–004) · RAG stack (005–009) · Phase 9 extensions (010–013) · Phase 10 (014–019).

## B3 — Configuration and how to run
Use Guide §13 settings table and the run commands.

## B4 — Numbers cheat sheet
Use Guide §15.

---

## Timing guide (20 minutes)

| Slides | Topic | Minutes |
|---|---|---:|
| 1–5 | Problem, idea, background, data | 4 |
| 6–10 | Architecture and retrieval | 5 |
| 11–13 | Demo, generation, product | 3 |
| 14–18 | Evaluation and results | 6 |
| 19–22 | Quality, limits, pending, takeaways | 2 |

## Anticipated questions → where the answer is

| Question | Answer location |
|---|---|
| Why not ChatGPT? / why not fine-tune? | Guide §14 |
| NER F1 is only 9% — failure? | Slide 8 · Guide §14 |
| 100% hit-rate suspicious? | Guide §7.2, §14 |
| Isn't depending on Groq a weakness? | Slide 12 · Guide §14 |
| Has the re-ranker been proven end to end? | Slide 18 — yes, judged on 28 questions (11 → 5 abstentions). Expansion on top adds nothing (relevancy 0.821 either way) |
| What did you get wrong? | Slide 17 |
