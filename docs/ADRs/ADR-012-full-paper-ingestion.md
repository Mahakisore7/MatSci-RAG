# ADR-012: Full Paper Ingestion via PDF Download + Section Parsing

**Status:** Accepted  
**Date:** 2026-09-12  
**Phase:** Phase 9 Extension  
**Branch:** `feature/full-paper-ingestion`

---

## Context

The RAGAS-style evaluation (ADR-010) showed that Answer Relevancy was 0.214 and Context Precision was 0.200. Root cause: the corpus only contains **abstracts** (~150 words each). Four of seven benchmark questions require information only present in the *Methods* or *Results* sections of the full papers — sections that were not available for retrieval.

The fix is to extend the offline ingestion pipeline to download and parse full paper PDFs, giving the retriever access to richer, more specific content.

---

## Decision

### PDF Library: `pypdf` (v6.x)

**Chosen:** `pypdf` (pure Python, no Java or Ghostscript dependency)

**Alternatives considered:**

| Library | Pros | Why Rejected |
|---|---|---|
| `pdfminer.six` | Good layout analysis | No Python 3.13 support yet; slow |
| `PyMuPDF` (fitz) | Fast, excellent quality | C extension, requires separate MuPDF install |
| `pdfplumber` | Built on pdfminer, nice API | Same Python 3.13 issue as pdfminer |
| `pypdf` ✅ | Pure Python, fast install, Python 3.13 compatible | — |

ArXiv PDFs are almost exclusively machine-generated (LaTeX → PDF), so `pypdf`'s text-layer extraction works well. The only known failure mode is pre-2010 scanned papers, which we gracefully skip.

---

### Section Detection Strategy: Regex Headings

Full academic papers are split into sections using regex pattern matching on heading lines (lines that consist only of the heading word, optionally preceded by a section number).

Patterns match:
- `Abstract`, `Introduction`, `Related Work`, `Background`, `Literature Review`
- `Method`, `Methods`, `Methodology`, `Experimental`, `Materials and Methods`, `Synthesis`
- `Results`, `Results and Discussion`, `Findings`
- `Discussion`, `Conclusion`, `Conclusions`, `Summary`
- `References`, `Bibliography` (indexed separately, then excluded from retrieval)

**Fallback:** If fewer than 2 headings are detected (some PDFs have unusual formatting), the entire extracted text is stored as a single `full_text` section to ensure no paper is completely lost.

---

### Each Section → Separate LangChain Document

Rather than concatenating all sections into one long string, each section becomes a separate `Document` object. This means:
- The `section` metadata field (e.g. `"methods"`, `"results"`) is stored per-chunk in ChromaDB.
- Future work can add section-filtered retrieval (e.g. "only look in Methods sections for synthesis procedures").
- Chunks are naturally bounded by section, preventing a methods-chunk from spanning into a results-discussion.

---

### Chunk Size: 800 chars (up from 500 for abstracts)

Abstracts are compressed, dense summaries (avg ~150 words). Full-paper prose is more expansive (avg sentence length is longer, ideas develop over paragraphs). The 500-char abstract chunk size was tuned for dense text; 800 chars gives each full-paper chunk the room to contain a complete idea.

---

### Re-indexing Strategy: Replace (Clean Rebuild)

When `--source full_paper` is passed to `build_index.py`, the existing `sofc_papers` ChromaDB collection is replaced with a completely new index built from full-paper sections. This is chosen over an additive approach because:
- Duplicate information (abstract text ≈ introduction text in many papers) would pollute retrieval results
- The section-level chunks are strictly better than abstract-only chunks for retrieval quality
- A clean rebuild ensures consistent metadata schema across all chunks

---

## New Files

| File | Purpose |
|---|---|
| `src/ingestion/pdf_downloader.py` | Downloads PDFs from arXiv; idempotent skip-if-exists |
| `src/ingestion/pdf_parser.py` | Extracts text, detects sections, outputs `data/full_papers.jsonl` |
| `scripts/ingest_full_papers.py` | End-to-end orchestrator (download → parse → reindex) |

## Modified Files

| File | Change |
|---|---|
| `src/rag/build_index.py` | Added `--source` flag, `load_full_papers()`, `build_documents_from_full_papers()`, larger chunk size constant |

---

## Consequences

- `data/pdfs/` directory: ~2 GB for 400 papers (gitignored)
- `data/full_papers.jsonl`: ~100–200 MB parsed sections (gitignored)
- ChromaDB collection will grow from ~1,330 abstract-only chunks to an estimated **4,000–8,000 section-level chunks**
- Re-embedding time: ~25–30 minutes on CPU (BGE-small-en-v1.5 is fast)
- `scripts/ingest_full_papers.py --dry-run` can preview the pipeline without side effects
- `--max N` flag allows testing on N papers before committing to the full run
