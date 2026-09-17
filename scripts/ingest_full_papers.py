"""
scripts/ingest_full_papers.py

End-to-end orchestrator for full-paper ingestion:
    1. Download PDFs from arXiv  (src.ingestion.pdf_downloader)
    2. Parse PDFs into sections  (src.ingestion.pdf_parser)
    3. Save parsed JSONL         -> data/full_papers.jsonl
    4. Rebuild ChromaDB index    (src.rag.build_index  --source full_paper)

Run from the project root:
    # Dry run: preview without downloading
    uv run python scripts/ingest_full_papers.py --dry-run

    # Download + index 10 papers first to test
    uv run python scripts/ingest_full_papers.py --max 10

    # Full ingestion (all 400 papers, ~1 hour)
    uv run python scripts/ingest_full_papers.py
"""

import argparse
import os
import sys

# Must run before the `from src...` imports below: running this file
# directly (rather than via `python -m scripts.ingest_full_papers`) puts
# scripts/ on sys.path[0], not the project root, so `src` isn't
# importable otherwise -- this bit anyone following this file's own
# documented usage above.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger  # noqa: E402
from src.ingestion.pdf_downloader import download_pdfs
from src.ingestion.pdf_parser import parse_all_pdfs
from src.rag.build_index import (
    CHROMA_DIR,
    CHUNK_SIZE_FULL,
    FACTS_PATH,
    FULL_PAPERS_PATH,
    build_documents_from_full_papers,
    build_index,
    chunk_documents,
    load_facts,
    load_full_papers,
)

log = get_logger(__name__)

CORPUS_PATH = "data/sofc_corpus.jsonl"
PDF_DIR = "data/pdfs"


def run_ingestion(max_papers: int | None, delay: float, dry_run: bool) -> None:
    """Orchestrates the full paper ingestion pipeline."""

    log.info("=" * 60)
    log.info("MatSci-RAG: Full Paper Ingestion Pipeline")
    log.info("=" * 60)

    # ── Step 0: Dry-run preview ───────────────────────────────────
    if dry_run:
        log.info("[DRY RUN] Would process steps:")
        log.info("  1. Download PDFs from arXiv to data/pdfs/")
        log.info("  2. Parse PDFs into sections -> data/full_papers.jsonl")
        log.info("  3. Rebuild ChromaDB with full-paper chunks")
        log.info(f"  Corpus: {CORPUS_PATH}")
        log.info(f"  PDF dir: {PDF_DIR}")
        log.info(f"  Max papers: {max_papers or 'all'}")
        return

    # ── Step 1: Download PDFs ─────────────────────────────────────
    log.info("\n[Step 1/3] Downloading PDFs...")
    downloaded, skipped = download_pdfs(
        corpus_path=CORPUS_PATH,
        pdf_dir=PDF_DIR,
        max_papers=max_papers,
        delay=delay,
    )
    log.info(f"Download complete: {downloaded} new, {skipped} skipped/failed")

    if downloaded == 0 and skipped > 0 and not os.path.exists(PDF_DIR):
        log.error("No PDFs downloaded and PDF dir does not exist. Aborting.")
        sys.exit(1)

    # ── Step 2: Parse PDFs into sections ──────────────────────────
    log.info("\n[Step 2/3] Parsing PDFs into sections...")
    parsed, failed = parse_all_pdfs(
        pdf_dir=PDF_DIR,
        corpus_path=CORPUS_PATH,
        output_path=FULL_PAPERS_PATH,
        max_papers=max_papers,
    )
    log.info(f"Parse complete: {parsed} parsed, {failed} failed/skipped")

    if parsed == 0:
        log.error("No papers were successfully parsed. Aborting index rebuild.")
        sys.exit(1)

    # ── Step 3: Rebuild ChromaDB index ────────────────────────────
    log.info("\n[Step 3/3] Rebuilding ChromaDB index with full-paper sections...")
    facts = load_facts(FACTS_PATH)
    full_papers = load_full_papers(FULL_PAPERS_PATH)
    docs = build_documents_from_full_papers(full_papers, facts)
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE_FULL)
    build_index(chunks)

    log.info("\n" + "=" * 60)
    log.info("Full paper ingestion complete!")
    log.info(f"  PDFs saved to: {PDF_DIR}/")
    log.info(f"  Parsed JSONL: {FULL_PAPERS_PATH}")
    log.info(f"  ChromaDB index: {CHROMA_DIR}/")
    log.info(
        "\nNext step: run scripts/evaluate_ragas.py to measure "
        "improvement in Context Precision and Answer Relevancy."
    )
    log.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end full paper ingestion: download PDFs, parse, re-index."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max papers to process (default: all 400). Use --max 10 for a test run.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between arXiv download requests (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without downloading or indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion(max_papers=args.max, delay=args.delay, dry_run=args.dry_run)
