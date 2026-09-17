"""
src/ingestion/pdf_downloader.py

Downloads full-text PDFs from arXiv for every paper in sofc_corpus.jsonl.
Implements idempotent, rate-limited downloading (1 req/sec default).

Usage (from project root):
    uv run python -m src.ingestion.pdf_downloader
    uv run python -m src.ingestion.pdf_downloader --max 10 --delay 2.0
"""

import argparse
import json
import os
import time

import arxiv
import requests

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

CORPUS_PATH = "data/sofc_corpus.jsonl"
PDF_DIR = "data/pdfs"
DEFAULT_DELAY = 1.0  # seconds between requests (arXiv ToS: be polite)


def load_paper_ids(corpus_path: str) -> list[tuple[str, str]]:
    """
    Load (paper_id, title) pairs from the corpus JSONL.
    paper_id is the full arXiv entry URL, e.g. http://arxiv.org/abs/1801.06109v2
    """
    papers = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            papers.append((doc["id"], doc.get("title", "")))
    return papers


def arxiv_id_from_url(url: str) -> str:
    """
    Extracts the arXiv ID from a full entry URL.
    e.g. 'http://arxiv.org/abs/1801.06109v2' -> '1801.06109v2'

    Old-style (pre-2007) IDs carry a category prefix that's part of the ID
    itself, not just a path segment -- taking only the last '/'-segment
    silently drops it:
    e.g. 'http://arxiv.org/abs/physics/0609025v1' -> 'physics/0609025v1'
    (NOT '0609025v1' -- that's not a valid arXiv ID on its own and the API
    rejects it with HTTP 400).
    """
    after_abs = url.rstrip("/").split("/abs/", 1)[-1]
    return after_abs


def safe_filename(arxiv_id: str) -> str:
    """arxiv_id may contain '/' for old-style IDs (e.g. 'physics/0609025v1'),
    which isn't valid in a filename -- flatten it for local storage."""
    return arxiv_id.replace("/", "_")


def download_pdfs(
    corpus_path: str = CORPUS_PATH,
    pdf_dir: str = PDF_DIR,
    max_papers: int | None = None,
    delay: float = DEFAULT_DELAY,
) -> tuple[int, int]:
    """
    Download PDFs for all papers in the corpus. Skips already-downloaded files.

    Returns:
        (downloaded, skipped) counts
    """
    os.makedirs(pdf_dir, exist_ok=True)

    papers = load_paper_ids(corpus_path)
    if max_papers:
        papers = papers[:max_papers]

    log.info(f"Processing {len(papers)} papers -> {pdf_dir}/")

    client = arxiv.Client()
    downloaded = 0
    skipped = 0

    for i, (paper_id, title) in enumerate(papers):
        arxiv_id = arxiv_id_from_url(paper_id)
        pdf_path = os.path.join(pdf_dir, f"{safe_filename(arxiv_id)}.pdf")

        # Idempotent: skip if already downloaded
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
            log.debug(f"[{i+1}/{len(papers)}] Skip (exists): {arxiv_id}")
            skipped += 1
            continue

        try:
            # Strip version suffix for the API lookup (e.g. 1801.06109v2 -> 1801.06109)
            base_id = arxiv_id.split("v")[0]
            search = arxiv.Search(id_list=[base_id])
            results = list(client.results(search))

            if not results:
                log.warning(f"[{i+1}/{len(papers)}] Not found on arXiv: {base_id}")
                skipped += 1
                continue

            result = results[0]
            # arxiv.Result has no download_pdf() method in this installed
            # version (4.0.0) -- it was silently failing every single
            # download before this fix. Fetch result.pdf_url directly.
            response = requests.get(result.pdf_url, timeout=30)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            downloaded += 1
            log.info(f"[{i+1}/{len(papers)}] Downloaded: {arxiv_id} — {title[:60]}")

        except Exception as e:
            log.warning(f"[{i+1}/{len(papers)}] Failed {arxiv_id}: {e}")
            skipped += 1

        time.sleep(delay)

    log.info(f"Done. Downloaded: {downloaded}, Skipped/failed: {skipped}")
    return downloaded, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download arXiv PDFs for the SOFC corpus.")
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--pdf-dir", default=PDF_DIR)
    parser.add_argument(
        "--max", type=int, default=None, help="Max papers to process (default: all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between requests (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_pdfs(
        corpus_path=args.corpus,
        pdf_dir=args.pdf_dir,
        max_papers=args.max,
        delay=args.delay,
    )
