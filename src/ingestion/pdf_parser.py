"""
src/ingestion/pdf_parser.py

Extracts and section-splits text from arXiv PDFs downloaded to data/pdfs/.
Uses pypdf for text extraction (pure Python, no Java/Ghostscript needed).

Section detection uses regex heuristics on common academic paper headings.
Gracefully handles corrupted/encrypted PDFs and image-only (scanned) papers.

Usage:
    uv run python -m src.ingestion.pdf_parser
    uv run python -m src.ingestion.pdf_parser --max 10
"""

import argparse
import json
import os
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

PDF_DIR = "data/pdfs"
CORPUS_PATH = "data/sofc_corpus.jsonl"
OUTPUT_PATH = "data/full_papers.jsonl"

# ── Section detection ─────────────────────────────────────────────────────────
# These patterns match common section headings in academic PDFs.
# Order matters: first match wins for section boundary detection.
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE | re.MULTILINE)),
    ("introduction", re.compile(r"^\s*\d*\.?\s*introduction\s*$", re.IGNORECASE | re.MULTILINE)),
    (
        "related_work",
        re.compile(
            r"^\s*\d*\.?\s*(related work|background|literature review)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "methods",
        re.compile(
            r"^\s*\d*\.?\s*(method|methods|methodology|experimental|materials and methods"
            r"|experimental methods|computational methods|synthesis)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "results",
        re.compile(
            r"^\s*\d*\.?\s*(results|results and discussion|findings)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    ("discussion", re.compile(r"^\s*\d*\.?\s*discussion\s*$", re.IGNORECASE | re.MULTILINE)),
    (
        "conclusion",
        re.compile(
            r"^\s*\d*\.?\s*(conclusion|conclusions|summary)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "references",
        re.compile(r"^\s*\d*\.?\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE),
    ),
]

# Sections we actually want to index (exclude boilerplate references)
SECTIONS_TO_INDEX = {"abstract", "introduction", "methods", "results", "discussion", "conclusion"}


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extract all text from a PDF. Returns None if the PDF is corrupted,
    encrypted, or image-only (no extractable text layer).
    """
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            log.warning(f"Encrypted PDF, skipping: {pdf_path}")
            return None

        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        if not pages_text:
            log.warning(f"No extractable text (likely scanned): {pdf_path}")
            return None

        return "\n".join(pages_text)

    except PdfReadError as e:
        log.warning(f"Corrupted PDF ({pdf_path}): {e}")
        return None
    except Exception as e:
        log.warning(f"Unexpected error reading {pdf_path}: {e}")
        return None


def detect_sections(full_text: str) -> dict[str, str]:
    """
    Split full paper text into named sections using regex heading detection.
    Falls back to storing the entire text under 'full_text' if no sections
    are detected (e.g. heavily formatted PDFs where headings aren't on their
    own lines).
    """
    # Find all section boundaries: (char_offset, section_name)
    boundaries: list[tuple[int, str]] = []
    for section_name, pattern in SECTION_PATTERNS:
        for match in pattern.finditer(full_text):
            boundaries.append((match.start(), section_name))

    if len(boundaries) < 2:
        # No structure detected — return whole text as body
        return {"full_text": full_text.strip()}

    # Sort by position in document
    boundaries.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}
    for i, (start_pos, section_name) in enumerate(boundaries):
        # Text for this section ends at the start of the next section
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)

        # Find the end of the heading line to skip it
        heading_end = full_text.find("\n", start_pos)
        if heading_end == -1 or heading_end >= end_pos:
            heading_end = start_pos

        content = full_text[heading_end:end_pos].strip()

        # Merge duplicate sections (e.g. two "results" matches)
        if section_name in sections:
            sections[section_name] += "\n" + content
        else:
            sections[section_name] = content

    return sections


def parse_pdf(pdf_path: str, paper_id: str, title: str) -> dict | None:
    """
    Parse a single PDF and return a structured record, or None on failure.
    """
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text or len(full_text) < 200:
        return None

    sections = detect_sections(full_text)

    # Only keep sections we want to index
    indexed_sections = {k: v for k, v in sections.items() if k in SECTIONS_TO_INDEX}

    # Fallback: if no named sections detected, use full_text
    if not indexed_sections and "full_text" in sections:
        indexed_sections["full_text"] = sections["full_text"]

    total_chars = sum(len(v) for v in indexed_sections.values())
    log.debug(f"Parsed {paper_id}: {list(indexed_sections.keys())}, {total_chars:,} chars total")

    return {
        "paper_id": paper_id,
        "title": title,
        "sections": indexed_sections,
    }


def parse_all_pdfs(
    pdf_dir: str = PDF_DIR,
    corpus_path: str = CORPUS_PATH,
    output_path: str = OUTPUT_PATH,
    max_papers: int | None = None,
) -> tuple[int, int]:
    """
    Parse all downloaded PDFs and write structured records to output_path JSONL.

    Returns:
        (parsed, failed) counts
    """
    # Load corpus to get paper_id -> title mapping
    id_to_title: dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            id_to_title[doc["id"]] = doc.get("title", "")

    # Find all PDFs on disk
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
    if max_papers:
        pdf_files = pdf_files[:max_papers]

    log.info(f"Parsing {len(pdf_files)} PDFs from {pdf_dir}/")

    parsed = 0
    failed = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, filename in enumerate(pdf_files):
            arxiv_id = filename.replace(".pdf", "")
            pdf_path = os.path.join(pdf_dir, filename)

            # Reconstruct paper_id URL from arxiv_id
            # Match against corpus (corpus uses full URL as id)
            paper_id = next(
                (pid for pid in id_to_title if arxiv_id in pid),
                f"http://arxiv.org/abs/{arxiv_id}",
            )
            title = id_to_title.get(paper_id, "")

            record = parse_pdf(pdf_path, paper_id, title)
            if record:
                out.write(json.dumps(record) + "\n")
                parsed += 1
                if (i + 1) % 50 == 0:
                    log.info(f"Progress: {i+1}/{len(pdf_files)} ({parsed} parsed, {failed} failed)")
            else:
                failed += 1
                log.warning(f"[{i+1}/{len(pdf_files)}] Failed to parse: {filename}")

    log.info(f"Done. Parsed: {parsed}, Failed/skipped: {failed}")
    log.info(f"Output -> {output_path}")
    return parsed, failed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse arXiv PDFs into section-split JSONL.")
    parser.add_argument("--pdf-dir", default=PDF_DIR)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--max", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parse_all_pdfs(
        pdf_dir=args.pdf_dir,
        corpus_path=args.corpus,
        output_path=args.output,
        max_papers=args.max,
    )
