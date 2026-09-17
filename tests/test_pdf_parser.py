"""Unit tests for src/ingestion/pdf_parser.py's section-detection logic.

detect_sections() is a pure function (string in, dict out) -- no PDF
files or network access needed, so these always run in CI.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.pdf_parser import detect_sections  # noqa: E402

SAMPLE_PAPER = """Some Paper Title

Abstract
This is the abstract text describing the study of YSZ electrolytes.

1. Introduction
This is the introduction section with background information about SOFCs.

2. Methods
We used pypdf to extract text from PDFs and regex to detect section headings.

3. Results
The results showed good extraction quality for standard academic layouts.

4. Conclusion
We conclude that this works well for most papers.

References
[1] Some citation, Journal of Fuel Cells, 2020.
[2] Another citation.
"""


def test_detect_sections_splits_recognized_headings():
    sections = detect_sections(SAMPLE_PAPER)
    assert "abstract" in sections
    assert "introduction" in sections
    assert "methods" in sections
    assert "results" in sections
    assert "conclusion" in sections
    assert "references" in sections


def test_detect_sections_content_is_correctly_scoped():
    sections = detect_sections(SAMPLE_PAPER)
    assert "YSZ electrolytes" in sections["abstract"]
    assert "background information about SOFCs" in sections["introduction"]
    # Methods content shouldn't bleed into the abstract or vice versa.
    assert "pypdf" not in sections["abstract"]
    assert "YSZ" not in sections["methods"]


def test_detect_sections_falls_back_to_full_text_when_unstructured():
    unstructured = "Just a blob of text with no recognizable section headings anywhere in it."
    sections = detect_sections(unstructured)
    assert sections == {"full_text": unstructured}


def test_detect_sections_falls_back_with_single_heading():
    # A single matched heading isn't enough structure (len(boundaries) < 2).
    single_heading_text = "Abstract\nJust one section, nothing else recognized."
    sections = detect_sections(single_heading_text)
    assert "full_text" in sections


def test_detect_sections_merges_duplicate_headings():
    text = (
        "Introduction\nFirst intro part.\n\n"
        "Methods\nSome method text.\n\n"
        "Results\nFirst results part.\n\n"
        "Results\nSecond results part (e.g. a discussion subsection also titled Results).\n"
    )
    sections = detect_sections(text)
    assert "First results part" in sections["results"]
    assert "Second results part" in sections["results"]
