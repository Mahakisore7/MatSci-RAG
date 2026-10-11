"""
Corpus relevance filter (ADR-017).

The arXiv keyword search that built the corpus matched words like "cell",
"anode" and "electrolyte", so most of the 400 papers are about Li-ion
batteries, catalysis or unrelated physics. A paper is kept only if its title
or abstract names a fuel-cell / solid-oxide technology explicitly.

Wider terms (ceria, YSZ, oxygen vacancy, ...) were tried and only added
off-topic papers (Li-ion cathodes, water treatment), so the list stays strict.
"""

import re

FUEL_CELL_PATTERN = re.compile(
    r"fuel cell|\bSOFCs?\b|solid[- ]oxide|\bSOECs?\b|electrolysis cell|\bPEMFCs?\b|\bPCFCs?\b",
    re.IGNORECASE,
)


def is_fuel_cell_paper(paper: dict) -> bool:
    """True if the paper's title or abstract mentions a fuel-cell technology."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    return bool(FUEL_CELL_PATTERN.search(text))


def filter_corpus(corpus: dict[str, dict]) -> dict[str, dict]:
    """Keep only fuel-cell papers from a corpus keyed by paper id."""
    return {pid: paper for pid, paper in corpus.items() if is_fuel_cell_paper(paper)}
