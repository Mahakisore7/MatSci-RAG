import argparse
import json
import os

import arxiv

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)


# Phase 1's original query/output — the corpus this project pivoted away
# from (see ADR-001/ADR-003). Kept as the default so running this script
# with no arguments reproduces data/corpus.jsonl exactly as before.
QUERY = "high entropy alloy hardness"
MAX_RESULTS = 400
OUTPUT_PATH = "data/corpus.jsonl"

# The query actually used (per ADR-003) to fetch data/sofc_corpus.jsonl —
# the corpus build_index.py and the rest of the RAG pipeline index and
# query today. Reproduce it with:
#   uv run python -m src.ingestion.fetch_corpus \
#       --query "solid oxide fuel cell SOFC electrolyte anode cathode material" \
#       --output data/sofc_corpus.jsonl
SOFC_QUERY = "solid oxide fuel cell SOFC electrolyte anode cathode material"


def fetch_corpus(query: str, max_results: int, output_path: str) -> int:
    """Fetch abstracts from arXiv matching `query` and save as JSONL.

    Returns the number of records saved.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for result in client.results(search):
            record = {
                "id": result.entry_id,
                "title": result.title,
                "abstract": result.summary.replace("\n", " ").strip(),
                "published": str(result.published),
                "categories": result.categories,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    log.info(f"Saved {count} abstracts to {output_path}")
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch an arXiv abstract corpus as JSONL.")
    parser.add_argument("--query", default=QUERY, help=f"arXiv search query (default: {QUERY!r})")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    parser.add_argument("--output", default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_corpus(args.query, args.max_results, args.output)
