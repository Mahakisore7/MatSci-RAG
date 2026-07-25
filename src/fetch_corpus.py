import json
import os

import arxiv

from logging_config import get_logger

log = get_logger(__name__)


QUERY = "high entropy alloy hardness"
MAX_RESULTS = 400
OUTPUT_PATH = "data/corpus.jsonl"


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


if __name__ == "__main__":
    fetch_corpus(QUERY, MAX_RESULTS, OUTPUT_PATH)
