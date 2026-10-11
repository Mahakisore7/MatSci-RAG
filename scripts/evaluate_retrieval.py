"""
Judge-free retrieval evaluation (ADR-019).

For each benchmark-v2 question with gold target papers (scripts/retrieval_gold.py),
measure whether retrieval puts an answering paper in the final top-k:

  - hit@k   : fraction of questions with at least one gold paper in the top k
  - MRR     : mean reciprocal rank of the first gold paper (0 if absent)
  - papers  : mean number of distinct papers in the top k

It calls no LLM, so it spends none of the Groq judge's daily token budget and can
be re-run freely. Run from the project root:
    uv run python scripts/evaluate_retrieval.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.retrieval_gold import GOLD_TARGETS  # noqa: E402

CONFIGS = {
    "baseline (RRF + NER)": {"use_rerank": False, "per_paper_cap": None},
    "cap 2 per paper": {"use_rerank": False, "per_paper_cap": 2},
    "cross-encoder": {"use_rerank": True, "per_paper_cap": None},
    "cross-encoder + cap 2": {"use_rerank": True, "per_paper_cap": 2},
}


def first_gold_rank(docs, patterns) -> int | None:
    """1-based rank of the first doc whose title matches a gold pattern, else None."""
    pats = [p.lower() for p in patterns]
    for rank, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "").lower()
        if any(p in title for p in pats):
            return rank
    return None


def check_gold_in_corpus(corpus_path: str = "data/sofc_corpus.jsonl") -> None:
    """Fail loudly if a gold pattern matches no corpus title (a typo would read as a miss)."""
    with open(corpus_path, encoding="utf-8") as f:
        titles = [json.loads(line)["title"].lower() for line in f]
    for question, patterns in GOLD_TARGETS.items():
        for p in patterns:
            if not any(p.lower() in t for t in titles):
                raise SystemExit(f"Gold pattern matches no corpus title: {p!r} ({question})")


def main(k: int, output: str | None) -> None:
    check_gold_in_corpus()
    from src.rag.hybrid_retrieval import hybrid_retrieve

    rows = []
    for name, cfg in CONFIGS.items():
        ranks, distinct = [], []
        for question, patterns in GOLD_TARGETS.items():
            docs = hybrid_retrieve(question, k=k, use_ner_boost=True, **cfg)
            ranks.append(first_gold_rank(docs, patterns))
            distinct.append(len({d.metadata.get("paper_id") for d in docs}))
        n = len(ranks)
        hit = sum(r is not None for r in ranks) / n
        mrr = sum(1 / r for r in ranks if r) / n
        rows.append(
            {
                "config": name,
                "hit_at_k": round(hit, 3),
                "mrr": round(mrr, 3),
                "distinct_papers": round(sum(distinct) / n, 2),
                "questions": n,
                "k": k,
            }
        )
        print(f"{name:26s} hit@{k}={hit:.3f}  MRR={mrr:.3f}  papers/top{k}={sum(distinct) / n:.2f}")

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge-free retrieval evaluation.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="docs/results/bench28/retrieval_gold.json")
    args = parser.parse_args()
    main(args.k, args.output)
