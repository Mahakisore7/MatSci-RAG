"""
RAG Evaluation Script using Groq as Judge LLM.

Measures three core RAG quality metrics without RAGAS dependency:
  - Faithfulness:      Does the answer only use facts from the retrieved context?
  - Answer Relevancy: Is the answer actually relevant to the question?
  - Context Precision: Is the retrieved context relevant to the question?

All scoring is done by groq/compound-mini via Groq API (free tier). (Llama-3.1-8b-instant
and llama3-8b-8192 were tried first and decommissioned/404'd before landing here — see ADR-010.)
Results saved to docs/results/ragas_evaluation.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Path setup — must run from project root: uv run python scripts/evaluate_ragas.py
# ------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from groq import Groq, RateLimitError  # noqa: E402

from src.rag.rag_query import answer_query  # noqa: E402

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
GROQ_MODEL = "groq/compound-mini"

BENCHMARK_QUESTIONS = [
    "What materials are used as electrolytes in solid oxide fuel cells?",
    "What is the effect of temperature on SOFC performance?",
    "Which cathode materials show high oxygen reduction activity in SOFCs?",
    "What causes anode degradation in Ni-YSZ SOFCs?",
    "How does grain size affect the ionic conductivity of ceria?",
    "What is the role of GDC as an interlayer in SOFC cathodes?",
    "What sintering temperatures are typically used for LSCF cathodes?",
]

# ------------------------------------------------------------------
# Groq judge
# ------------------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def groq_score(prompt: str, retries: int = 3) -> float:
    """Send a scoring prompt to Groq; expect JSON back with key 'score' 0-1.

    Retries on rate-limit errors (with backoff) so a transient 429 on the
    free tier records as a retried success, not a false 0.0 "unfaithful"
    score -- conflating infrastructure failure with content quality would
    silently corrupt the evaluation numbers.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            # Try to extract a JSON score
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                return float(json.loads(match.group())["score"])
            # Fallback: look for a bare number
            num = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
            if num:
                return float(num.group())
            return 0.0
        except RateLimitError as e:
            wait = 5.0 * (attempt + 1)
            print(f"    [Groq rate limit, retry {attempt + 1}/{retries} in {wait:.0f}s] {e}")
            time.sleep(wait)
        except Exception as e:
            print(f"    [Groq error] {e}")
            return 0.0
    print("    [Groq error] rate-limited after all retries, scoring as 0.0")
    return 0.0


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are a strict faithfulness evaluator.\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"ANSWER: {answer}\n\n"
        "Question: Does the answer rely ONLY on information in the context above, "
        "with no hallucination or external knowledge?\n"
        'Reply with JSON only: {"score": 1.0} if fully faithful, {"score": 0.0} if not, '
        'or {"score": 0.5} if partially faithful.'
    )
    return groq_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = (
        "You are a relevancy evaluator.\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n\n"
        "Question: Is this answer directly relevant and useful for the question? "
        'Reply with JSON only: {"score": 1.0} for very relevant, {"score": 0.5} '
        'for partially relevant, {"score": 0.0} for irrelevant or "I don\'t know".'
    )
    return groq_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    ctx = "\n\n".join(f"[{i+1}] {c[:300]}..." for i, c in enumerate(contexts))
    prompt = (
        "You are a retrieval precision evaluator.\n"
        f"QUESTION: {question}\n"
        f"RETRIEVED CONTEXTS (first 300 chars each):\n{ctx}\n\n"
        "Question: What fraction of the retrieved passages above are actually "
        "relevant to answering the question?\n"
        'Reply with JSON only: {"score": X} where X is a float between 0.0 and 1.0.'
    )
    return groq_score(prompt)


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------
def main(expand_q: bool = False, output_path: str = "docs/results/ragas_evaluation.csv"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []

    print(f"{'='*60}")
    print(f"  MatSci-RAG Evaluation  |  Judge: {GROQ_MODEL}  |  expand_q={expand_q}")
    print(f"{'='*60}\n")

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"[{i}/{len(BENCHMARK_QUESTIONS)}] {question}")

        answer, docs = answer_query(question, use_hybrid=True, expand_q=expand_q)
        contexts = [d.page_content for d in docs]

        if "Multiple candidate materials found" in answer:
            print("  → Abstained (ambiguous query), skipping.\n")
            continue

        print(f"  Answer: {answer[:80]}...")

        # Rate-limit: Groq free tier allows ~30 req/min
        faith = score_faithfulness(question, answer, contexts)
        time.sleep(1.5)
        relevancy = score_answer_relevancy(question, answer)
        time.sleep(1.5)
        precision = score_context_precision(question, contexts)
        time.sleep(1.5)

        print(f"  Faithfulness={faith:.2f}  Relevancy={relevancy:.2f}  Precision={precision:.2f}\n")

        results.append(
            {
                "question": question,
                "answer": answer,
                "faithfulness": faith,
                "answer_relevancy": relevancy,
                "context_precision": precision,
            }
        )

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    if results:
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_prec = sum(r["context_precision"] for r in results) / len(results)

        print(f"\n{'='*60}")
        print("  AGGREGATE SCORES")
        print(f"{'='*60}")
        print(f"  Faithfulness:       {avg_faith:.3f}")
        print(f"  Answer Relevancy:   {avg_rel:.3f}")
        print(f"  Context Precision:  {avg_prec:.3f}")
        print(f"  Questions scored:   {len(results)}/{len(BENCHMARK_QUESTIONS)}")
        print(f"{'='*60}")

        csv_path = output_path
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question",
                    "answer",
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
            writer.writerow(
                {
                    "question": "AVERAGE",
                    "answer": "",
                    "faithfulness": round(avg_faith, 3),
                    "answer_relevancy": round(avg_rel, 3),
                    "context_precision": round(avg_prec, 3),
                }
            )

        print(f"\n  Detailed results saved -> {csv_path}")
    else:
        print("No results were collected.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Groq LLM-judge RAG evaluation "
            "(Faithfulness / Answer Relevancy / Context Precision)."
        )
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Enable LLM query expansion before retrieval (default: off, matches the app).",
    )
    parser.add_argument(
        "--output",
        default="docs/results/ragas_evaluation.csv",
        help="Where to write the results CSV (default: docs/results/ragas_evaluation.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(expand_q=args.expand, output_path=args.output)
