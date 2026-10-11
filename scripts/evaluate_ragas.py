"""
RAG Evaluation Script using Groq as Judge LLM.

Measures three core RAG quality metrics without RAGAS dependency:
  - Faithfulness:      Does the answer only use facts from the retrieved context?
  - Answer Relevancy: Is the answer actually relevant to the question?
  - Context Precision: Is the retrieved context relevant to the question?

Scoring is done by the Groq model in src.core.config.GROQ_JUDGE_MODEL (free tier).
Groq has retired three judge models so far (llama-3.1-8b-instant, llama3-8b-8192,
groq/compound-mini -- see ADR-010), so scores are only comparable across runs that
used the same judge.
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

# Generated answers can contain symbols the Windows console codepage (cp1252) cannot
# encode; a print() failure must not kill a long evaluation run.
sys.stdout.reconfigure(errors="replace")

from groq import (  # noqa: E402
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

from scripts.benchmark_questions import BENCHMARKS  # noqa: E402
from src.core.config import GROQ_JUDGE_MODEL, active_generator  # noqa: E402
from src.rag.rag_query import answer_query  # noqa: E402

# Transient infrastructure failures worth retrying (429, 5xx "over capacity",
# dropped connections). Anything else -- e.g. a 404 for a retired model --
# aborts the run instead of being recorded as a score.
_RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


# ------------------------------------------------------------------
# Groq judge
# ------------------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def parse_score(raw: str) -> float:
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        return float(json.loads(match.group())["score"])
    num = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
    if num:
        return float(num.group())
    raise ValueError(f"Judge reply has no score: {raw!r}")


def groq_score(prompt: str, retries: int = 5) -> float:
    """Send a scoring prompt to the judge; expect JSON back with key 'score' 0-1.

    Never returns a made-up 0.0 on failure: an infrastructure error scored as
    "unfaithful" is indistinguishable from a real judgment and silently
    corrupts the averages. Transient errors are retried with exponential
    backoff; everything else raises and stops the run.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                # gpt-oss is a reasoning model; its reasoning tokens count
                # against max_tokens, so leave ample headroom for the JSON.
                max_tokens=1024,
                reasoning_effort="low",
            )
            return parse_score(response.choices[0].message.content or "")
        except _RETRYABLE as e:
            wait = 5.0 * 2**attempt
            print(f"    [Groq transient error, retry {attempt + 1}/{retries} in {wait:.0f}s] {e}")
            time.sleep(wait)
    raise RuntimeError(f"Judge still failing after {retries} retries; aborting evaluation.")


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
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
    ctx = "\n\n".join(f"[{i + 1}] {c[:300]}..." for i, c in enumerate(contexts))
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
def main(
    expand_q: bool = False,
    output_path: str = "docs/results/ragas_evaluation.csv",
    benchmark: str = "v1",
    resume: bool = False,
):
    questions = BENCHMARKS[benchmark]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []

    # Each scored question is appended here so a run stopped by Groq's daily token
    # cap (a 28-question run uses nearly all of it) can continue instead of restart.
    partial_path = output_path + ".partial.jsonl"
    done: dict[str, dict] = {}
    if resume and os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            done = {r["question"]: r for r in map(json.loads, f)}
        print(f"Resuming: {len(done)} questions already scored in {partial_path}\n")
    elif os.path.exists(partial_path):
        os.remove(partial_path)

    if expand_q:
        # expand_query() degrades to the raw query on any error, which would
        # silently turn an "--expand" run into a no-expansion run. Fail fast.
        from src.rag.query_expansion import get_expansion_llm

        llm = get_expansion_llm()
        if llm is None:
            raise SystemExit("--expand needs GROQ_API_KEY set.")
        llm.invoke("Reply with OK.")

    print(f"{'=' * 60}")
    print(
        f"  MatSci-RAG Evaluation  |  Generator: {active_generator()}  |  "
        f"Judge: {GROQ_JUDGE_MODEL}  |  expand_q={expand_q}"
    )
    print(f"{'=' * 60}\n")

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")

        if question in done:
            print("  (already scored, reusing)\n")
            results.append(done[question])
            continue

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

        row = {
            "question": question,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
        }
        results.append(row)
        with open(partial_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    if results:
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_prec = sum(r["context_precision"] for r in results) / len(results)

        print(f"\n{'=' * 60}")
        print("  AGGREGATE SCORES")
        print(f"{'=' * 60}")
        print(f"  Faithfulness:       {avg_faith:.3f}")
        print(f"  Answer Relevancy:   {avg_rel:.3f}")
        print(f"  Context Precision:  {avg_prec:.3f}")
        print(f"  Questions scored:   {len(results)}/{len(questions)}")
        print(f"{'=' * 60}")

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
        if os.path.exists(partial_path):
            os.remove(partial_path)
    else:
        print("No results were collected.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Groq LLM-judge RAG evaluation (Faithfulness / Answer Relevancy / Context Precision)."
        )
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Enable LLM query expansion before retrieval (default: off, matches the app).",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default="v1",
        help="v1 = original 7 questions (default); v2 = 28 questions (ADR-018).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from <output>.partial.jsonl (same settings only).",
    )
    parser.add_argument(
        "--output",
        default="docs/results/ragas_evaluation.csv",
        help="Where to write the results CSV (default: docs/results/ragas_evaluation.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        expand_q=args.expand,
        output_path=args.output,
        benchmark=args.benchmark,
        resume=args.resume,
    )
