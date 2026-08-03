"""Applies the fine-tuned MatSciBERT NER model to every abstract in our own
corpus (not the Matscholar benchmark) and produces a structured facts table."""

import json

from transformers import AutoTokenizer, pipeline

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)


def truncate_to_512(text: str, tokenizer) -> str:
    """Truncates text to at most 512 tokens and decodes back to a string.
    This is required because BERT has a hard 512-token position limit."""
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors=None,
    )
    # Decode back to a string, skipping the special [CLS] and [SEP] tokens
    return tokenizer.decode(tokens["input_ids"][1:-1], skip_special_tokens=True)


def run_inference(corpus_path: str, model_dir: str, output_path: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ner_pipeline = pipeline(
        "token-classification",
        model=model_dir,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0,  # 0 means use the GPU
    )

    results = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            # Pre-truncate long abstracts before they hit BERT's 512 token limit
            abstract = truncate_to_512(paper["abstract"], tokenizer)
            entities = ner_pipeline(abstract)

            materials = [e["word"] for e in entities if "MATERIAL" in e["entity_group"]]
            devices = [e["word"] for e in entities if "DEVICE" in e["entity_group"]]
            experiments = [e["word"] for e in entities if "EXPERIMENT" in e["entity_group"]]
            values = [e["word"] for e in entities if "VALUE" in e["entity_group"]]

            results.append(
                {
                    "source_id": paper["id"],
                    "title": paper["title"],
                    "materials": materials,
                    "devices": devices,
                    "experiments": experiments,
                    "values": values,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info(f"Wrote structured facts for {len(results)} abstracts to {output_path}")
    return len(results)


if __name__ == "__main__":
    run_inference("data/corpus.jsonl", "models/matscibert_ner", "data/structured_facts.jsonl")
