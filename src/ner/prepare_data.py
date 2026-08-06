import csv
import os
import random

from datasets import Dataset, DatasetDict

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

# SOFC-Exp Coarse Tags
LABELS = [
    "O",
    "B-EXPERIMENT",
    "I-EXPERIMENT",
    "B-MATERIAL",
    "I-MATERIAL",
    "B-DEVICE",
    "I-DEVICE",
    "B-VALUE",
    "I-VALUE",
]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl for i, lbl in enumerate(LABELS)}


def load_sofc_exp(csv_dir: str, text_dir: str) -> list[dict]:
    """Parses the SOFC-Exp dataset into HuggingFace dataset format."""
    examples = []

    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue

        csv_path = os.path.join(csv_dir, filename)
        txt_path = os.path.join(text_dir, filename.replace(".csv", ".txt"))

        with open(txt_path, encoding="utf-8", newline="") as f:
            full_text = f.read()

        sentences = {}  # sentence_id -> {"tokens": [], "tags": []}

        with open(csv_path, encoding="utf-8") as f:
            # Format: sentence_id, token_id, start_char, end_char, coarse_tag, fine_tag
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue

                sent_id = row[0]
                start_char = int(row[2])
                end_char = int(row[3])
                coarse_tag = row[4]

                # Some tags might not be in our LABELS list

                if coarse_tag not in LABELS:
                    coarse_tag = "O"

                token = full_text[start_char:end_char]

                if sent_id not in sentences:
                    sentences[sent_id] = {"tokens": [], "tags": []}

                sentences[sent_id]["tokens"].append(token)
                sentences[sent_id]["tags"].append(coarse_tag)

        for _sent_id, data in sentences.items():
            if len(data["tokens"]) > 0:
                examples.append(data)

    return examples


def build_dataset_dict(examples: list[dict], seed: int = 42) -> DatasetDict:
    random.Random(seed).shuffle(examples)
    n = len(examples)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    log.info(f"Total Sentences: {n}")
    log.info(f"Split: {train_end} train / {val_end - train_end} val / {n - val_end} test")

    return DatasetDict(
        {
            "train": Dataset.from_list(examples[:train_end]),
            "validation": Dataset.from_list(examples[train_end:val_end]),
            "test": Dataset.from_list(examples[val_end:]),
        }
    )


if __name__ == "__main__":
    csv_dir = "data/sofc_exp_repo/sofc-exp-corpus/annotations/entity_types_and_slots"
    text_dir = "data/sofc_exp_repo/sofc-exp-corpus/texts"

    examples = load_sofc_exp(csv_dir, text_dir)

    log.info(f"First parsed example: {examples[0]}")

    ds = build_dataset_dict(examples)
    os.makedirs("data/ner_processed", exist_ok=True)
    ds.save_to_disk("data/ner_processed")
    log.info(str(ds))
