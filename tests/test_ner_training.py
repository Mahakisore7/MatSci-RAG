"""Tests to verify Phase 2 NER training outputs exist and are valid."""

import os


def test_matscibert_model_saved():
    """MatSciBERT fine-tuned model directory must exist after training."""
    assert os.path.exists("models/matscibert_ner"), (
        "models/matscibert_ner not found — run src/train_ner.py first"
    )


def test_bert_baseline_model_saved():
    """Generic BERT fine-tuned model directory must exist after training."""
    assert os.path.exists("models/bert_baseline_ner"), (
        "models/bert_baseline_ner not found — run src/train_ner.py first"
    )


def test_results_recorded():
    """RESULTS.md must exist and contain the F1 comparison."""
    assert os.path.exists("RESULTS.md"), "RESULTS.md not found"
    content = open("RESULTS.md", encoding="utf-8").read()
    assert "MatSciBERT" in content
    assert "bert-base-uncased" in content
    assert "F1" in content


def test_ner_processed_dataset_exists():
    """Parsed NER dataset must exist under data/ner_processed/."""
    assert os.path.exists("data/ner_processed"), (
        "data/ner_processed not found — run src/prepare_ner_data.py first"
    )
