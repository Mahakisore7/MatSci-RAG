import json
import os


def test_corpus_file_exists_and_has_records():
    assert os.path.exists("data/corpus.jsonl")
    with open("data/corpus.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 100


def test_corpus_records_have_required_fields():
    with open("data/corpus.jsonl", encoding="utf-8") as f:
        record = json.loads(f.readline())
    for field in ["id", "title", "abstract", "published"]:
        assert field in record
