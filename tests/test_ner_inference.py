import json
import os


def test_structured_facts_file_exists():
    assert os.path.exists("data/structured_facts.jsonl")


def test_structured_facts_have_required_fields():
    with open("data/structured_facts.jsonl") as f:
        record = json.loads(f.readline())
    for field in ["source_id", "title", "materials", "devices", "experiments", "values"]:
        assert field in record
