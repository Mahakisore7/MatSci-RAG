import json
import os

import pytest

requires_facts = pytest.mark.skipif(
    not os.path.exists("data/structured_facts.jsonl"),
    reason="data/structured_facts.jsonl not present (gitignored local artifact)",
)


@requires_facts
def test_structured_facts_file_exists():
    assert os.path.exists("data/structured_facts.jsonl")


@requires_facts
def test_structured_facts_have_required_fields():
    with open("data/structured_facts.jsonl") as f:
        record = json.loads(f.readline())
    for field in ["source_id", "title", "materials", "devices", "experiments", "values"]:
        assert field in record
