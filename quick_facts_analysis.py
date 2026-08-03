import json
from collections import Counter

materials = Counter()
devices = Counter()
experiments = Counter()
values = Counter()

with open("data/structured_facts.jsonl") as f:
    for line in f:
        record = json.loads(line)
        materials.update(record.get("materials", []))
        devices.update(record.get("devices", []))
        experiments.update(record.get("experiments", []))
        values.update(record.get("values", []))

print("Top 10 most-mentioned MATERIALS:")
for mat, count in materials.most_common(10):
    print(f"  {mat}: {count}")

print("\nTop 10 most-mentioned DEVICES:")
for dev, count in devices.most_common(10):
    print(f"  {dev}: {count}")

print("\nTop 10 most-mentioned EXPERIMENTS:")
for exp, count in experiments.most_common(10):
    print(f"  {exp}: {count}")

print("\nTop 10 most-mentioned VALUES:")
for val, count in values.most_common(10):
    print(f"  {val}: {count}")
