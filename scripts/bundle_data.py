"""Package the runtime knowledge base (ChromaDB + facts + corpus) for cloud deployment.

Usage:
    uv run python scripts/bundle_data.py
"""

import os
import tarfile
import time


def create_bundle(output_file: str = "data_bundle.tar.gz") -> str:
    start_time = time.time()
    targets = [
        ("data/chroma_db", "data/chroma_db"),
        ("data/structured_facts.jsonl", "data/structured_facts.jsonl"),
        ("data/sofc_corpus.jsonl", "data/sofc_corpus.jsonl"),
        ("data/full_papers.jsonl", "data/full_papers.jsonl"),
    ]

    for local_path, _ in targets:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Missing required artifact: {local_path}")

    print(f"Creating archive '{output_file}'...")
    with tarfile.open(output_file, "w:gz") as tar:
        for local_path, arcname in targets:
            print(f"  Adding {local_path} -> {arcname}")
            tar.add(local_path, arcname=arcname)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    duration = time.time() - start_time
    print(f"\nBundle created successfully: {output_file}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Time taken: {duration:.1f}s")
    return output_file


if __name__ == "__main__":
    create_bundle()
