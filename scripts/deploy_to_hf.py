"""Deployment helper: Push MatSci-RAG code and LFS-tracked data to Hugging Face Spaces.

Usage:
    uv run python scripts/deploy_to_hf.py --space <username>/MatSci-RAG
    uv run python scripts/deploy_to_hf.py --space <username>/MatSci-RAG --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and stream or return output."""
    print(f"--> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deploy MatSci-RAG to Hugging Face Spaces with Git LFS."
    )
    parser.add_argument(
        "--space",
        required=True,
        help="Hugging Face Space identifier in format '<username>/<spacename>' or full URL",
    )
    parser.add_argument(
        "--remote",
        default="space",
        help="Git remote name for the Hugging Face Space (default: 'space')",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to push (default: 'main')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform pre-flight checks without actually running git push",
    )

    args = parser.parse_args()

    print("=================================================================")
    print("   MatSci-RAG Hugging Face Spaces Deployment Pre-Flight Check   ")
    print("=================================================================")

    # 1. Check Git and Git LFS
    if not shutil.which("git"):
        sys.exit("Error: 'git' is not installed or not in PATH.")
    if not shutil.which("git-lfs"):
        sys.exit("Error: 'git-lfs' is not installed. Please install Git LFS first.")

    lfs_check = run_cmd(["git", "lfs", "version"])
    print(f"Git LFS verified: {lfs_check.stdout.strip()}")

    # 2. Check essential knowledge base artifacts
    required_paths = [
        "data/chroma_db",
        "data/structured_facts.jsonl",
        "data/sofc_corpus.jsonl",
        "app.py",
        "requirements.txt",
        ".gitattributes",
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"Error: Missing required deployment artifacts: {missing}")

    print("All required data artifacts and configuration files are present.")

    # 3. Format Space URL
    space_identifier = args.space.strip().rstrip("/")
    if space_identifier.startswith("http://") or space_identifier.startswith("https://"):
        space_url = space_identifier
    else:
        space_url = f"https://huggingface.co/spaces/{space_identifier}"

    print(f"Target Hugging Face Space: {space_url}")

    # 4. Configure Git Remote
    remotes_res = run_cmd(["git", "remote"], check=False)
    existing_remotes = remotes_res.stdout.splitlines()

    if args.remote in existing_remotes:
        print(f"Remote '{args.remote}' exists. Updating URL to {space_url} ...")
        run_cmd(["git", "remote", "set-url", args.remote, space_url])
    else:
        print(f"Adding remote '{args.remote}' -> {space_url} ...")
        run_cmd(["git", "remote", "add", args.remote, space_url])

    # 5. Initialize Git LFS
    run_cmd(["git", "lfs", "install"])

    if args.dry_run:
        print("\n[DRY RUN] Pre-flight checks passed successfully! No changes were pushed.")
        print("To deploy live, run without --dry-run:")
        print(f"    uv run python scripts/deploy_to_hf.py --space {args.space}")
        return

    # 6. Execute deployment
    print("\nPre-flight checks passed. Pushing code and LFS assets to Hugging Face Spaces...")
    print("Note: First-time push will upload ChromaDB vectors (~200MB) via Git LFS.\n")

    push_res = run_cmd(["git", "push", args.remote, f"{args.branch}:{args.branch}"])
    if push_res.stdout:
        print(push_res.stdout)
    if push_res.stderr:
        print(push_res.stderr)

    print("\n=================================================================")
    print("   Deployment completed successfully!                           ")
    print(f"   View your live Space at: {space_url}")
    print("   Make sure to set GROQ_API_KEY in Space Settings -> Secrets.  ")
    print("=================================================================")


if __name__ == "__main__":
    main()
