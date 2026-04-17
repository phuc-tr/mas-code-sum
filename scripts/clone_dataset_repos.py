#!/usr/bin/env python3
"""Clone each dataset repo at its recorded sha.

Every sample in dataset/{python,java}/{train,test}.jsonl carries a `sha` field,
and all samples for a given repo share the same sha. This script clones each
repo and checks out that exact commit, so file paths and function locations in
the dataset resolve against the correct snapshot.

Usage:
    python scripts/clone_repos_before_test.py --output-dir repos
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path


def gather_repo_shas(dataset_dir: Path) -> dict[str, str]:
    """Return {repo: sha} across python and java train+test sets.

    Raises if a repo appears with conflicting shas across splits/languages.
    """
    repo_sha: dict[str, str] = {}
    for lang in ("python", "java"):
        for split in ("train", "test"):
            path = dataset_dir / lang / f"{split}.jsonl"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    repo = rec["repo"]
                    sha = rec["sha"]
                    existing = repo_sha.get(repo)
                    if existing is not None and existing != sha:
                        raise ValueError(
                            f"Conflicting sha for {repo}: {existing} vs {sha} "
                            f"(found in {lang}/{split}.jsonl)"
                        )
                    repo_sha[repo] = sha
    return repo_sha


def clone_at_sha(repo: str, sha: str, output_dir: Path) -> None:
    """Clone repo and reset to the given sha as a shallow depth-1 checkout."""
    repo_url = f"https://github.com/{repo}.git"
    clone_dir = output_dir / repo.replace("/", "__")

    if clone_dir.exists():
        print(f"  [skip] {clone_dir} already exists")
        return

    # Init an empty repo and fetch only the target commit (depth 1).
    print(f"  Fetching {repo}@{sha[:12]} (shallow)...")
    clone_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=clone_dir, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", repo_url],
        cwd=clone_dir,
        check=True,
    )
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "--depth", "1", "origin", sha],
            cwd=clone_dir,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Some servers disallow fetching arbitrary shas by default; fall back
        # to a full clone and then reset.
        print(f"  [info] shallow fetch refused, falling back to full clone")
        shutil.rmtree(clone_dir)
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(clone_dir)],
            check=True,
        )

    subprocess.run(
        ["git", "reset", "--quiet", "--hard", sha],
        cwd=clone_dir,
        check=True,
    )
    print(f"  Done: {sha[:12]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="Path to dataset/ directory (default: ../dataset relative to script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset" / "repos",
        help="Directory to clone repos into (default: ../dataset/repos relative to script)",
    )
    args = parser.parse_args()

    repo_sha = gather_repo_shas(args.dataset_dir)
    if not repo_sha:
        print("No repos found in dataset jsonl files", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(repo_sha)} repos:\n")
    for repo, sha in sorted(repo_sha.items()):
        print(f"  {repo}: {sha}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for repo, sha in sorted(repo_sha.items()):
        print(f"[{repo}]")
        clone_at_sha(repo, sha, args.output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
