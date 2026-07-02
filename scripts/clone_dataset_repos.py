#!/usr/bin/env python3
"""Clone each dataset repo at its recorded sha.

Every sample in dataset/{full,small}/{train,test}.jsonl carries a `sha` field
(the commit the sample was extracted at) and a `blame_sha` field (the last
commit that modified the target function). All samples for a given repo share
the same `sha`. Repos are cloned with full history so that
`git show {blame_sha}:{path}` resolves during evaluation.

Usage:
    python scripts/clone_dataset_repos.py
    python scripts/clone_dataset_repos.py --deepen   # unshallow any legacy shallow clones
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def gather_repo_shas(dataset_dir: Path) -> dict[str, str]:
    """Return {repo: sha} across the full and small datasets' train/test splits.

    Raises if a repo appears with conflicting shas across splits/datasets.
    """
    repo_sha: dict[str, str] = {}
    for dataset in ("full", "small"):
        for split in ("train", "test"):
            path = dataset_dir / dataset / f"{split}.jsonl"
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
                            f"(found in {dataset}/{split}.jsonl)"
                        )
                    repo_sha[repo] = sha
    return repo_sha


def _is_shallow(clone_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=clone_dir,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "true"


def clone_at_sha(repo: str, sha: str, output_dir: Path, deepen: bool = False) -> None:
    """Clone repo with full history and reset to the given sha."""
    repo_url = f"https://github.com/{repo}.git"
    clone_dir = output_dir / repo.replace("/", "__")

    if clone_dir.exists():
        if deepen and _is_shallow(clone_dir):
            print(f"  [deepen] unshallowing existing clone for {repo}...")
            subprocess.run(
                ["git", "fetch", "--quiet", "--unshallow", "origin"],
                cwd=clone_dir,
                check=True,
            )
            print(f"  Done (full history)")
        else:
            print(f"  [skip] {clone_dir} already exists")
        return

    print(f"  Cloning {repo}@{sha[:12]} (full history)...")
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
    parser.add_argument(
        "--deepen",
        action="store_true",
        help="Unshallow any existing shallow clones left over from older dataset versions",
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
        clone_at_sha(repo, sha, args.output_dir, deepen=args.deepen)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
