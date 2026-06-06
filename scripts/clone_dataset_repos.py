#!/usr/bin/env python3
"""Clone each dataset repo at its recorded sha.

Every sample in dataset/{python,java}/{train,test}.jsonl carries a `sha` field,
and all samples for a given repo share the same sha. This script clones each
repo and checks out that exact commit, so file paths and function locations in
the dataset resolve against the correct snapshot.

For v3 data each sample also has a `blame_sha` field (the last commit that
modified the target function). Repos from v3 are cloned with full history so
that `git show {blame_sha}:{path}` works during evaluation.

Usage:
    python scripts/clone_dataset_repos.py
    python scripts/clone_dataset_repos.py --deepen   # unshallow existing v3 repos
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


def gather_v3_repos(dataset_dir: Path) -> dict[str, str]:
    """Return {repo: sha} for all repos in dataset/v3/.

    V3 repos need full history so blame_sha commits are reachable.
    Raises on conflicting shas within v3.
    """
    repo_sha: dict[str, str] = {}
    for split in ("train", "test"):
        path = dataset_dir / "v3" / f"{split}.jsonl"
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
                        f"Conflicting sha for {repo} in v3: {existing} vs {sha}"
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


def clone_at_sha(repo: str, sha: str, output_dir: Path, full_history: bool = False) -> None:
    """Clone repo and reset to the given sha.

    When full_history=True, fetches the complete history instead of depth-1,
    which is required when blame_sha commits (ancestors of sha) must be
    accessible via `git show`.
    """
    repo_url = f"https://github.com/{repo}.git"
    clone_dir = output_dir / repo.replace("/", "__")

    if clone_dir.exists():
        if full_history and _is_shallow(clone_dir):
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

    if full_history:
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
        return

    # Shallow depth-1 path (v1/v2 repos that don't need blame history).
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
    v3_repos = gather_v3_repos(args.dataset_dir)

    # Merge v3 repos in; they may overlap with v1/v2 (possibly at a different sha).
    for repo, sha in v3_repos.items():
        existing = repo_sha.get(repo)
        if existing is not None and existing != sha:
            print(
                f"  [note] {repo} sha differs between v1/v2 ({existing[:12]}) "
                f"and v3 ({sha[:12]}); v3 sha takes precedence"
            )
        repo_sha[repo] = sha

    if not repo_sha:
        print("No repos found in dataset jsonl files", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(repo_sha)} repos ({len(v3_repos)} require full history for v3 blame_sha):\n")
    for repo, sha in sorted(repo_sha.items()):
        marker = " [full history]" if repo in v3_repos else ""
        print(f"  {repo}: {sha}{marker}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for repo, sha in sorted(repo_sha.items()):
        print(f"[{repo}]")
        clone_at_sha(repo, sha, args.output_dir, full_history=(repo in v3_repos))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
