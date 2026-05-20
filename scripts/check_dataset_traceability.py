#!/usr/bin/env python3
"""Check that every dataset sample can be traced to its position in the cloned repo.

For each record in dataset/{python,java}/{train,test}.jsonl, verify:
  1. The repo is cloned under dataset/repos/<owner>__<name>/.
  2. The file at `path` exists in that clone.
  3. The `original_string` appears verbatim in that file.

Reports per-split counts of OK / missing-file / snippet-not-found, and prints
a small sample of failures for inspection.

Usage:
    python scripts/check_dataset_traceability.py
    python scripts/check_dataset_traceability.py --lang python --split test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def check_split(
    jsonl_path: Path,
    repos_dir: Path,
    max_failure_examples: int = 5,
) -> dict:
    """Check every sample in one jsonl file; return a summary dict."""
    counts = Counter()
    failures_by_reason: dict[str, list[dict]] = {
        "missing_repo": [],
        "missing_file": [],
        "snippet_not_found": [],
    }
    # Cache file contents so we read each source file at most once per split.
    # Read as bytes to preserve original line endings; universal-newline
    # translation in read_text() would convert CRLF->LF and cause spurious
    # mismatches against snippets that kept their original CRLF.
    file_cache: dict[Path, bytes | None] = {}

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            counts["total"] += 1

            repo_dir = repos_dir / rec["repo"].replace("/", "__")
            if not repo_dir.is_dir():
                counts["missing_repo"] += 1
                if len(failures_by_reason["missing_repo"]) < max_failure_examples:
                    failures_by_reason["missing_repo"].append(
                        {"id": rec["id"], "repo": rec["repo"]}
                    )
                continue

            file_path = repo_dir / rec["path"]
            if file_path not in file_cache:
                try:
                    file_cache[file_path] = file_path.read_bytes()
                except (FileNotFoundError, IsADirectoryError):
                    file_cache[file_path] = None

            contents = file_cache[file_path]
            if contents is None:
                counts["missing_file"] += 1
                if len(failures_by_reason["missing_file"]) < max_failure_examples:
                    failures_by_reason["missing_file"].append(
                        {
                            "id": rec["id"],
                            "repo": rec["repo"],
                            "path": rec["path"],
                        }
                    )
                continue

            if rec["original_string"].encode("utf-8") in contents:
                counts["ok"] += 1
            else:
                counts["snippet_not_found"] += 1
                if (
                    len(failures_by_reason["snippet_not_found"])
                    < max_failure_examples
                ):
                    failures_by_reason["snippet_not_found"].append(
                        {
                            "id": rec["id"],
                            "repo": rec["repo"],
                            "path": rec["path"],
                            "func_name": rec["func_name"],
                        }
                    )

    return {"counts": counts, "failures": failures_by_reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/repos",
    )
    parser.add_argument("--lang", choices=["python", "java"], default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    args = parser.parse_args()

    repos_dir = args.repos_dir or (args.dataset_dir / "repos")
    if not repos_dir.is_dir():
        print(f"repos dir not found: {repos_dir}", file=sys.stderr)
        sys.exit(1)

    langs = [args.lang] if args.lang else ["python", "java"]
    splits = [args.split] if args.split else ["train", "test"]

    any_failures = False
    for lang in langs:
        for split in splits:
            jsonl = args.dataset_dir / lang / f"{split}.jsonl"
            if not jsonl.exists():
                print(f"[skip] {jsonl} (not found)")
                continue

            result = check_split(jsonl, repos_dir)
            c = result["counts"]
            print(f"\n=== {lang}/{split}.jsonl ===")
            print(f"  total               : {c['total']}")
            print(f"  ok                  : {c['ok']}")
            print(f"  missing_repo        : {c['missing_repo']}")
            print(f"  missing_file        : {c['missing_file']}")
            print(f"  snippet_not_found   : {c['snippet_not_found']}")

            for reason, examples in result["failures"].items():
                if examples:
                    any_failures = True
                    print(f"\n  examples [{reason}]:")
                    for ex in examples:
                        print(f"    - {ex}")

    sys.exit(1 if any_failures else 0)


if __name__ == "__main__":
    main()
