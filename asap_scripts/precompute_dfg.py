"""
Pre-compute DFG context for all dataset samples and save as URL-keyed JSON.

Uses modern tree-sitter (v0.21+) with the DFG/utils logic from the CodeXGLUE
parser package (asap_scripts/scripts/parser/).

Usage (from repo root):
    python scripts/precompute_dfg.py [--languages python java] [--splits train test]

Output:
    dataset/{language}/{split}_dfg.json  — {url: dfg_text} mapping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = REPO_ROOT / "asap_scripts" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from parser.utils import (
        index_to_code_token,
        remove_comments_and_docstrings,
        tree_to_token_index,
    )
    from parser.DFG import DFG_python, DFG_java
except ImportError as e:
    sys.exit(
        f"Cannot import parser module: {e}\n"
        f"Expected files at: {SCRIPTS_DIR / 'parser'}/"
    )

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser as TSParser

# ── parser setup (modern tree-sitter, no .so needed) ─────────────────────────

_PARSERS: dict[str, tuple[TSParser, object]] = {}

def _get_parser(language: str) -> tuple[TSParser, object]:
    if language not in _PARSERS:
        if language == "python":
            lang = Language(tspython.language())
            dfg_fn = DFG_python
        elif language == "java":
            lang = Language(tsjava.language())
            dfg_fn = DFG_java
        else:
            raise ValueError(f"Unsupported language: {language}")
        _PARSERS[language] = (TSParser(lang), dfg_fn)
    return _PARSERS[language]


# ── DFG extraction ────────────────────────────────────────────────────────────

def extract_dataflow(code: str, language: str) -> str:
    """Return an ASAP-format DFG context string for *code*, or '' on failure."""
    parser, dfg_fn = _get_parser(language)

    try:
        clean = remove_comments_and_docstrings(code, language)
    except Exception:
        clean = code

    try:
        tree = parser.parse(bytes(clean, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = clean.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]

        index_to_code_map = {}
        for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code_map[index] = (idx, tok)

        try:
            dfg, _ = dfg_fn(root_node, index_to_code_map, {})
        except Exception:
            dfg = []

        dfg = sorted(dfg, key=lambda x: x[1])

        # Keep only nodes that have edges
        active = set()
        for d in dfg:
            if len(d[-1]) != 0:
                active.add(d[1])
            for x in d[-1]:
                active.add(x)
        dfg = [d for d in dfg if d[1] in active]
    except Exception:
        dfg = []

    if not dfg:
        return ""

    lines = ["Please find the dataflow of the function. We present the source and list of target indices."]
    seen: dict[str, list[int]] = {}
    count = 0
    for w in dfg:
        if len(w[4]) == 0:
            continue
        key = f"{w[0]}-{w[4]}"
        if key not in seen:
            seen[key] = []
        seen[key].append(w[1])
        count += 1
        if count >= 30:
            break

    if not seen:
        return ""

    for key, targets in seen.items():
        lines.append(f"{key} {targets}")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def precompute(languages: list[str], splits: list[str]) -> None:
    dataset_dir = REPO_ROOT / "dataset"

    for language in languages:
        lang_dir = dataset_dir / language
        if not lang_dir.exists():
            print(f"  Skipping {language}: {lang_dir} not found")
            continue

        for split in splits:
            jsonl_path = lang_dir / f"{split}.jsonl"
            if not jsonl_path.exists():
                print(f"  Skipping {language}/{split}: not found")
                continue

            out_path = lang_dir / f"{split}_dfg.json"
            print(f"Processing {language}/{split}.jsonl → {out_path.name}")

            samples = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))

            lookup: dict[str, str] = {}
            for i, sample in enumerate(samples):
                if i % 1000 == 0:
                    print(f"  {i}/{len(samples)}...")
                url = sample.get("url", "")
                code = sample.get("original_string", sample.get("code", ""))
                if not url or not code:
                    continue
                dfg_text = extract_dataflow(code, language)
                if dfg_text:
                    lookup[url] = dfg_text

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(lookup, f, ensure_ascii=False)

            print(f"  Done: {len(lookup)}/{len(samples)} samples had non-empty DFG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["python", "java"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    precompute(args.languages, args.splits)


if __name__ == "__main__":
    main()
