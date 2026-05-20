"""
Extract (code, docstring) pairs from raw source files in dataset/repos/.

Outputs dataset/few_shots/python.jsonl and dataset/few_shots/java.jsonl,
using the same schema as the existing train/test splits.  Functions that
already appear in any existing split are excluded to prevent leakage.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langdetect import LangDetectException, detect
from tree_sitter import Language, Node, Parser
import tree_sitter_python as ts_python
import tree_sitter_java as ts_java

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
REPOS_DIR = DATASET_DIR / "repos"
OUTPUT_DIR = DATASET_DIR / "few_shots"

LANG_REPOS: dict[str, list[str]] = {
    "python": [
        "apache__airflow",
        "vaexio__vaex",
        "h2oai__h2o-3",
        "Qiskit__qiskit-terra",
        "PyCQA__pylint",
    ],
    "java": [
        "spring-projects__spring-security",
        "real-logic__aeron",
        "orientechnologies__orientdb",
        "oblac__jodd",
        "wildfly__wildfly",
        "h2oai__h2o-3",
    ],
}

# ── tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Split into word tokens and single punctuation characters."""
    return re.findall(r"\w+|[^\w\s]", text)


# ── deduplication ─────────────────────────────────────────────────────────────

def load_excluded_keys() -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for lang in ("python", "java"):
        for split in ("train", "test"):
            p = DATASET_DIR / lang / f"{split}.jsonl"
            if p.exists():
                with open(p) as f:
                    for line in f:
                        d = json.loads(line)
                        keys.add((d["path"], d["func_name"]))
    return frozenset(keys)


def _is_test_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    name = parts[-1] if parts else ""
    return (
        "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("Test.java")
        or name.endswith("Tests.java")
    )


# ── docstring cleaning ────────────────────────────────────────────────────────

def _first_paragraph(lines: list[str]) -> list[str]:
    """Return lines up to (not including) the first blank line after content starts."""
    result = []
    started = False
    for ln in lines:
        if not ln:
            if started:
                break  # first blank line after content — end of paragraph
        else:
            started = True
            result.append(ln)
    return result


def _clean_python_docstring(raw: str) -> str:
    text = raw.strip()
    for q in ('"""', "'''"):
        if text.startswith(q) and text.endswith(q) and len(text) >= 6:
            text = text[3:-3]
            break
    else:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1]
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(_first_paragraph(lines)).strip()


_JAVADOC_TAG = re.compile(r"^\s*@\w+")


def _clean_javadoc(raw: str) -> str:
    text = raw.strip()
    if text.startswith("/**"):
        text = text[3:]
    if text.endswith("*/"):
        text = text[:-2]
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*"):
            stripped = stripped[1:].strip()
        if _JAVADOC_TAG.match(stripped):
            break
        lines.append(stripped)
    return " ".join(_first_paragraph(lines)).strip()


# ── name filters ─────────────────────────────────────────────────────────────

# Python: all dunder methods (covers __init__, __str__, __repr__, etc.)
_PYTHON_DUNDER_RE = re.compile(r"^__\w+__$")

# Java: standard Object/Comparable override methods
_JAVA_STANDARD_METHODS = frozenset(
    {"toString", "equals", "hashCode", "compareTo", "clone"}
)


def _skip_by_name(bare_name: str, language: str) -> bool:
    """Return True if the method name should be excluded."""
    if "test" in bare_name.lower():
        return True
    if language == "python":
        return bool(_PYTHON_DUNDER_RE.match(bare_name))
    else:
        return bare_name in _JAVA_STANDARD_METHODS


# ── quality filters ───────────────────────────────────────────────────────────

_SPECIAL_TOKEN_RE = re.compile(
    r"<[a-zA-Z][^>]*>"        # HTML/XML tags  e.g. <img ...> <a href=...>
    r"|https?://\S+"           # URLs           e.g. https://example.com
    r"|www\.\S+\.\S+"          # bare www URLs  e.g. www.example.com/path
)


def _has_special_tokens(text: str) -> bool:
    return bool(_SPECIAL_TOKEN_RE.search(text))


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _python_ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _java_ast_ok(node: Node) -> bool:
    """Return False if the tree-sitter node or any descendant is an ERROR node."""
    if node.type == "ERROR":
        return False
    return all(_java_ast_ok(child) for child in node.children)


# ── stats collector ───────────────────────────────────────────────────────────

@dataclass
class Stats:
    extracted: int = 0
    skipped_dedup: int = 0
    skipped_name: int = 0
    skipped_short: int = 0
    skipped_long: int = 0
    skipped_special_tokens: int = 0
    skipped_non_english: int = 0
    skipped_unparseable: int = 0
    skipped_test: int = 0
    skipped_parse_error: int = 0
    per_repo: dict[str, int] = field(default_factory=dict)


# ── Python extractor ──────────────────────────────────────────────────────────

class PythonExtractor:
    def __init__(self) -> None:
        self._parser = Parser(Language(ts_python.language()))

    def _find_docstring_node(self, func_node: Node) -> Node | None:
        """Return the string node inside the docstring expression, or None."""
        block = next((c for c in func_node.children if c.type == "block"), None)
        if block is None:
            return None
        first = next(
            (c for c in block.named_children if c.type not in ("comment",)),
            None,
        )
        if first is None or first.type != "expression_statement":
            return None
        child = first.named_children[0] if first.named_children else None
        if child is None:
            return None
        if child.type in ("string", "concatenated_string"):
            return child
        return None

    def _is_nested(self, node: Node) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.type == "function_definition":
                return True
            parent = parent.parent
        return False

    def _class_name(self, node: Node) -> str | None:
        parent = node.parent
        while parent is not None:
            if parent.type == "class_definition":
                for c in parent.children:
                    if c.type == "identifier":
                        return c.text.decode("utf-8", errors="replace")
            parent = parent.parent
        return None

    def _func_name(self, node: Node) -> str:
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        fname = name_node.text.decode("utf-8", errors="replace") if name_node else ""
        cls = self._class_name(node)
        return f"{cls}.{fname}" if cls else fname

    def _walk(self, node: Node) -> Iterator[Node]:
        if node.type == "function_definition":
            yield node
        for child in node.children:
            yield from self._walk(child)

    def extract_from_file(
        self,
        src_bytes: bytes,
        rel_path: str,
        repo: str,
        excluded: frozenset[tuple[str, str]],
        seen_in_run: set[tuple[str, str]],
        next_id: list[int],
        stats: Stats,
    ) -> list[dict]:
        try:
            tree = self._parser.parse(src_bytes)
        except Exception:
            stats.skipped_parse_error += 1
            return []

        records: list[dict] = []
        for func_node in self._walk(tree.root_node):
            if self._is_nested(func_node):
                continue

            doc_node = self._find_docstring_node(func_node)
            if doc_node is None:
                continue

            raw_doc = doc_node.text.decode("utf-8", errors="replace")
            docstring = _clean_python_docstring(raw_doc)
            docstring_tokens = tokenize(docstring)

            if len(docstring_tokens) < 3:
                stats.skipped_short += 1
                continue
            if len(docstring_tokens) > 256:
                stats.skipped_long += 1
                continue
            if _has_special_tokens(docstring):
                stats.skipped_special_tokens += 1
                continue
            if not _is_english(docstring):
                stats.skipped_non_english += 1
                continue

            func_name = self._func_name(func_node)
            bare = func_name.split(".")[-1]
            if _skip_by_name(bare, "python"):
                stats.skipped_name += 1
                continue

            key = (rel_path, func_name)
            if key in excluded or key in seen_in_run:
                stats.skipped_dedup += 1
                continue

            code = func_node.text.decode("utf-8", errors="replace")
            if not _python_ast_ok(code):
                stats.skipped_unparseable += 1
                continue

            # code_tokens excludes the docstring (slice out the docstring bytes)
            doc_start = doc_node.start_byte - func_node.start_byte
            doc_end = doc_node.end_byte - func_node.start_byte
            func_bytes = func_node.text
            code_no_doc = func_bytes[:doc_start] + func_bytes[doc_end:]
            code_tokens = tokenize(code_no_doc.decode("utf-8", errors="replace"))

            seen_in_run.add(key)
            record = {
                "id": next_id[0],
                "repo": repo,
                "path": rel_path,
                "func_name": func_name,
                "language": "python",
                "original_string": code,
                "code": code,
                "code_tokens": code_tokens,
                "docstring": docstring,
                "docstring_tokens": docstring_tokens,
                "sha": "",
                "url": "",
                "latest_blame_timestamp": None,
            }
            next_id[0] += 1
            stats.extracted += 1
            stats.per_repo[repo] = stats.per_repo.get(repo, 0) + 1
            records.append(record)

        return records


# ── Java extractor ────────────────────────────────────────────────────────────

class JavaExtractor:
    def __init__(self) -> None:
        self._parser = Parser(Language(ts_java.language()))

    def _javadoc_node(self, method_node: Node) -> Node | None:
        sibling = method_node.prev_named_sibling
        if sibling is None:
            return None
        if sibling.type != "block_comment":
            return None
        text = sibling.text.decode("utf-8", errors="replace").strip()
        if not text.startswith("/**"):
            return None
        return sibling

    def _class_name(self, node: Node) -> str | None:
        parent = node.parent
        while parent is not None:
            if parent.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):
                for c in parent.children:
                    if c.type == "identifier":
                        return c.text.decode("utf-8", errors="replace")
            parent = parent.parent
        return None

    def _func_name(self, node: Node) -> str:
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        fname = name_node.text.decode("utf-8", errors="replace") if name_node else ""
        cls = self._class_name(node)
        return f"{cls}.{fname}" if cls else fname

    def _walk(self, node: Node) -> Iterator[Node]:
        if node.type == "method_declaration":
            yield node
        for child in node.children:
            yield from self._walk(child)

    def extract_from_file(
        self,
        src_bytes: bytes,
        rel_path: str,
        repo: str,
        excluded: frozenset[tuple[str, str]],
        seen_in_run: set[tuple[str, str]],
        next_id: list[int],
        stats: Stats,
    ) -> list[dict]:
        try:
            tree = self._parser.parse(src_bytes)
        except Exception:
            stats.skipped_parse_error += 1
            return []

        records: list[dict] = []
        for method_node in self._walk(tree.root_node):
            # Skip abstract/interface methods (no body block)
            if not any(c.type == "block" for c in method_node.children):
                continue

            javadoc = self._javadoc_node(method_node)
            if javadoc is None:
                continue

            raw_doc = javadoc.text.decode("utf-8", errors="replace")
            docstring = _clean_javadoc(raw_doc)
            docstring_tokens = tokenize(docstring)

            if len(docstring_tokens) < 3:
                stats.skipped_short += 1
                continue
            if len(docstring_tokens) > 256:
                stats.skipped_long += 1
                continue
            if _has_special_tokens(docstring):
                stats.skipped_special_tokens += 1
                continue
            if not _is_english(docstring):
                stats.skipped_non_english += 1
                continue

            func_name = self._func_name(method_node)
            bare = func_name.split(".")[-1]
            if _skip_by_name(bare, "java"):
                stats.skipped_name += 1
                continue

            key = (rel_path, func_name)
            if key in excluded or key in seen_in_run:
                stats.skipped_dedup += 1
                continue

            if not _java_ast_ok(method_node):
                stats.skipped_unparseable += 1
                continue

            code = method_node.text.decode("utf-8", errors="replace")
            code_tokens = tokenize(code)

            seen_in_run.add(key)
            record = {
                "id": next_id[0],
                "repo": repo,
                "path": rel_path,
                "func_name": func_name,
                "language": "java",
                "original_string": code,
                "code": code,
                "code_tokens": code_tokens,
                "docstring": docstring,
                "docstring_tokens": docstring_tokens,
                "sha": "",
                "url": "",
                "latest_blame_timestamp": None,
            }
            next_id[0] += 1
            stats.extracted += 1
            stats.per_repo[repo] = stats.per_repo.get(repo, 0) + 1
            records.append(record)

        return records


# ── main ──────────────────────────────────────────────────────────────────────

def extract_language(
    language: str,
    excluded: frozenset[tuple[str, str]],
    next_id: list[int],
) -> tuple[list[dict], Stats]:
    ext = ".py" if language == "python" else ".java"
    extractor: PythonExtractor | JavaExtractor = (
        PythonExtractor() if language == "python" else JavaExtractor()
    )
    stats = Stats()
    records: list[dict] = []
    seen_in_run: set[tuple[str, str]] = set()

    for repo_dir_name in LANG_REPOS[language]:
        repo_path = REPOS_DIR / repo_dir_name
        if not repo_path.exists():
            print(f"  [warn] repo not found: {repo_path}", file=sys.stderr)
            continue
        repo_name = repo_dir_name.replace("__", "/")

        src_files = list(repo_path.rglob(f"*{ext}"))
        for src_file in src_files:
            rel_path = str(src_file.relative_to(repo_path))
            if _is_test_path(rel_path):
                stats.skipped_test += 1
                continue
            try:
                src_bytes = src_file.read_bytes()
            except OSError:
                continue
            new_records = extractor.extract_from_file(
                src_bytes, rel_path, repo_name, excluded, seen_in_run, next_id, stats
            )
            records.extend(new_records)

    return records, stats


def _print_stats(language: str, stats: Stats) -> None:
    print(f"\n=== {language.upper()} ===")
    print(f"  Extracted:               {stats.extracted:>7,}")
    print(f"  Skipped (dedup):         {stats.skipped_dedup:>7,}")
    print(f"  Skipped (name):          {stats.skipped_name:>7,}")
    print(f"  Skipped (short <3):      {stats.skipped_short:>7,}")
    print(f"  Skipped (long >256):     {stats.skipped_long:>7,}")
    print(f"  Skipped (special tokens):{stats.skipped_special_tokens:>7,}")
    print(f"  Skipped (non-English):   {stats.skipped_non_english:>7,}")
    print(f"  Skipped (unparseable):   {stats.skipped_unparseable:>7,}")
    print(f"  Skipped (test files):    {stats.skipped_test:>7,}")
    print(f"  Skipped (parse error):   {stats.skipped_parse_error:>7,}")
    print("  Per-repo:")
    for repo, count in sorted(stats.per_repo.items(), key=lambda x: -x[1]):
        print(f"    {repo:<45} {count:>6,}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = load_excluded_keys()
    next_id = [3919]  # one past max existing ID

    for language in ("python", "java"):
        print(f"Extracting {language}...", flush=True)
        records, stats = extract_language(language, excluded, next_id)

        out_path = OUTPUT_DIR / f"{language}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        _print_stats(language, stats)
        print(f"  Written to: {out_path}")


if __name__ == "__main__":
    main()
