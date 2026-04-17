"""Extract file-level context (module docstring, enclosing class, imports) for Python files.

Given a dataset row's repo and path, loads the source file from `dataset/repos/`
and pulls signals that help the LLM summarize a target function:

- module_doc: top-of-file docstring (stripped)
- class_name / class_doc: the enclosing class's name and docstring, if the
  target function is a method
- imports: top-level `import` / `from ... import ...` statements, as source text

Python only for v1. Callers should gate on language before invoking.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DEF_RE = re.compile(r"\b(?:async\s+)?def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(")

REPOS_ROOT = Path(__file__).parents[3] / "dataset" / "repos"


@dataclass
class FileContext:
    module_doc: str | None
    class_name: str | None
    class_doc: str | None
    imports: list[str]

    def is_empty(self) -> bool:
        return not (self.module_doc or self.class_name or self.imports)


def _repo_dir(repo: str) -> Path:
    """Map 'apache/airflow' -> dataset/repos/apache__airflow."""
    return REPOS_ROOT / repo.replace("/", "__")


@lru_cache(maxsize=512)
def _load_source(repo: str, path: str) -> str:
    file_path = _repo_dir(repo) / path
    # Let failures surface — per the no-silent-failures rule.
    return file_path.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=512)
def _parse(repo: str, path: str) -> tuple[ast.Module, str] | None:
    """Parse the source file. Returns None (with one-time warning) for Py2/unparseable files.

    Dataset contains legacy Python 2 source (e.g., `print "..."`), which Py3's ast
    rejects. This is an external-data boundary — we surface the issue with a warning
    rather than crashing the run.
    """
    src = _load_source(repo, path)
    try:
        return ast.parse(src), src
    except SyntaxError as e:
        warnings.warn(
            f"file_context: cannot parse {repo}/{path} (line {e.lineno}: {e.msg}); "
            f"falling back to repo-level context for this file.",
            stacklevel=3,
        )
        return None


def _extract_func_name_from_code(code: str) -> str | None:
    """Find the first `def NAME` in a code snippet.

    Tries AST first; falls back to regex because dataset code is often the
    single-line tokenized form (no indentation), which ast.parse rejects.
    """
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        pass
    m = _DEF_RE.search(code)
    return m.group(1) if m else None


def _find_enclosing_class(tree: ast.Module, target_name: str) -> ast.ClassDef | None:
    """Return the ClassDef directly containing a function named target_name, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target_name:
                    return node
    return None


def _collect_imports(tree: ast.Module, src: str, max_imports: int) -> list[str]:
    """Return source text of top-level import statements (capped by max_imports; 0 = none)."""
    if max_imports <= 0:
        return []
    lines = src.splitlines()
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # ast lineno is 1-based, end_lineno present in py3.8+
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno)
            stmt = "\n".join(lines[start:end])
            out.append(stmt)
            if len(out) >= max_imports:
                break
    return out


def extract_file_context(
    repo: str,
    path: str,
    func_name: str | None = None,
    code: str | None = None,
    max_imports: int = 25,
    language: str = "python",
) -> FileContext:
    """Parse the repo file at `path` and extract module/class/imports context.

    `func_name` may be a bare name ('get_conn') or 'ClassName.method'; either works.
    If not given, we try to parse it from `code`.
    `max_imports` caps how many import statements are collected (0 disables).
    `language` selects the extractor backend ('python' or 'java').
    """
    if language == "java":
        # Lazy import so python-only runs don't pay tree-sitter startup.
        from .file_context_java import extract_java_file_context
        return extract_java_file_context(
            repo, path, func_name=func_name, code=code, max_imports=max_imports
        )
    if language != "python":
        raise ValueError(f"Unsupported language for file context: {language!r}")

    parsed = _parse(repo, path)
    if parsed is None:
        return FileContext(module_doc=None, class_name=None, class_doc=None, imports=[])
    tree, src = parsed

    module_doc = ast.get_docstring(tree)

    # Normalize func name: accept 'Class.method' or bare 'method'.
    if func_name is None and code is not None:
        func_name = _extract_func_name_from_code(code)

    class_name: str | None = None
    class_doc: str | None = None
    if func_name:
        bare = func_name.split(".")[-1]
        cls = _find_enclosing_class(tree, bare)
        if cls is not None:
            class_name = cls.name
            class_doc = ast.get_docstring(cls)

    imports = _collect_imports(tree, src, max_imports=max_imports)

    return FileContext(
        module_doc=module_doc.strip() if module_doc else None,
        class_name=class_name,
        class_doc=class_doc.strip() if class_doc else None,
        imports=imports,
    )


def render_file_context(
    ctx: FileContext,
    max_module_doc_chars: int = 400,
    max_class_doc_chars: int = 300,
) -> str:
    """Render a FileContext as a compact block for prompt inclusion. Empty -> ''."""
    parts: list[str] = []
    if ctx.module_doc:
        doc = ctx.module_doc
        if len(doc) > max_module_doc_chars:
            doc = doc[:max_module_doc_chars].rstrip() + "..."
        parts.append(f"Module docstring: {doc}")
    if ctx.class_name:
        line = f"Enclosing class: {ctx.class_name}"
        # Skip class_doc if it matches module_doc (common in Java, where the
        # top-level class's Javadoc is the "module doc" analog).
        if ctx.class_doc and ctx.class_doc.strip() != (ctx.module_doc or "").strip():
            cdoc = ctx.class_doc
            if len(cdoc) > max_class_doc_chars:
                cdoc = cdoc[:max_class_doc_chars].rstrip() + "..."
            line += f" — {cdoc}"
        parts.append(line)
    if ctx.imports:
        parts.append("Imports:\n" + "\n".join(ctx.imports))
    return "\n".join(parts)
