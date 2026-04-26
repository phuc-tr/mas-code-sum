"""Extract file-level context (module docstring, enclosing class, imports) for Python files.

Given a dataset row's repo and path, loads the source file from `dataset/repos/`
and pulls signals that help the LLM summarize a target function.

Language-specific signals
-------------------------
Python:
  - module_doc: top-of-file docstring (stripped)
  - class_name / class_doc: enclosing class name and docstring (if method)
  - imports: top-level import statements

Java (delegated to file_context_java.py):
  - class_name / class_doc: enclosing class name and Javadoc
  - outer_class_name / outer_class_doc: top-level class name and Javadoc,
    only set when the method lives in a nested/inner class (differs from enclosing)
  - imports: top-level import statements
  - module_doc is always None (Java has no file-level docstring)

There is intentionally no forced common mapping between languages.
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
    language: str  # "python" | "java"
    # Python only: top-of-file docstring. Always None for Java.
    module_doc: str | None
    # Direct enclosing class of the target function (Python and Java).
    class_name: str | None
    class_doc: str | None
    # Java only: the top-level class, set only when the method lives in a
    # nested/inner class (i.e. outer_class_name != class_name). Always None for Python.
    outer_class_name: str | None
    outer_class_doc: str | None
    imports: list[str]

    def is_empty(self) -> bool:
        if self.language == "python":
            return not (self.module_doc or self.class_name or self.imports)
        else:
            return not (self.class_name or self.outer_class_name or self.imports)


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
        return FileContext(language="python", module_doc=None, class_name=None, class_doc=None, outer_class_name=None, outer_class_doc=None, imports=[])
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
        language="python",
        module_doc=module_doc.strip() if module_doc else None,
        class_name=class_name,
        class_doc=class_doc.strip() if class_doc else None,
        outer_class_name=None,
        outer_class_doc=None,
        imports=imports,
    )


def render_file_context(
    ctx: FileContext,
    max_module_doc_chars: int = 400,
    max_class_doc_chars: int = 300,
) -> str:
    """Render a FileContext as a compact block for prompt inclusion. Empty -> ''.

    Labels are language-specific:
      Python: "Module docstring:", "Enclosing class:"
      Java:   "Top-level class:" (outer, nested case only), "Class:" / "Enclosing class:"
    """
    parts: list[str] = []

    def _truncate(text: str, max_chars: int) -> str:
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    if ctx.language == "python":
        if ctx.module_doc:
            parts.append(f"Module docstring: {_truncate(ctx.module_doc, max_module_doc_chars)}")
        if ctx.class_name:
            line = f"Enclosing class: {ctx.class_name}"
            if ctx.class_doc:
                line += f" — {_truncate(ctx.class_doc, max_class_doc_chars)}"
            parts.append(line)

    else:  # java
        # When the method is in a nested class, show the outer (top-level) class first.
        if ctx.outer_class_name:
            line = f"Top-level class: {ctx.outer_class_name}"
            if ctx.outer_class_doc:
                line += f" — {_truncate(ctx.outer_class_doc, max_class_doc_chars)}"
            parts.append(line)
        if ctx.class_name:
            label = "Enclosing class" if ctx.outer_class_name else "Class"
            line = f"{label}: {ctx.class_name}"
            if ctx.class_doc:
                line += f" — {_truncate(ctx.class_doc, max_class_doc_chars)}"
            parts.append(line)

    if ctx.imports:
        parts.append("Imports:\n" + "\n".join(ctx.imports))
    return "\n".join(parts)


def _build_outline_python(tree: ast.Module, src: str, exclude_name: str | None) -> list[str]:
    """Walk AST and collect signature + first-line docstring for each function/class.

    Skips any node whose name matches exclude_name (bare, without ClassName. prefix).
    Returns a list of formatted blocks.
    """
    lines = src.splitlines()
    blocks: list[str] = []

    def _first_doc_line(node: ast.AST) -> str | None:
        doc = ast.get_docstring(node)
        if not doc:
            return None
        return doc.split("\n")[0].strip()

    def _sig_line(node: ast.AST) -> str:
        return lines[node.lineno - 1].rstrip()  # type: ignore[attr-defined]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == exclude_name:
                continue
            sig = _sig_line(node)
            doc = _first_doc_line(node)
            blocks.append(f"{sig}\n    \"{doc}\"" if doc else sig)
        elif isinstance(node, ast.ClassDef):
            if node.name == exclude_name:
                continue
            class_sig = _sig_line(node)
            class_doc = _first_doc_line(node)
            class_block = f"{class_sig}\n    \"{class_doc}\"" if class_doc else class_sig
            method_blocks: list[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == exclude_name:
                        continue
                    sig = "    " + _sig_line(item).lstrip()
                    doc = _first_doc_line(item)
                    method_blocks.append(f"{sig}\n        \"{doc}\"" if doc else sig)
            if method_blocks:
                blocks.append(class_block + "\n\n" + "\n\n".join(method_blocks))
            else:
                blocks.append(class_block)

    return blocks


def extract_file_outline(
    repo: str,
    path: str,
    exclude_func_name: str | None = None,
    language: str = "python",
    max_chars: int = 4000,
    cutoff_timestamp: str | None = None,
) -> str:
    """Return a compact outline of the file suitable for prompt inclusion.

    For Python: extracts function/class signatures and their first-line docstrings
    via AST, skipping the target function. This gives the LLM naming conventions
    and documentation style without flooding it with implementation details.

    For Java / other: falls back to truncated raw source.

    `exclude_func_name` may be a bare name or 'ClassName.method' — only the bare
    name is used for matching, so either form works.
    """
    bare_exclude: str | None = None
    if exclude_func_name:
        bare_exclude = exclude_func_name.split(".")[-1]

    if language == "python":
        parsed = _parse(repo, path)
        if parsed is not None:
            tree, src = parsed
            blocks = _build_outline_python(tree, src, bare_exclude)
            outline = "\n\n".join(blocks)
            if len(outline) > max_chars:
                outline = outline[:max_chars].rstrip() + "\n... [truncated]"
            return outline
        # Parse failed (e.g. Python 2 syntax) — return nothing rather than
        # risk leaking the target function's docstring via raw source.
        return ""

    if language == "java":
        from .file_context_java import _build_outline_java, _parse as _java_parse
        root, src = _java_parse(repo, path)
        blocks = _build_outline_java(root, src, bare_exclude)
        outline = "\n\n".join(blocks)
        if len(outline) > max_chars:
            outline = outline[:max_chars].rstrip() + "\n... [truncated]"
        return outline

    # Unsupported language — return nothing rather than risk leaking docstrings.
    return ""
