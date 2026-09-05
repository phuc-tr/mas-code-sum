"""Extract file-level context (module docstring, enclosing class, imports) from source files.

Given a dataset row's repo and path, loads the source file from `dataset/repos/`
and pulls signals that help the LLM summarize a target function.

This module holds the shared pieces — source loading, the FileContext record,
rendering, and language dispatch. Both backends parse with tree-sitter:

Python (file_context_python.py):
  - module_doc: top-of-file docstring (stripped)
  - class_name / class_doc: enclosing class name and docstring (if method)
  - imports: top-level import statements

Java (file_context_java.py):
  - class_name / class_doc: enclosing class name and Javadoc
  - imports: top-level import statements
  - module_doc is always None (Java has no file-level docstring)

There is intentionally no forced common mapping between languages.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPOS_ROOT = Path(__file__).parents[3] / "dataset" / "repos"


@dataclass
class FileContext:
    language: str  # "python" | "java"
    # Python only: top-of-file docstring. Always None for Java.
    module_doc: str | None
    # Direct enclosing class of the target function (Python and Java).
    class_name: str | None
    class_doc: str | None
    imports: list[str]

    def is_empty(self) -> bool:
        if self.language == "python":
            return not (self.module_doc or self.class_name or self.imports)
        else:
            return not (self.class_name or self.imports)


def _repo_dir(repo: str) -> Path:
    """Map 'apache/airflow' -> dataset/repos/apache__airflow."""
    return REPOS_ROOT / repo.replace("/", "__")


@lru_cache(maxsize=512)
def _path_at_sha(repo: str, current_path: str, sha: str) -> str:
    """Return the path `current_path` had at `sha`, following renames backwards.

    git blame --follow can attribute a line to a commit where the file lived
    under a different path. Walk the rename chain to find that path.
    """
    repo_dir = str(_repo_dir(repo))
    # Walk rename history for current_path from HEAD back to sha.
    result = subprocess.run(
        ["git", "log", "--follow", "--diff-filter=R", "--name-status", "--format=%H", "--", current_path],
        cwd=repo_dir, capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if not lines[i].startswith("R"):
            commit = lines[i]
            i += 1
            if i < len(lines) and lines[i].startswith("R"):
                parts = lines[i].split("\t")
                if len(parts) == 3:
                    old_path = parts[1]
                    # If sha is an ancestor of (or equal to) this rename commit,
                    # the file was at old_path at sha.
                    r = subprocess.run(
                        ["git", "merge-base", "--is-ancestor", sha, commit],
                        cwd=repo_dir, capture_output=True,
                    )
                    if r.returncode == 0:
                        return old_path
                i += 1
        else:
            i += 1
    return current_path


@lru_cache(maxsize=512)
def _load_bytes(repo: str, path: str, sha: str | None = None) -> bytes | None:
    """Read the file, pinned to `sha` when given. tree-sitter parses bytes."""
    if sha is not None:
        resolved = _path_at_sha(repo, path, sha)
        result = subprocess.run(
            ["git", "show", f"{sha}:{resolved}"],
            cwd=str(_repo_dir(repo)),
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    return (_repo_dir(repo) / path).read_bytes()


def _load_source(repo: str, path: str, sha: str | None = None) -> str | None:
    """Text form of `_load_bytes`, for callers that want str (e.g. rtc.py)."""
    raw = _load_bytes(repo, path, sha)
    return None if raw is None else raw.decode("utf-8", errors="replace")


def _extract_func_name_from_code(code: str) -> str | None:
    """Find the first `def NAME` in a Python code snippet.

    Re-exported from the Python backend; kept here because callers import it
    from this module.
    """
    from .file_context_python import _extract_func_name_from_code as _impl
    return _impl(code)


def extract_file_context(
    repo: str,
    path: str,
    func_name: str | None = None,
    code: str | None = None,
    max_imports: int = 25,
    language: str = "python",
    sha: str | None = None,
) -> FileContext:
    """Parse the repo file at `path` and extract module/class/imports context.

    `func_name` may be a bare name ('get_conn') or 'ClassName.method'; either works.
    If not given, we try to parse it from `code`.
    `max_imports` caps how many import statements are collected (0 disables).
    `language` selects the extractor backend ('python' or 'java').
    """
    # Lazy imports so a single-language run only pays for one grammar.
    if language == "python":
        from .file_context_python import extract_python_file_context
        return extract_python_file_context(
            repo, path, func_name=func_name, code=code, max_imports=max_imports, sha=sha
        )
    if language == "java":
        from .file_context_java import extract_java_file_context
        return extract_java_file_context(
            repo, path, func_name=func_name, code=code, max_imports=max_imports, sha=sha
        )
    raise ValueError(f"Unsupported language for file context: {language!r}")


def render_file_context(
    ctx: FileContext,
    max_module_doc_chars: int = 400,
    max_class_doc_chars: int = 300,
) -> str:
    """Render a FileContext as a compact block for prompt inclusion. Empty -> ''.

    Labels are language-specific:
      Python: "Module docstring:", "Enclosing class:"
      Java:   "Class:"
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
        if ctx.class_name:
            line = f"Class: {ctx.class_name}"
            if ctx.class_doc:
                line += f" — {_truncate(ctx.class_doc, max_class_doc_chars)}"
            parts.append(line)

    if ctx.imports:
        parts.append("Imports:\n" + "\n".join(ctx.imports))
    return "\n".join(parts)


def extract_file_outline(
    repo: str,
    path: str,
    exclude_func_name: str | None = None,
    language: str = "python",
    max_chars: int = 4000,
    sha: str | None = None,
) -> str:
    """Return a compact outline of the file suitable for prompt inclusion.

    Temporal safety comes from `sha`: the file is read at the sample's blame
    commit via `git show`, so the outline cannot contain code written after
    the target function. (A `cutoff_timestamp` parameter used to sit here but
    was never read by the body — the sha pin was always doing the work.)

    Both languages extract function/class signatures and their first-line
    docs from the tree-sitter parse, skipping the target function. This gives
    the LLM naming conventions and documentation style without flooding it
    with implementation details, and never emits raw source.

    `exclude_func_name` may be a bare name or 'ClassName.method' — only the bare
    name is used for matching, so either form works.
    """
    bare_exclude: str | None = None
    if exclude_func_name:
        bare_exclude = exclude_func_name.split(".")[-1]

    if language == "python":
        from .file_context_python import _build_outline_python, _parse as _py_parse
        parsed = _py_parse(repo, path, sha)
        if parsed is None:
            return ""
        root, src = parsed
        blocks = _build_outline_python(root, src, bare_exclude)
    elif language == "java":
        from .file_context_java import _build_outline_java, _parse as _java_parse
        parsed = _java_parse(repo, path, sha)
        if parsed is None:
            return ""
        root, src = parsed
        blocks = _build_outline_java(root, src, bare_exclude)
    else:
        # Unsupported language — return nothing rather than risk leaking docstrings.
        return ""

    outline = "\n\n".join(blocks)
    if len(outline) > max_chars:
        outline = outline[:max_chars].rstrip() + "\n... [truncated]"
    return outline
