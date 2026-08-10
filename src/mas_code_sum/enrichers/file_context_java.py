"""Java file-context extractor using tree-sitter-java.

Java-specific signals (no forced mapping to Python concepts):
  - class_name / class_doc   -> direct enclosing class of the target method
                                and its Javadoc
  - imports                  -> top-level `import ...;` statements
  - module_doc               -> always None (Java has no file-level docstring)
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from .file_context import FileContext

REPOS_ROOT = Path(__file__).parents[3] / "dataset" / "repos"

_JAVADOC_LINE_RE = re.compile(r"^\s*\*?\s?", re.MULTILINE)


def _repo_dir(repo: str) -> Path:
    return REPOS_ROOT / repo.replace("/", "__")


@lru_cache(maxsize=512)
def _path_at_sha(repo: str, current_path: str, sha: str) -> str:
    """Return the path `current_path` had at `sha`, following renames backwards."""
    repo_dir = str(_repo_dir(repo))
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


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tree_sitter_java.language()))


@lru_cache(maxsize=512)
def _load_bytes(repo: str, path: str, sha: str | None = None) -> bytes | None:
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


@lru_cache(maxsize=512)
def _parse(repo: str, path: str, sha: str | None = None) -> tuple[Node, bytes] | None:
    src = _load_bytes(repo, path, sha)
    if src is None:
        return None
    tree = _parser().parse(src)
    return tree.root_node, src


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _clean_javadoc(raw: str) -> str:
    """Strip /**, */, and per-line leading * from a Javadoc block."""
    s = raw.strip()
    if s.startswith("/**"):
        s = s[3:]
    elif s.startswith("/*"):
        s = s[2:]
    if s.endswith("*/"):
        s = s[:-2]
    s = _JAVADOC_LINE_RE.sub("", s)
    # Collapse runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _preceding_javadoc(node: Node, src: bytes) -> str | None:
    """Return cleaned Javadoc text immediately preceding `node`, if any."""
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "block_comment":
        text = _text(prev, src)
        if text.startswith("/**"):
            return _clean_javadoc(text)
    return None


def _find_enclosing_class(root: Node, target_method: str, qualifier: str | None = None) -> Node | None:
    """Walk the tree; return the class_declaration that directly contains a
    method_declaration named target_method.

    Several classes in one file can declare the same bare method name (e.g. a
    nested Builder and its outer class both declaring connect()). When
    qualifier is given, prefer the class whose name matches it; otherwise fall
    back to the first match in pre-order."""
    candidates: list[Node] = []

    def walk(node: Node, current_class: Node | None) -> None:
        if node.type == "class_declaration":
            current_class = node
        if node.type == "method_declaration":
            name = node.child_by_field_name("name")
            if name is not None and name.text.decode() == target_method and current_class is not None:
                candidates.append(current_class)
        for c in node.children:
            walk(c, current_class)

    walk(root, None)
    if not candidates:
        return None
    if qualifier:
        for cls in candidates:
            name_node = cls.child_by_field_name("name")
            if name_node is not None and name_node.text.decode() == qualifier:
                return cls
    return candidates[0]


def _collect_imports(root: Node, src: bytes, max_imports: int) -> list[str]:
    if max_imports <= 0:
        return []
    out: list[str] = []
    for ch in root.children:
        if ch.type == "import_declaration":
            out.append(_text(ch, src).strip())
            if len(out) >= max_imports:
                break
    return out


def _extract_method_name_from_code(code: str) -> str | None:
    """Parse a tokenized Java method snippet (wrapped in a dummy class) and
    return the method's name."""
    wrapped = b"class __D { " + code.encode("utf-8", errors="replace") + b" }"
    tree = _parser().parse(wrapped)

    def walk(n: Node) -> str | None:
        if n.type == "method_declaration":
            name = n.child_by_field_name("name")
            if name is not None:
                return wrapped[name.start_byte:name.end_byte].decode()
        for c in n.children:
            found = walk(c)
            if found is not None:
                return found
        return None

    return walk(tree.root_node)


def _build_outline_java(root: Node, src: bytes, exclude_name: str | None) -> list[str]:
    """Walk the tree and collect method/class signatures + first-line Javadoc,
    skipping any method named exclude_name."""

    def _first_doc_line(node: Node) -> str | None:
        doc = _preceding_javadoc(node, src)
        if not doc:
            return None
        return doc.split("\n")[0].strip()

    def _sig(node: Node) -> str:
        """Return the method signature (return type + name + params), no body."""
        parts: list[str] = []
        for ch in node.children:
            if ch.type == "block":
                break
            parts.append(_text(ch, src))
        return " ".join(parts).strip()

    blocks: list[str] = []

    def walk(node: Node, indent: str = "") -> None:
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _text(name_node, src) if name_node else "?"
            doc_line = _first_doc_line(node)
            class_block = f"{indent}class {class_name}"
            if doc_line:
                class_block += f"\n{indent}    \"{doc_line}\""
            blocks.append(class_block)
            for ch in node.children:
                walk(ch, indent + "    ")
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            method_name = _text(name_node, src) if name_node else None
            if method_name == exclude_name:
                return
            sig = _sig(node)
            doc_line = _first_doc_line(node)
            method_block = f"{indent}{sig}"
            if doc_line:
                method_block += f"\n{indent}    \"{doc_line}\""
            blocks.append(method_block)
        else:
            for ch in node.children:
                walk(ch, indent)

    for ch in root.children:
        walk(ch)

    return blocks


def extract_java_file_context(
    repo: str,
    path: str,
    func_name: str | None = None,
    code: str | None = None,
    max_imports: int = 25,
    sha: str | None = None,
) -> FileContext:
    parsed = _parse(repo, path, sha)
    if parsed is None:
        return FileContext(language="java", module_doc=None, class_name=None, class_doc=None,
                           imports=[])
    root, src = parsed

    if func_name is None and code is not None:
        func_name = _extract_method_name_from_code(code)

    class_name: str | None = None
    class_doc: str | None = None

    if func_name:
        parts = func_name.split(".")
        bare = parts[-1]
        qualifier = parts[-2] if len(parts) > 1 else None
        cls = _find_enclosing_class(root, bare, qualifier)
        if cls is not None:
            name_node = cls.child_by_field_name("name")
            if name_node is not None:
                class_name = _text(name_node, src)
            raw_doc = _preceding_javadoc(cls, src)
            class_doc = raw_doc.strip() if raw_doc else None

    imports = _collect_imports(root, src, max_imports=max_imports)

    return FileContext(
        language="java",
        module_doc=None,
        class_name=class_name,
        class_doc=class_doc,
        imports=imports,
    )
