"""Java file-context extractor using tree-sitter-java.

Returns the same FileContext shape as the Python extractor so the summarizer
stays language-agnostic.

Signal mapping:
  - module_doc   -> Javadoc immediately preceding the top-level class declaration
  - class_name   -> enclosing class of the target method
  - class_doc    -> Javadoc immediately preceding that class
  - imports      -> top-level `import ...;` statements (capped by max_imports)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from .file_context import FileContext

REPOS_ROOT = Path(__file__).parents[3] / "dataset" / "repos"

_JAVADOC_LINE_RE = re.compile(r"^\s*\*?\s?", re.MULTILINE)


def _repo_dir(repo: str) -> Path:
    return REPOS_ROOT / repo.replace("/", "__")


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tree_sitter_java.language()))


@lru_cache(maxsize=512)
def _load_bytes(repo: str, path: str) -> bytes:
    return (_repo_dir(repo) / path).read_bytes()


@lru_cache(maxsize=512)
def _parse(repo: str, path: str) -> tuple[Node, bytes]:
    src = _load_bytes(repo, path)
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


def _find_top_level_class(root: Node) -> Node | None:
    for ch in root.children:
        if ch.type == "class_declaration":
            return ch
    return None


def _find_enclosing_class(root: Node, target_method: str) -> Node | None:
    """Walk the tree; return the class_declaration that directly contains a
    method_declaration named target_method."""
    def walk(node: Node, current_class: Node | None) -> Node | None:
        if node.type == "class_declaration":
            current_class = node
        if node.type == "method_declaration":
            name = node.child_by_field_name("name")
            if name is not None and name.text.decode() == target_method:
                return current_class
        for c in node.children:
            found = walk(c, current_class)
            if found is not None:
                return found
        return None

    return walk(root, None)


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


def extract_java_file_context(
    repo: str,
    path: str,
    func_name: str | None = None,
    code: str | None = None,
    max_imports: int = 25,
) -> FileContext:
    root, src = _parse(repo, path)

    # Module-doc analog: Javadoc on the top-level class.
    top_class = _find_top_level_class(root)
    module_doc = _preceding_javadoc(top_class, src) if top_class is not None else None

    # Enclosing class for the target method.
    if func_name is None and code is not None:
        func_name = _extract_method_name_from_code(code)

    class_name: str | None = None
    class_doc: str | None = None
    if func_name:
        bare = func_name.split(".")[-1]
        cls = _find_enclosing_class(root, bare)
        if cls is not None:
            name_node = cls.child_by_field_name("name")
            if name_node is not None:
                class_name = _text(name_node, src)
            class_doc = _preceding_javadoc(cls, src)

    imports = _collect_imports(root, src, max_imports=max_imports)

    return FileContext(
        module_doc=module_doc.strip() if module_doc else None,
        class_name=class_name,
        class_doc=class_doc.strip() if class_doc else None,
        imports=imports,
    )
