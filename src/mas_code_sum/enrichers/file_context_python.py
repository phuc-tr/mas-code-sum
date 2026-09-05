"""Python file-context extractor using tree-sitter-python.

Mirrors file_context_java.py so both languages go through the same kind of
parser (concrete syntax tree + error recovery) instead of Python's stdlib ast.

Python-specific signals:
  - module_doc               -> top-of-file docstring
  - class_name / class_doc   -> direct enclosing class of the target function
                                and its docstring
  - imports                  -> top-level `import` / `from ... import` statements
"""

from __future__ import annotations

import inspect
import re
from functools import lru_cache

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from .file_context import FileContext, _load_bytes

_DEF_RE = re.compile(r"\b(?:async\s+)?def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(")


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tree_sitter_python.language()))


@lru_cache(maxsize=512)
def _parse(repo: str, path: str, sha: str | None = None) -> tuple[Node, bytes] | None:
    """Parse the source file. Returns None only when the file cannot be read.

    Unlike ast.parse, tree-sitter recovers from syntax errors, so legacy
    Python 2 sources in the dataset (e.g. `print "..."`) still yield a usable
    tree instead of forcing a fallback to repo-level context.
    """
    src = _load_bytes(repo, path, sha)
    if src is None:
        return None
    return _parser().parse(src).root_node, src


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _definition(node: Node) -> Node:
    """Unwrap a `decorated_definition` to the function/class it decorates."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            return inner
    return node


def _named(node: Node, src: bytes) -> str | None:
    name = node.child_by_field_name("name")
    return _text(name, src) if name is not None else None


def _body_statements(node: Node) -> list[Node]:
    """Return the statements inside a function/class body block (or module).

    Comments are dropped: tree-sitter keeps them as named nodes, but ast never
    saw them, and a license header before the docstring must not hide it.
    """
    if node.type == "module":
        children = node.named_children
    else:
        body = node.child_by_field_name("body")
        children = body.named_children if body is not None else []
    return [c for c in children if c.type != "comment"]


def _unescape(raw: str) -> str:
    """Resolve the escape sequences that actually show up in docstrings.

    ast returned the decoded string value; tree-sitter hands back source text.
    codecs' 'unicode_escape' would mangle non-ASCII, so translate the common
    escapes only and leave anything else (\\d in a regex docstring, say) alone.
    """
    out: list[str] = []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "\n": ""}
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw) and raw[i + 1] in simple:
            out.append(simple[raw[i + 1]])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _docstring(node: Node, src: bytes) -> str | None:
    """Return the docstring of a module / class / function node, or None.

    Dedented the same way ast.get_docstring does, via inspect.cleandoc.
    """
    stmts = _body_statements(node)
    if not stmts:
        return None
    first = stmts[0]
    if first.type != "expression_statement" or not first.named_children:
        return None
    string_node = first.named_children[0]
    if string_node.type != "string":
        return None
    content = [c for c in string_node.children if c.type == "string_content"]
    if not content:
        return ""
    # Python normalizes newlines at tokenization; the raw bytes may be CRLF.
    raw = "".join(_text(c, src) for c in content).replace("\r\n", "\n")
    prefix = _text(string_node, src).split('"')[0].split("'")[0].lower()
    if "r" not in prefix:
        raw = _unescape(raw)
    return inspect.cleandoc(raw)


def _extract_func_name_from_code(code: str) -> str | None:
    """Find the name of the first `def` in a code snippet.

    Tries tree-sitter first; falls back to regex because dataset code is often
    the single-line tokenized form, which no parser reads as a definition.
    """
    src = code.encode("utf-8", errors="replace")
    root = _parser().parse(src).root_node

    for stmt in root.named_children:
        node = _definition(stmt)
        if node.type == "function_definition":
            name = _named(node, src)
            if name:
                return name

    def walk(n: Node) -> str | None:
        if n.type == "function_definition":
            name = _named(n, src)
            if name:
                return name
        for c in n.children:
            found = walk(c)
            if found is not None:
                return found
        return None

    found = walk(root)
    if found is not None:
        return found
    m = _DEF_RE.search(code)
    return m.group(1) if m else None


def _find_enclosing_class(root: Node, target_name: str, qualifier: str | None = None) -> Node | None:
    """Return the class_definition directly containing a function named
    target_name, or None.

    Several classes in one file can define the same method name, so when
    qualifier is given prefer the class whose name matches it; otherwise fall
    back to the first match in pre-order.
    """
    candidates: list[Node] = []

    def walk(node: Node) -> None:
        if node.type == "class_definition":
            for stmt in _body_statements(node):
                item = _definition(stmt)
                if item.type == "function_definition" and item.child_by_field_name("name") is not None:
                    if item.child_by_field_name("name").text.decode() == target_name:
                        candidates.append(node)
                        break
        for c in node.children:
            walk(c)

    walk(root)
    if not candidates:
        return None
    if qualifier:
        for cls in candidates:
            name_node = cls.child_by_field_name("name")
            if name_node is not None and name_node.text.decode() == qualifier:
                return cls
    return candidates[0]


def _collect_imports(root: Node, src: bytes, max_imports: int) -> list[str]:
    """Return source text of top-level import statements (capped; 0 = none)."""
    if max_imports <= 0:
        return []
    out: list[str] = []
    for ch in root.named_children:
        if ch.type in ("import_statement", "import_from_statement", "future_import_statement"):
            out.append(_text(ch, src))
            if len(out) >= max_imports:
                break
    return out


def _build_outline_python(root: Node, src: bytes, exclude_name: str | None) -> list[str]:
    """Collect signature + first-line docstring for each top-level
    function/class (and each method of those classes).

    Skips any node whose name matches exclude_name (bare, without ClassName.
    prefix). Returns a list of formatted blocks.
    """
    lines = src.decode("utf-8", errors="replace").splitlines()
    blocks: list[str] = []

    def _first_doc_line(node: Node) -> str | None:
        doc = _docstring(node, src)
        if not doc:
            return None
        return doc.split("\n")[0].strip()

    def _sig_line(node: Node) -> str:
        """The physical `def`/`class` line (decorators excluded)."""
        return lines[node.start_point[0]].rstrip()

    for stmt in root.named_children:
        node = _definition(stmt)
        name = _named(node, src)
        if node.type == "function_definition":
            if name == exclude_name:
                continue
            sig = _sig_line(node)
            doc = _first_doc_line(node)
            blocks.append(f"{sig}\n    \"{doc}\"" if doc else sig)
        elif node.type == "class_definition":
            if name == exclude_name:
                continue
            class_sig = _sig_line(node)
            class_doc = _first_doc_line(node)
            class_block = f"{class_sig}\n    \"{class_doc}\"" if class_doc else class_sig
            method_blocks: list[str] = []
            for item_stmt in _body_statements(node):
                item = _definition(item_stmt)
                if item.type != "function_definition":
                    continue
                if _named(item, src) == exclude_name:
                    continue
                sig = "    " + _sig_line(item).lstrip()
                doc = _first_doc_line(item)
                method_blocks.append(f"{sig}\n        \"{doc}\"" if doc else sig)
            if method_blocks:
                blocks.append(class_block + "\n\n" + "\n\n".join(method_blocks))
            else:
                blocks.append(class_block)

    return blocks


def extract_python_file_context(
    repo: str,
    path: str,
    func_name: str | None = None,
    code: str | None = None,
    max_imports: int = 25,
    sha: str | None = None,
) -> FileContext:
    parsed = _parse(repo, path, sha)
    if parsed is None:
        return FileContext(language="python", module_doc=None, class_name=None, class_doc=None,
                           imports=[])
    root, src = parsed

    module_doc = _docstring(root, src)

    # Normalize func name: accept 'Class.method' or bare 'method'.
    if func_name is None and code is not None:
        func_name = _extract_func_name_from_code(code)

    class_name: str | None = None
    class_doc: str | None = None
    if func_name:
        parts = func_name.split(".")
        bare = parts[-1]
        qualifier = parts[-2] if len(parts) > 1 else None
        cls = _find_enclosing_class(root, bare, qualifier)
        if cls is not None:
            class_name = _named(cls, src)
            raw_doc = _docstring(cls, src)
            class_doc = raw_doc.strip() if raw_doc else None

    imports = _collect_imports(root, src, max_imports=max_imports)

    return FileContext(
        language="python",
        module_doc=module_doc.strip() if module_doc else None,
        class_name=class_name,
        class_doc=class_doc,
        imports=imports,
    )
