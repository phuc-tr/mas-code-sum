"""
ASAP-style identifier analysis (id3) using tree-sitter.

Classifies identifiers in a function by syntactic role:
  - Parameters
  - Return identifiers
  - Method invocations
  - Method arguments
  - Assignments / variable declarations
  - Attribute / dotted-name access

Supports Python and Java. Returns a formatted text block that can be
injected into an LLM prompt.
"""

from __future__ import annotations

from functools import lru_cache

from tree_sitter import Language, Node, Parser

# ── language loading ──────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _get_parser(language: str) -> tuple[Parser, str]:
    """Return (Parser, canonical_language_name) for 'python' or 'java'."""
    lang = language.lower()
    if lang == "python":
        import tree_sitter_python as ts_lang
        ts_language = Language(ts_lang.language())
    elif lang == "java":
        import tree_sitter_java as ts_lang
        ts_language = Language(ts_lang.language())
    else:
        raise ValueError(f"Unsupported language for identifier extraction: {language!r}")
    return Parser(ts_language), lang


# ── parent-type → category mapping ───────────────────────────────────────────

# ASAP uses ty[1] (the immediate parent type of each identifier leaf).
# Python and Java share most categories but differ in a few node names.

_PYTHON_RULES: dict[str, str] = {
    "parameters": "parameters",
    "return_statement": "return",
    "assignment": "assignments",
    "argument_list": "arguments",
    "call": "method_invocations",
}

_JAVA_RULES: dict[str, str] = {
    "formal_parameter": "parameters",
    "return_statement": "return",
    "variable_declarator": "assignments",
    "argument_list": "arguments",
    "method_invocation": "method_invocations",
}

# For both languages, parent types containing "attribute", "dotted_name",
# or "field_access", "array_access" map to "access".
_ACCESS_KEYWORDS = ("attribute", "dotted_name", "access")


def _classify(parent_type: str, language: str) -> str | None:
    rules = _PYTHON_RULES if language == "python" else _JAVA_RULES
    if parent_type in rules:
        return rules[parent_type]
    if any(kw in parent_type for kw in _ACCESS_KEYWORDS):
        return "access"
    return None


# ── tree walking ──────────────────────────────────────────────────────────────

def _walk_identifiers(node: Node, code: bytes) -> list[tuple[str, str]]:
    """DFS walk; yield (text, parent_type) for every identifier leaf."""
    results: list[tuple[str, str]] = []
    if node.child_count == 0 and node.type == "identifier":
        text = code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        parent_type = node.parent.type if node.parent else ""
        results.append((text, parent_type))
    else:
        for child in node.children:
            results.extend(_walk_identifiers(child, code))
    return results


# ── public API ────────────────────────────────────────────────────────────────

def extract_identifier_context(code: str, language: str, func_name: str | None = None) -> str:
    """
    Parse *code* and return an ASAP-style identifier context block.

    Example output::

        We categorized the identifiers into different classes. Please find the information below.
        Function name: process_data
        Parameters of the function: ['data', 'config']
        Identifier to be returned: ['result']
        Method Invocation: ['transform']
        Method Arguments: ['data', 'result']
        Assignments: ['result']
        Identifier to access attribute/dotted name: ['config', 'value']

    Returns an empty string if parsing fails or no identifiers are found.
    """
    lang = language.lower()
    try:
        parser, lang = _get_parser(lang)
    except ValueError:
        return ""

    try:
        tree = parser.parse(code.encode("utf-8", errors="replace"))
    except Exception:
        return ""

    root = tree.root_node
    pairs = _walk_identifiers(root, code.encode("utf-8", errors="replace"))

    # Determine bare function name (strip "ClassName." prefix if present)
    if func_name and "." in func_name:
        fname = func_name.split(".")[-1]
    else:
        fname = func_name or ""

    buckets: dict[str, list[str]] = {
        "parameters": [],
        "return": [],
        "method_invocations": [],
        "arguments": [],
        "assignments": [],
        "access": [],
    }

    for text, parent_type in pairs:
        if text == fname:
            continue
        category = _classify(parent_type, lang)
        if category and text not in buckets[category]:
            buckets[category].append(text)

    lines: list[str] = [
        "We categorized the identifiers into different classes. Please find the information below."
    ]
    if func_name:
        lines.append(f"Function name: {func_name}")

    label_map = {
        "parameters": "Parameters of the function",
        "return": "Identifier to be returned",
        "method_invocations": "Method Invocation",
        "arguments": "Method Arguments",
        "assignments": "Assigments" if lang == "python" else "Variable declaration",
        "access": "Identifier to access attribute/dotted name" if lang == "python" else "Identifier to access field/array",
    }

    has_any = False
    for key, label in label_map.items():
        if buckets[key]:
            lines.append(f"{label}: {buckets[key]}")
            has_any = True

    if not has_any:
        return ""

    return "\n".join(lines)
