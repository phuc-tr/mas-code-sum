"""Data-flow graph context for ASAP prompts, computed on demand.

ASAP shows the model a compact data-flow sketch of the function: one line per
variable occurrence, listing the indices it draws its value from. The DFG logic
itself is CodeXGLUE's, vendored under `asap_scripts/scripts/parser/`; this module
is the thin runtime wrapper around it.

Extraction costs about 1ms per function, so there is nothing to precompute --
results are memoized per process, which is enough to cover a function appearing
as a retrieved example many times over a run.

Extract from real source (`original_string` or `code`), never from a joined
`code_tokens` string: tokenization drops Python's indentation, and the parse
that follows produces a different graph.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# The vendored CodeXGLUE parser package lives outside src/ and is imported by
# path, the way asap_scripts' own entry points do.
_SCRIPTS_DIR = Path(__file__).parents[4] / "asap_scripts" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from parser.DFG import DFG_java, DFG_python  # noqa: E402
from parser.utils import (  # noqa: E402
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)

import tree_sitter_java as tsjava  # noqa: E402
import tree_sitter_python as tspython  # noqa: E402
from tree_sitter import Language, Parser  # noqa: E402

_HEADER = (
    "Please find the dataflow of the function. "
    "We present the source and list of target indices."
)
# Shown when a function has no usable data flow, so the prompt block keeps its
# shape across examples.
NO_DFG = f"{_HEADER}\nNo DFG available"

# ASAP keeps at most 30 lines in the block. The cap applies to emitted lines,
# after the full edge dictionary is built — capping the edges consumed instead
# would silently truncate the target list of the last key kept.
_MAX_LINES = 30

_PARSERS: dict[str, tuple[Parser, object]] = {}


def _get_parser(language: str) -> tuple[Parser, object]:
    if language not in _PARSERS:
        if language == "python":
            lang, dfg_fn = Language(tspython.language()), DFG_python
        elif language == "java":
            lang, dfg_fn = Language(tsjava.language()), DFG_java
        else:
            raise ValueError(f"Unsupported language for DFG: {language!r}")
        _PARSERS[language] = (Parser(lang), dfg_fn)
    return _PARSERS[language]


@lru_cache(maxsize=8192)
def extract_dataflow(code: str, language: str) -> str:
    """Return the ASAP-format DFG block for *code*, or '' when it has none.

    Parse failures degrade to an empty result rather than raising: the dataset
    holds legacy and partial source, and a missing sketch is a weaker prompt,
    not a failed run.
    """
    if not code:
        return ""
    parser, dfg_fn = _get_parser(language)

    try:
        clean = remove_comments_and_docstrings(code, language)
    except Exception:
        clean = code

    try:
        root = parser.parse(bytes(clean, "utf8")).root_node
        tokens_index = tree_to_token_index(root)
        code_lines = clean.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
        index_to_code = {
            index: (idx, tok)
            for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens))
        }
        try:
            dfg, _ = dfg_fn(root, index_to_code, {})
        except Exception:
            dfg = []
        dfg = sorted(dfg, key=lambda x: x[1])
        # Keep only nodes that participate in an edge.
        active = set()
        for d in dfg:
            if len(d[-1]) != 0:
                active.add(d[1])
            active.update(d[-1])
        dfg = [d for d in dfg if d[1] in active]
    except Exception:
        dfg = []

    if not dfg:
        return ""

    edges: dict[str, list[int]] = {}
    for w in dfg:
        if len(w[4]) == 0:
            continue
        edges.setdefault(f"{w[0]}-{w[4]}", []).append(w[1])

    if not edges:
        return ""
    lines = [f"{k} {v}" for k, v in list(edges.items())[:_MAX_LINES]]
    return "\n".join([_HEADER, *lines])


def get_dfg_context(code: str, language: str) -> str:
    """DFG block for a prompt: the sketch, or the `No DFG available` placeholder."""
    return extract_dataflow(code, language) or NO_DFG
