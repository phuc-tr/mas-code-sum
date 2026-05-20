"""Compare file-context overlap vs code-body overlap with ground-truth summaries.

The core question: does the class/module docstring tell the model things the
function body *doesn't already say*?

For each test sample we compute Jaccard similarity (content tokens) between
the GT summary and two sources:
  - CODE: the function body tokens
  - CTX:  the best-matching class/module docstring

If CTX > CODE on average, the file context adds signal beyond what any
summarizer already reads.  We also find the "high-gap" cases — where the code
body scores low but the class doc scores high — and show them as concrete
examples of where file context matters most.

Run from the project root:
  uv run python scripts/analyze_file_context_overlap.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mas_code_sum.enrichers.file_context import extract_file_context  # noqa: E402

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "on", "at", "by", "from", "as", "or", "and", "but", "if", "with",
    "it", "its", "this", "that", "not", "no", "all", "any", "some",
    "each", "which", "what", "when", "where", "how", "who", "returns",
    "return", "method", "class", "object", "instance", "value", "values",
    "given", "used", "use", "get", "set", "list", "type", "null", "true", "false",
}


def tokenize_content(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


# Matches triple-quoted strings (both ''' and """) at the start of a function body.
_DOCSTRING_RE = re.compile(
    r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
    re.MULTILINE,
)

# Matches /** ... */ Javadoc blocks.
_JAVADOC_RE = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)


def strip_inline_docstring(code: str, lang: str) -> str:
    """Remove the embedded docstring/Javadoc from a code snippet.

    Python: strips the first triple-quoted string that appears right after
    the `def` line (which is always the docstring in this dataset).
    Java: strips /** ... */ blocks that precede the method body.
    """
    if lang == "python":
        return _DOCSTRING_RE.sub("", code, count=1)
    elif lang == "java":
        return _JAVADOC_RE.sub("", code)
    return code


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def bucket_distribution(values: list[float], thresholds=(0.05, 0.10, 0.20, 0.30)) -> str:
    n = len(values)
    if n == 0:
        return "(no data)"
    lines = []
    prev = 0.0
    for t in list(thresholds) + [float("inf")]:
        count = sum(1 for v in values if prev <= v < t)
        label = f"[{prev:.2f}, {t:.2f})" if t != float("inf") else f">= {thresholds[-1]:.2f}"
        pct = 100 * count / n
        bar = "█" * int(pct / 2)
        lines.append(f"  {label}: {count:>4} ({pct:5.1f}%) {bar}")
        prev = t
    return "\n".join(lines)


def analyze_language(lang: str, test_path: Path) -> None:
    print(f"\n{'='*64}")
    print(f"  {lang.upper()} test set")
    print(f"{'='*64}")

    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]

    code_scores: list[float] = []
    ctx_scores: list[float] = []
    gaps: list[dict] = []          # high-gap cases: ctx >> code
    ctx_wins: list[dict] = []      # ctx > code (any margin)
    code_wins: list[dict] = []     # code > ctx
    coverage = {"any": 0, "class_doc": 0, "module_doc": 0}
    errors = 0

    for row in rows:
        gt_content = tokenize_content(row["docstring"])
        clean_code = strip_inline_docstring(row["code"], lang)
        code_content = tokenize_content(clean_code)

        try:
            ctx = extract_file_context(
                row["repo"], row["path"],
                func_name=row["func_name"],
                language=lang, max_imports=0,
            )
        except Exception:
            errors += 1
            continue

        ctx_docs: list[str] = []
        if ctx.module_doc:
            ctx_docs.append(ctx.module_doc)
            coverage["module_doc"] += 1
        if ctx.class_doc:
            ctx_docs.append(ctx.class_doc)
            coverage["class_doc"] += 1
        if ctx.outer_class_doc:
            ctx_docs.append(ctx.outer_class_doc)
        if ctx_docs:
            coverage["any"] += 1

        code_j = jaccard(gt_content, code_content)
        code_scores.append(code_j)

        if not ctx_docs:
            continue

        best_ctx_j = max(jaccard(gt_content, tokenize_content(d)) for d in ctx_docs)
        ctx_scores.append(best_ctx_j)

        gap = best_ctx_j - code_j
        record = {
            "repo": row["repo"],
            "func": row["func_name"],
            "gt": row["docstring"].strip(),
            "code_snippet": row["code"][:300],
            "ctx_doc": ctx_docs[0][:300],
            "code_j": round(code_j, 3),
            "ctx_j": round(best_ctx_j, 3),
            "gap": round(gap, 3),
        }
        if best_ctx_j > code_j:
            ctx_wins.append(record)
        elif code_j > best_ctx_j:
            code_wins.append(record)
        if gap >= 0.15:
            gaps.append(record)

    n = len(rows)
    n_ctx = len(ctx_scores)

    print(f"\n[COVERAGE]  n={n}, errors={errors}")
    print(f"  Has any context doc : {coverage['any']/n:.1%}  ({coverage['any']}/{n})")
    print(f"  Has class_doc       : {coverage['class_doc']/n:.1%}")
    if lang == "python":
        print(f"  Has module_doc      : {coverage['module_doc']/n:.1%}")

    print(f"\n[CODE vs CONTEXT DOC — content-token Jaccard with GT]")
    print(f"  {'Source':<20}  {'Mean Jaccard':>14}  {'> 0.05':>8}  {'> 0.10':>8}")
    print(f"  {'-'*54}")
    for label, scores in [("Code body", code_scores), ("Context doc", ctx_scores)]:
        m = mean(scores)
        p05 = 100 * sum(1 for v in scores if v > 0.05) / len(scores) if scores else 0
        p10 = 100 * sum(1 for v in scores if v > 0.10) / len(scores) if scores else 0
        print(f"  {label:<20}  {m:>14.4f}  {p05:>7.1f}%  {p10:>7.1f}%")

    lift = (mean(ctx_scores) - mean(code_scores)) / mean(code_scores) * 100 if mean(code_scores) else 0
    print(f"\n  Context doc mean Jaccard is {lift:+.1f}% vs code body")

    print(f"\n[WIN RATES]  (over {n_ctx} samples with a context doc)")
    n_ctx_w = len(ctx_wins)
    n_code_w = len(code_wins)
    n_tie = n_ctx - n_ctx_w - n_code_w
    print(f"  Context doc wins  (ctx_j > code_j): {n_ctx_w:>4} ({100*n_ctx_w/n_ctx:.1f}%)")
    print(f"  Code body wins    (code_j > ctx_j): {n_code_w:>4} ({100*n_code_w/n_ctx:.1f}%)")
    print(f"  Ties              (equal):           {n_tie:>4} ({100*n_tie/n_ctx:.1f}%)")

    print(f"\n[DISTRIBUTIONS]")
    print("  Code body Jaccard:")
    print(bucket_distribution(code_scores))
    print("  Context doc Jaccard:")
    print(bucket_distribution(ctx_scores))

    # Sort high-gap examples by gap descending, take top 6
    gaps.sort(key=lambda x: x["gap"], reverse=True)
    print(f"\n[HIGH-GAP EXAMPLES]  ctx_j - code_j >= 0.15  ({len(gaps)} cases)")
    print("  These are functions where the code body barely overlaps GT,")
    print("  but the class/module doc does — the clearest wins for file context.\n")
    for ex in gaps[:6]:
        print(f"  func     : {ex['func']}")
        print(f"  GT       : {ex['gt'][:180]}")
        print(f"  ctx_doc  : {ex['ctx_doc'][:180]}")
        print(f"  code_j={ex['code_j']}  ctx_j={ex['ctx_j']}  gap={ex['gap']}")
        print()


def main() -> None:
    dataset_root = Path(__file__).parents[1] / "dataset"
    for lang in ("python", "java"):
        analyze_language(lang, dataset_root / lang / "test.jsonl")


if __name__ == "__main__":
    main()
