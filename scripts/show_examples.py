import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mas_code_sum.enrichers.file_context import extract_file_context

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","can","could",
    "of","in","to","for","on","at","by","from","as","or","and","but","if","with",
    "it","its","this","that","not","no","all","any","some","each","which","what",
    "when","where","how","who","returns","return","method","class","object",
    "instance","value","values","given","used","use","get","set","list","type",
    "null","true","false",
}

_DOCSTRING_RE = re.compile(r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', re.MULTILINE)
_JAVADOC_RE   = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)


def tok(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 2]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def strip_doc(code, lang):
    if lang == "python":
        return _DOCSTRING_RE.sub("", code, count=1)
    return _JAVADOC_RE.sub("", code)


def highlight(text, words):
    """Wrap matching words in [ ] so they stand out."""
    words = set(words)
    def repl(m):
        w = m.group(0)
        return f"[{w}]" if w.lower() in words else w
    return re.sub(r"[A-Za-z][A-Za-z0-9]*", repl, text)


root = Path(__file__).parents[1]

for lang in ("python", "java"):
    rows = [json.loads(l) for l in (root / "dataset" / lang / "test.jsonl").read_text().splitlines()]
    cases = []
    for row in rows:
        gt_toks  = tok(row["docstring"])
        code_toks = tok(strip_doc(row["code"], lang))
        try:
            ctx = extract_file_context(
                row["repo"], row["path"],
                func_name=row["func_name"],
                language=lang, max_imports=0,
            )
        except Exception:
            continue
        docs = [d for d in [ctx.module_doc, ctx.class_doc, ctx.outer_class_doc] if d]
        if not docs:
            continue
        best_doc  = max(docs, key=lambda d: jaccard(gt_toks, tok(d)))
        ctx_j     = jaccard(gt_toks, tok(best_doc))
        code_j    = jaccard(gt_toks, code_toks)
        gap       = ctx_j - code_j
        if gap >= 0.15:
            cases.append((gap, row, best_doc, code_j, ctx_j, gt_toks))

    cases.sort(key=lambda x: x[0], reverse=True)

    print()
    print("=" * 72)
    print(f"  {lang.upper()} — high-gap examples  (class doc beats code body)")
    print("=" * 72)

    for gap, row, best_doc, code_j, ctx_j, gt_toks in cases[:5]:
        gt_set = set(gt_toks)
        clean_code = strip_doc(row["code"], lang).strip()

        print()
        print(f"  func      : {row['func_name']}")
        print(f"  GT        : {row['docstring'].strip()[:200]}")
        print()
        print(f"  code body (stripped, first 6 lines):")
        for line in clean_code.splitlines()[:6]:
            print(f"    {line}")
        print()
        print(f"  class doc : {highlight(best_doc.strip()[:200], gt_set)}")
        print()
        print(f"  code_j={code_j:.3f}  ctx_j={ctx_j:.3f}  gap=+{gap:.3f}")
        print()
