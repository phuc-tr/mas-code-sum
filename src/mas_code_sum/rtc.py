"""Round-trip correctness (RTC) metric.

Reproduces the methodology from Allamanis et al., "Unsupervised Evaluation of
Code LLMs with Round-Trip Correctness" (ICML 2024): a *backward* model is
asked to reconstruct the original function from the *forward* prediction
(the description under test), spliced back into the source file in place of
the function (fill-in-the-middle). The reconstruction is then compared to the
original code with smoothed BLEU-4 -- a system whose descriptions round-trip
to more similar code is producing summaries that better capture the code's
actual behavior.

Ported from `scratch/roundtrip_correctness.ipynb`; see that notebook for the
full derivation and validation against multiple systems.
"""

from __future__ import annotations

import ast
import asyncio
import re

from codebleu import AVAILABLE_LANGS, calc_codebleu
from tqdm.asyncio import tqdm as atqdm

from .enrichers.file_context import _load_source
from .evaluator import bleu as _bleu
from .methods.llm_client import _call_with_rate_limit_retry, get_concurrency, make_clients

DEFAULT_BACKWARD_BACKEND = "openrouter"
BACKWARD_TEMPERATURE = 0.2
BACKWARD_MAX_TOKENS = 512

# Trimmed from the far ends of the file, keeping what's nearest the hole.
MAX_FILE_CONTEXT_CHARS = 4000

_LINE_COMMENT_PREFIX = {
    "python": "#",
    "ruby": "#",
    "java": "//",
    "javascript": "//",
    "go": "//",
    "php": "//",
}

BACKWARD_PROMPT_FILE_TEMPLATE = '''You are reconstructing a {language} function from its description, \
in place in the file it belongs to. Below is the file with the target function \
replaced by a TODO comment describing what it should do. Write the complete \
function implementation that should replace the TODO, consistent with the \
rest of the file (naming conventions, enclosing class). The file's existing \
imports already cover what you need -- do not add any import statements. Do \
not add explanation, and do not add a docstring or comment restating the \
description -- respond with only the executable function code.

File ({path}):
{file_with_todo}

Code:'''

BACKWARD_PROMPT_SIGNATURE_TEMPLATE = '''You are reconstructing a {language} function from its description. \
You are given the function signature and a natural-language description of what \
the function does. Write a complete, plausible implementation of the function \
body that satisfies the description. Do not add import statements. Do not add \
explanation, and do not add a docstring or comment restating the description \
-- respond with only the executable code, from the signature through the end \
of the function.

Signature:
{signature}

Description:
{description}

Code:'''

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def split_file_around_code(repo: str, path: str, code: str, blame_sha: str | None) -> tuple[str, str] | None:
    """Returns (text_before, text_after) surrounding `code` in the file it came from,
    or None if the file can't be loaded or `code` can't be found verbatim."""
    source = _load_source(repo, path, blame_sha)
    if source is None:
        return None
    idx = source.find(code)
    if idx == -1:
        idx = source.find(code.strip())
        if idx == -1:
            return None
    return source[:idx], source[idx + len(code):]


def truncate_file_context(before: str, after: str, max_chars: int) -> tuple[str, str]:
    half = max_chars // 2
    return before[-half:], after[:half]


def extract_signature(code: str, language: str) -> str:
    """Fallback-only: first line(s) up to the opening `:`/`{`, used when file splicing fails."""
    lines = code.strip("\n").splitlines()
    if language == "python":
        for i, line in enumerate(lines):
            if line.rstrip().endswith(":"):
                return "\n".join(lines[: i + 1])
        return lines[0] if lines else ""
    for i, line in enumerate(lines):
        if "{" in line:
            return "\n".join(lines[: i + 1])
    return lines[0] if lines else ""


def strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def strip_docstring(code: str, language: str) -> str:
    """Remove a Python function's docstring, if any, so RTC compares regenerated
    code logic rather than whether the backward model echoed the description
    back as a docstring. No-op for other languages (no body-embedded docstring
    convention) or code that doesn't parse as a single function."""
    if language != "python":
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return code
    func_body = tree.body[0].body
    if not func_body:
        return code
    first = func_body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
        return code
    lines = code.splitlines(keepends=True)
    del lines[first.lineno - 1 : first.end_lineno]
    return "".join(lines)


_BLOCK_DOC_COMMENT_LANGS = {"java", "javascript", "php"}
_BLOCK_DOC_COMMENT_RE = re.compile(r"/\*\*.*?\*/\s*", re.DOTALL)


def strip_block_doc_comment(code: str, language: str) -> str:
    """Remove Javadoc/JSDoc/PHPDoc-style `/** ... */` comment blocks, wherever
    they appear (normally just before the signature). No-op for languages that
    don't use this convention (python's docstring is handled separately;
    go/ruby don't have a body-embedded or `/** */`-style doc-comment form)."""
    if language not in _BLOCK_DOC_COMMENT_LANGS:
        return code
    return _BLOCK_DOC_COMMENT_RE.sub("", code)


_LEADING_DECORATOR_RE = re.compile(r"^[ \t]*@")


def strip_decorators(code: str) -> str:
    """Remove leading decorator/annotation lines (Python `@decorator`, Java/JS
    `@Annotation`) that precede a function/method signature. A backward model
    has no way to guess exact decorator arguments (e.g. `@expose('/extra_links')`)
    from a description alone, so their wording shouldn't count against
    reconstruction quality. Only strips from the top of the snippet, since
    decorators/annotations only ever appear there in a single extracted
    function; handles multi-line decorator calls by tracking paren balance."""
    lines = code.splitlines(keepends=True)
    i = 0
    while i < len(lines) and _LEADING_DECORATOR_RE.match(lines[i]):
        depth = lines[i].count("(") - lines[i].count(")")
        i += 1
        while depth > 0 and i < len(lines):
            depth += lines[i].count("(") - lines[i].count(")")
            i += 1
    return "".join(lines[i:])


_IMPORT_LINE_RE = {
    "python": re.compile(r"^[ \t]*(import\s|from\s+\S+\s+import\b)"),
    "java": re.compile(r"^[ \t]*import\s+"),
    "javascript": re.compile(r"^[ \t]*(import\s|const\s+.*=\s*require\()"),
    "go": re.compile(r"^[ \t]*import\s+"),
    "php": re.compile(r"^[ \t]*use\s+\S+;"),
    "ruby": re.compile(r"^[ \t]*require(_relative)?\s"),
}


def strip_import_lines(code: str, language: str) -> str:
    """Drop any import/require lines. The backward model is instructed not to
    add these (the surrounding file context already has them), but this is a
    defensive strip in case it does anyway -- an extraneous import line is
    n-gram noise, not a signal about reconstruction quality. No-op if `code`
    doesn't parse as a single function (i.e. we're comparing the fallback
    signature-only string, where there should be no import lines anyway)."""
    pattern = _IMPORT_LINE_RE.get(language)
    if pattern is None:
        return code
    lines = code.splitlines(keepends=True)
    return "".join(line for line in lines if not pattern.match(line))


def normalize_for_rtc_comparison(code: str, language: str) -> str:
    """Full cleanup pipeline applied to both sides before BLEU/CodeBLEU: strip
    doc comments (docstring/Javadoc/JSDoc/PHPDoc), decorators/annotations, and
    import lines, so the score reflects reconstructed code logic rather than
    prose or boilerplate neither side was asked to reproduce verbatim."""
    code = strip_block_doc_comment(code, language)
    code = strip_decorators(code)
    code = strip_docstring(code, language)
    code = strip_import_lines(code, language)
    return code


def build_backward_prompt(row: dict) -> str:
    """`row` needs: code, language, repo, path, blame_sha, prediction."""
    halves = split_file_around_code(row["repo"], row["path"], row["code"], row.get("blame_sha"))
    comment_prefix = _LINE_COMMENT_PREFIX.get(row["language"], "#")
    if halves is None:
        return BACKWARD_PROMPT_SIGNATURE_TEMPLATE.format(
            language=row["language"],
            signature=extract_signature(row["code"], row["language"]),
            description=row["prediction"],
        )
    before, after = truncate_file_context(*halves, max_chars=MAX_FILE_CONTEXT_CHARS)
    todo_comment = f"{comment_prefix} TODO(LLM): {row['prediction']}"
    file_with_todo = f"{before}{todo_comment}\n{after}"
    return BACKWARD_PROMPT_FILE_TEMPLATE.format(
        language=row["language"], path=row["path"], file_with_todo=file_with_todo
    )


def codebleu_score(original_code: str, backward_code: str, language: str) -> dict:
    """CodeBLEU (Ren et al., 2020) between original and round-tripped code, 0-100 scale.

    Unlike plain n-gram BLEU, this also weights language keywords more heavily
    (weighted n-gram match) and rewards AST/dataflow structural similarity
    (syntax match, dataflow match), so a reconstruction that's structurally
    equivalent but textually different (renamed variables, reordered args)
    scores higher than under BLEU alone. Falls back to zeros for languages
    `codebleu` doesn't support.
    """
    if language not in AVAILABLE_LANGS:
        return {"rtc_codebleu": 0.0, "codebleu_ngram_match": 0.0, "codebleu_weighted_ngram_match": 0.0,
                "codebleu_syntax_match": 0.0, "codebleu_dataflow_match": 0.0}
    result = calc_codebleu([original_code], [backward_code], lang=language)
    return {
        "rtc_codebleu": result["codebleu"] * 100,
        "codebleu_ngram_match": result["ngram_match_score"] * 100,
        "codebleu_weighted_ngram_match": result["weighted_ngram_match_score"] * 100,
        "codebleu_syntax_match": result["syntax_match_score"] * 100,
        "codebleu_dataflow_match": result["dataflow_match_score"] * 100,
    }


def rtc_compare(original_code: str, backward_code: str, language: str) -> dict:
    """Normalize both sides (strip doc comments, decorators, imports) and score
    with smoothed corpus-style BLEU-4 and CodeBLEU, 0-100 scale. Returns the
    exact strings BLEU compares alongside the scores, so callers can inspect
    what was actually diffed.

    Normalization happens first: the backward model is prompted from the
    description alone, so an original docstring restating that same
    description, a decorator/annotation it couldn't have guessed, or an
    incidental import line would otherwise count against/for reconstruction
    quality instead of measuring actual code-logic similarity.
    """
    og_code_compared = normalize_for_rtc_comparison(original_code, language)
    rtc_code_compared = normalize_for_rtc_comparison(backward_code, language)
    return {
        "rtc_bleu": _bleu([og_code_compared], rtc_code_compared)[0] * 100,
        "og_code_compared": og_code_compared,
        "rtc_code_compared": rtc_code_compared,
        **codebleu_score(og_code_compared, rtc_code_compared, language),
    }


def rtc_bleu(original_code: str, backward_code: str, language: str) -> float:
    """Smoothed corpus-style BLEU-4 between original and round-tripped code, 0-100 scale."""
    return rtc_compare(original_code, backward_code, language)["rtc_bleu"]


async def compute_rtc_scores(
    samples: list[dict],
    predictions: list[str],
    model: str,
    backend: str = DEFAULT_BACKWARD_BACKEND,
) -> list[dict]:
    """Round-trip each (sample, prediction) pair and return per-sample RTC detail.

    `samples` items need: code, language, repo, path, blame_sha (as loaded by
    `data.load_projects`). One backward-model call per sample.

    Each returned dict has: rtc_bleu (float), og_code_compared and
    rtc_code_compared (the exact post-docstring-strip strings BLEU compared),
    and backward_code (the raw fence-stripped model output, pre-docstring-strip).
    """
    _, async_client = make_clients(backend)
    sem = asyncio.Semaphore(get_concurrency(backend))

    async def _one(sample: dict, prediction: str) -> dict:
        row = {
            "code": sample["code"],
            "language": sample["language"],
            "repo": sample["repo"],
            "path": sample.get("path"),
            "blame_sha": sample.get("blame_sha"),
            "prediction": prediction,
        }
        prompt = build_backward_prompt(row)

        async def _call():
            async with sem:
                return await async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=BACKWARD_TEMPERATURE,
                    max_tokens=BACKWARD_MAX_TOKENS,
                )

        response = await _call_with_rate_limit_retry(_call)
        backward_code = strip_code_fence(response.choices[0].message.content or "")
        return {
            "backward_code": backward_code,
            **rtc_compare(sample["code"], backward_code, sample["language"]),
        }

    return list(await atqdm.gather(
        *[_one(sample, pred) for sample, pred in zip(samples, predictions)],
        desc="rtc backward",
    ))


def compute_rtc_scores_sync(
    samples: list[dict],
    predictions: list[str],
    model: str,
    backend: str = DEFAULT_BACKWARD_BACKEND,
) -> list[dict]:
    """Sync wrapper around `compute_rtc_scores`, for callers outside an event loop."""
    return asyncio.run(compute_rtc_scores(samples, predictions, model, backend))
