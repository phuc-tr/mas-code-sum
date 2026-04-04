"""
ASAP-style few-shot summarizer.

Enriches each prompt with identifier context (ASAP "id3"), optional DFG context,
and optional repository metadata, following the prompt structure from:
  "ASAP: A Semi-Automated Pipeline for Context Enrichment in LLM-based
   Code Summarization" (turbo.py / davinci.py scripts).

Prompt structure per few-shot example:
  {code}
  {repo_context?}
  {id3_context}
  {dfg_context?}
  Write down the original comment written by the developer.
  Comment: {docstring}

Then for the query:
  {code}
  {repo_context?}
  {id3_context}
  {dfg_context?}
  Write down the original comment written by the developer.
  Comment:

DFG requires pre-computation:
  bash scripts/setup_dfg_parser.sh
  python scripts/precompute_dfg.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from ..enrichers.dfg_loader import get_dfg_loader
from ..enrichers.identifier_extractor import extract_identifier_context
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, strip_code_fences
from .zero_shot_context_enriched import _get_metadata_index

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_COMMENT_PROMPT = "Write down the original comment written by the developer."
_NO_DFG = "Please find the dataflow of the function. We present the source and list of target indices.\nNo DFG available"


def _build_block(
    code: str,
    language: str,
    func_name: str | None,
    repo_ctx: str | None,
    dfg_ctx: str | None,
    docstring: str | None,
    *,
    dfg_before_id3: bool = False,
) -> str:
    """Build one prompt block in ASAP's format.

    In ASAP (turbo.py):
      - examples: code → repo → id3 → dfg → comment prompt → docstring
      - query:    code → repo → dfg → id3 → comment prompt → (blank)
    Use dfg_before_id3=True for the query block.
    """
    parts = [code.strip()]
    if repo_ctx:
        parts.append(repo_ctx.strip())

    id3 = extract_identifier_context(code, language, func_name=func_name)

    if dfg_before_id3:
        # query order: dfg then id3
        if dfg_ctx:
            parts.append(dfg_ctx.strip())
        if id3:
            parts.append(id3)
    else:
        # example order: id3 then dfg
        if id3:
            parts.append(id3)
        if dfg_ctx:
            parts.append(dfg_ctx.strip())

    parts.append(_COMMENT_PROMPT)
    if docstring is not None:
        parts.append(f"Comment: {docstring}")
    else:
        parts.append("Comment:")
    return "\n".join(parts)


class FewShotAsapSummarizer(BaseSummarizer):
    """
    Few-shot LLM summarizer with ASAP-style context enrichment.

    Parameters
    ----------
    model:
        Model ID (OpenRouter slug).
    retriever:
        A BaseRetriever (e.g. BM25Retriever) that returns training samples.
    use_repo:
        Whether to inject repository name + description into the prompt.
    use_dfg:
        Whether to inject pre-computed DFG context.
        Requires running setup_dfg_parser.sh + precompute_dfg.py first.
    """

    name = "few_shot_asap"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        retriever: BaseRetriever | None = None,
        use_repo: bool = True,
        use_dfg: bool = False,
    ):
        self.model = model
        self.retriever = retriever
        self.use_repo = use_repo
        self.use_dfg = use_dfg
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project)

        # Repo context (same block for all examples and the query)
        repo_ctx: str | None = None
        if self.use_repo and project:
            meta = _get_metadata_index().get(project, {})
            about = meta.get("about") or "N/A"
            repo_ctx = f"Repository: {project}\nDescription: {about}\nFile: {path or 'unknown'}"

        # DFG loader (only accessed if use_dfg=True)
        dfg_loader = get_dfg_loader() if self.use_dfg else None

        # Few-shot example blocks (order: repo → id3 → dfg)
        blocks: list[str] = []
        for s in examples:
            ex_code = " ".join(s["code_tokens"])
            ex_docstring = " ".join(s["docstring_tokens"])
            ex_func = s.get("func_name")
            ex_lang = s.get("language", language)
            ex_dfg: str | None = None
            if dfg_loader:
                ex_url = s.get("url", "")
                ex_dfg = dfg_loader.get(ex_lang, ex_url, split="train") or _NO_DFG
            block = _build_block(ex_code, ex_lang, ex_func, repo_ctx, ex_dfg, docstring=ex_docstring)
            blocks.append(block)

        # Query block (order: repo → dfg → id3, matching turbo.py lines 211-220)
        query_dfg: str | None = None
        if dfg_loader:
            query_dfg = dfg_loader.get(language, url or "", split="test") or _NO_DFG
        query_block = _build_block(code, language, None, repo_ctx, query_dfg, docstring=None, dfg_before_id3=True)
        blocks.append(query_block)

        prompt = "\n\n".join(blocks)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""

        first_line = raw.split("\n")[0].strip()
        return strip_code_fences(first_line)

    def params(self) -> dict:
        return {
            "model": self.model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "use_repo": self.use_repo,
            "use_dfg": self.use_dfg,
        }
