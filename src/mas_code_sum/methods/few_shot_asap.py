"""
ASAP-style few-shot summarizer.

Enriches each prompt with optional identifier context (ASAP "id3"), optional
DFG context, and optional repository metadata, following the prompt structure
from:
  "ASAP: A Semi-Automated Pipeline for Context Enrichment in LLM-based
   Code Summarization" (turbo.py / davinci.py scripts).

Prompt structure per few-shot example:
  {code}
  {repo_context?}
  {id3_context?}
  {dfg_context?}
  Write down the original comment written by the developer.
  Comment: {docstring}

Then for the query:
  {code}
  {repo_context?}
  {id3_context?}
  {dfg_context?}
  Write down the original comment written by the developer.
  Comment:

DFG context is computed from the source on demand — see enrichers/asap/dfg.py.
"""

from __future__ import annotations

from ..enrichers.asap.dfg import get_dfg_context
from ..enrichers.asap.identifier_extractor import extract_identifier_context
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_clients, strip_code_fences

_COMMENT_PROMPT = "Write down the original comment written by the developer."


def _build_block(
    code: str,
    language: str,
    func_name: str | None,
    repo_ctx: str | None,
    dfg_ctx: str | None,
    docstring: str | None,
    *,
    use_id: bool = True,
    use_stop_tag: bool = False,
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

    id3 = extract_identifier_context(code, language, func_name=func_name) if use_id else ""

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
    if use_stop_tag:
        if docstring is not None:
            parts.append(f"Comment: <s>{docstring}</s>")
        else:
            parts.append("Comment: <s>")
    else:
        if docstring is not None:
            parts.append(f"Comment: {docstring}")
        else:
            parts.append("Comment: ")
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
        Whether to inject the ASAP data-flow context block. Computed from the
        function's source at prompt-build time; no setup required.
    use_id:
        Whether to inject the ASAP "id3" identifier-role context block.
    use_stop_tag:
        Whether to wrap the target comment in "<s>"/"</s>" stop tags (the
        original prompt format this codebase used before matching turbo.py
        exactly). When True, prompts, response parsing, and truncation are
        byte-for-byte identical to that earlier behavior. Default False
        matches turbo.py, which has no such tags.
    """

    name = "few_shot_asap"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        retriever: BaseRetriever | None = None,
        use_repo: bool = True,
        use_dfg: bool = False,
        use_id: bool = True,
        use_stop_tag: bool = False,
        backend: str = "featherless",
    ):
        self.model = model
        self.retriever = retriever
        self.use_repo = use_repo
        self.use_dfg = use_dfg
        self.use_id = use_id
        self.use_stop_tag = use_stop_tag
        self.backend = backend
        _, self._async_client = make_clients(backend)

    def build_prompt(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        """Assemble the flat ASAP prompt (few-shot blocks + query block)."""
        examples = self.retriever.retrieve(code, language, project=project, path=path)

        repo_ctx: str | None = None
        if self.use_repo and project:
            repo_ctx = f"Repository: {project}\nFile: {path or 'unknown'}"

        blocks: list[str] = []
        for s in examples:
            ex_code = " ".join(s["code_tokens"])
            ex_docstring = " ".join(s["docstring_tokens"])
            ex_func = s.get("func_name")
            ex_lang = s.get("language", language)
            ex_dfg: str | None = None
            if self.use_dfg:
                # From real source, not the joined tokens the block renders:
                # tokenization drops indentation and changes the parse.
                ex_dfg = get_dfg_context(s.get("original_string") or s.get("code") or "", ex_lang)
            blocks.append(_build_block(
                ex_code, ex_lang, ex_func, repo_ctx, ex_dfg, docstring=ex_docstring,
                use_id=self.use_id, use_stop_tag=self.use_stop_tag,
            ))

        query_dfg: str | None = None
        if self.use_dfg:
            query_dfg = get_dfg_context(code, language)
        blocks.append(_build_block(
            code, language, None, repo_ctx, query_dfg, docstring=None,
            use_id=self.use_id, use_stop_tag=self.use_stop_tag, dfg_before_id3=True,
        ))

        return "\n\n".join(blocks)

    def parse_reply(self, raw: str) -> str:
        """Cut the model's continuation down to the single comment line."""
        if self.use_stop_tag:
            start = raw.find("<s>")
            begin = start + len("<s>") if start != -1 else 0
            end = raw.find("</s>", begin)
            comment = raw[begin:end].strip() if end != -1 else raw.split("\n")[0].strip()
        else:
            comment = raw.split("\n")[0].strip()
        return strip_code_fences(comment)

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        prompt = self.build_prompt(code, language, project=project, path=path, url=url)
        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.0,
        )
        return self.parse_reply(response.choices[0].text or "")

    def params(self) -> dict:
        return {
            "model": self.model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "use_repo": self.use_repo,
            "use_dfg": self.use_dfg,
            "use_id": self.use_id,
            "use_stop_tag": self.use_stop_tag,
            "backend": self.backend,
        }
