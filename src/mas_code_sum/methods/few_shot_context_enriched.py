"""Few-shot summarizer enriched with repository context (name, about, file path).

Uses the completion endpoint with <s>...</s> delimiters, matching the ASAP
prompt style. Each few-shot example includes the example's own repository name
and description so the model sees that context alongside the code.

Prompt structure per few-shot example:
  Repository: {repo}
  Repository description: {about}
  Code:
  {code}
  Summary: <s>{docstring}</s>

Then for the query:
  Repository: {repo}
  Repository description: {about}
  Code:
  {code}
  Summary: <s>
"""

from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_local_clients, make_openai_clients, strip_code_fences
from .zero_shot_context_enriched import _get_metadata_index


def _build_block(
    code: str,
    repo: str | None,
    about: str | None,
    path: str | None,
    docstring: str | None,
) -> str:
    parts: list[str] = []
    if repo:
        parts.append(f"Repository: {repo}")
    if about:
        parts.append(f"Repository description: {about}")
    if path:
        parts.append(f"File: {path}")
    parts.append(f"Code:\n{code}")
    if docstring is not None:
        parts.append(f"Summary: <s>{docstring}</s>")
    else:
        parts.append("Summary: <s>")
    return "\n".join(parts)


class FewShotContextEnrichedSummarizer(BaseSummarizer):
    """Few-shot LLM summarizer enriched with repo name, description, and file path."""

    name = "few_shot_context_enriched"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        retriever: BaseRetriever = None,
        example_paths: bool = False,
        backend: str = "openrouter",
        device: str = "cuda",
    ):
        self.model = model
        self.retriever = retriever
        self.example_paths = example_paths
        self.backend = backend
        self.max_concurrency = 10
        if backend == "local":
            self._client, self._async_client = make_local_clients(model, device)
        else:
            self._client, self._async_client = make_openai_clients()

    def _example_block(self, s: dict) -> str:
        code = " ".join(s["code_tokens"])
        docstring = " ".join(s["docstring_tokens"])
        repo = s.get("repo")
        about: str | None = None
        if repo:
            about = _get_metadata_index().get(repo, {}).get("about")
        path = s.get("path") if self.example_paths else None
        return _build_block(code, repo, about, path, docstring)

    def _query_block(self, code: str, project: str | None, path: str | None) -> str:
        about: str | None = None
        if project:
            about = _get_metadata_index().get(project, {}).get("about")
        return _build_block(code, project, about, path, docstring=None)

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        blocks = [self._example_block(s) for s in examples]
        blocks.append(self._query_block(code, project, path))
        prompt = "\n\n".join(blocks)

        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=30,
            temperature=0.0,
        )
        raw = response.choices[0].text or ""
        end = raw.find("</s>")
        comment = raw[:end].strip() if end != -1 else raw.split("\n")[0].strip()
        return strip_code_fences(comment)

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        blocks = [self._example_block(s) for s in examples]
        blocks.append(self._query_block(code, project, path))
        prompt = "\n\n".join(blocks)

        response = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=30,
            temperature=0.0,
        )
        raw = response.choices[0].text or ""
        end = raw.find("</s>")
        comment = raw[:end].strip() if end != -1 else raw.split("\n")[0].strip()
        return strip_code_fences(comment)

    def params(self) -> dict:
        return {
            "model": self.model,
            "backend": self.backend,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "example_paths": self.example_paths,
        }
