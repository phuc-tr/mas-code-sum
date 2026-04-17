"""Few-shot summarizer that adds file-level context (module doc, class, imports) to the query.

Extends the repo-level enrichment of FewShotContextEnrichedSummarizer by pulling
extra signals from the actual file at `dataset/repos/{repo}/{path}`:

  - module docstring
  - enclosing class name + docstring (if the target function is a method)
  - top-of-file imports

v1 scope:
  - Python only (non-Python queries fall back to repo-level enrichment).
  - Query-only enrichment: few-shot examples keep the lighter repo-level block
    to save tokens and isolate the effect of file context.

Prompt structure per few-shot example (same as few_shot_context_enriched):
  Repository: {repo}
  Repository description: {about}
  Code:
  {code}
  Summary: <s>{docstring}</s>

Query block (Python only):
  Repository: {repo}
  Repository description: {about}
  File: {path}
  Module docstring: ...
  Enclosing class: Foo — ...
  Imports:
  import ...
  Code:
  {code}
  Summary: <s>
"""

from ..enrichers.file_context import extract_file_context, render_file_context
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_openai_clients, strip_code_fences
from .few_shot_context_enriched import _build_block
from .zero_shot_context_enriched import _get_metadata_index


class FewShotFileContextSummarizer(BaseSummarizer):
    """Few-shot LLM summarizer with query-side file-level context (Python only)."""

    name = "few_shot_file_context"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        retriever: BaseRetriever = None,
        example_paths: bool = False,
        use_module_doc: bool = True,
        use_class_context: bool = True,
        max_imports: int = 25,
    ):
        self.model = model
        self.retriever = retriever
        self.example_paths = example_paths
        self.use_module_doc = use_module_doc
        self.use_class_context = use_class_context
        self.max_imports = max_imports  # 0 disables imports entirely
        self.max_concurrency = 10
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

    def _query_block(self, code: str, language: str, project: str | None, path: str | None) -> str:
        about: str | None = None
        if project:
            about = _get_metadata_index().get(project, {}).get("about")

        # Base block (repo/about/path/code) — identical to few_shot_context_enriched.
        parts: list[str] = []
        if project:
            parts.append(f"Repository: {project}")
        if about:
            parts.append(f"Repository description: {about}")
        if path:
            parts.append(f"File: {path}")

        # File-level context for supported languages; only when we can locate the file.
        if language in ("python", "java") and project and path:
            ctx = extract_file_context(
                project, path, code=code, max_imports=self.max_imports, language=language
            )
            # Apply per-field gates before rendering.
            if not self.use_module_doc:
                ctx.module_doc = None
            if not self.use_class_context:
                ctx.class_name = None
                ctx.class_doc = None
            rendered = render_file_context(ctx)
            if rendered:
                parts.append(rendered)

        parts.append(f"Code:\n{code}")
        parts.append("Summary: <s>")
        return "\n".join(parts)

    def build_prompt(self, code: str, language: str, project: str | None, path: str | None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        blocks = [self._example_block(s) for s in examples]
        blocks.append(self._query_block(code, language, project, path))
        return "\n\n".join(blocks)

    async def async_summarize(
        self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None
    ) -> str:
        prompt = self.build_prompt(code, language, project, path)
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

    def summarize(
        self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None
    ) -> str:
        prompt = self.build_prompt(code, language, project, path)
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
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "example_paths": self.example_paths,
            "use_module_doc": self.use_module_doc,
            "use_class_context": self.use_class_context,
            "max_imports": self.max_imports,
        }
