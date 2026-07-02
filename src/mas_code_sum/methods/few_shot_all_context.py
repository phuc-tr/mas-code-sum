"""Few-shot file-context summarizer that collapses the two-stage critic into one step.

The critic's two stages were:
  1. Completions call: few-shot examples + file context (module doc, class, imports).
  2. Chat critic call: file outline of sibling functions, used to fix verb form/style.

Here both signals are merged into a single completions prompt — the file outline is
appended to the query block so the base model sees naming conventions and doc style
upfront, without a separate refinement pass.

Query block structure:
  Repository: {repo}
  Repository description: {about}
  File: {path}
  <file context: module doc / enclosing class / imports>
  Other functions in file:
  <outline of sibling functions/classes>
  Code:
  {code}
  Summary: <s>
"""

from ..enrichers.file_context import (
    _extract_func_name_from_code,
    extract_file_context,
    extract_file_outline,
    render_file_context,
)
from ..enrichers.file_context_java import _extract_method_name_from_code as _extract_java_method_name
from ..retrievers.base import BaseRetriever
from .base import (
    BaseSummarizer,
    _call_with_rate_limit_retry,
    extract_summary,
    make_clients,
    strip_code_fences,
)
from ..data import get_metadata_index
from ..enrichers.repo_context import build_block


class FewShotAllContextSummarizer(BaseSummarizer):
    """Single-step few-shot summarizer merging file context and file outline."""

    name = "few_shot_all_context"

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B",
        retriever: BaseRetriever = None,
        example_paths: bool = False,
        use_outer_context: bool = True,
        use_class_context: bool = True,
        use_repo: bool = True,
        max_imports: int = 0,
        max_file_chars: int = 4000,
        backend: str = "featherless",
    ):
        self.model = model
        self.retriever = retriever
        self.example_paths = example_paths
        self.use_outer_context = use_outer_context
        self.use_class_context = use_class_context
        self.use_repo = use_repo
        self.max_imports = max_imports
        self.max_file_chars = max_file_chars
        self.backend = backend
        _, self._async_client = make_clients(backend)

    def _example_block(self, s: dict) -> str:
        code = " ".join(s["code_tokens"])
        docstring = " ".join(s["docstring_tokens"])
        repo = s.get("repo")
        about: str | None = None
        if repo and self.use_repo:
            about = get_metadata_index().get(repo, {}).get("about")
        path = s.get("path") if self.example_paths else None
        return build_block(code, repo, about, path, docstring)

    def _query_block(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
    ) -> str:
        about: str | None = None
        if project and self.use_repo:
            about = get_metadata_index().get(project, {}).get("about")

        parts: list[str] = []
        if project:
            parts.append(f"Repository: {project}")
        if about:
            parts.append(f"Repository description: {about}")
        if path:
            parts.append(f"File: {path}")

        # File-level context (module doc, enclosing class, imports)
        if language in ("python", "java") and project and path:
            ctx = extract_file_context(
                project, path, code=code, max_imports=self.max_imports, language=language, sha=blame_sha
            )
            if not self.use_outer_context:
                ctx.module_doc = None
                ctx.outer_class_name = None
                ctx.outer_class_doc = None
            if not self.use_class_context:
                ctx.class_name = None
                ctx.class_doc = None
            rendered = render_file_context(ctx)
            if rendered:
                parts.append(rendered)

        # File outline: sibling function signatures + first-line docs
        func_name: str | None = None
        if language in ("python", "java") and project and path:
            func_name = _extract_java_method_name(code) if language == "java" else _extract_func_name_from_code(code)
            outline = extract_file_outline(
                project,
                path,
                exclude_func_name=func_name,
                language=language,
                max_chars=self.max_file_chars,
                cutoff_timestamp=blame_timestamp,
            )
            if outline:
                parts.append(f"Other functions in file:\n{outline}")

        parts.append(f"Code:\n{code}")
        parts.append("Summary: <s>")
        return "\n".join(parts)

    def build_prompt(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
    ) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        blocks = [self._example_block(s) for s in examples]
        blocks.append(self._query_block(code, language, project, path, blame_timestamp=blame_timestamp, blame_sha=blame_sha))
        return "\n\n".join(blocks)

    async def async_summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
    ) -> str:
        prompt = self.build_prompt(code, language, project, path, blame_timestamp=blame_timestamp, blame_sha=blame_sha)
        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=60,
            temperature=0.0,
        )
        raw = response.choices[0].text or ""
        return strip_code_fences(extract_summary(raw))

    def summarize_batch(
        self,
        codes: list[str],
        languages: list[str],
        projects: list[str | None] | None = None,
        paths: list[str | None] | None = None,
        urls: list[str | None] | None = None,
        ground_truths: list[str | None] | None = None,
        blame_timestamps: list[str | None] | None = None,
        blame_shas: list[str | None] | None = None,
    ) -> list[str]:
        import asyncio
        from tqdm.asyncio import tqdm as atqdm

        n = len(codes)
        if projects is None:
            projects = [None] * n
        if paths is None:
            paths = [None] * n
        if urls is None:
            urls = [None] * n
        if blame_timestamps is None:
            blame_timestamps = [None] * n
        if blame_shas is None:
            blame_shas = [None] * n

        async def _gather():
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _one(code, lang, proj, path, url, blame_ts, blame_sha):
                async with sem:
                    return await _call_with_rate_limit_retry(
                        lambda: self.async_summarize(code, lang, proj, path, url, blame_timestamp=blame_ts, blame_sha=blame_sha)
                    )

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url, blame_ts, blame_sha)
                for code, lang, proj, path, url, blame_ts, blame_sha
                in zip(codes, languages, projects, paths, urls, blame_timestamps, blame_shas)
            ], desc="samples")

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        return {
            "model": self.model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "example_paths": self.example_paths,
            "use_outer_context": self.use_outer_context,
            "use_class_context": self.use_class_context,
            "use_repo": self.use_repo,
            "max_imports": self.max_imports,
            "max_file_chars": self.max_file_chars,
            "backend": self.backend,
        }
