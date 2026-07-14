"""Agentic RAG summarizer enriched with the target file's own local context.

Extends `AgenticRagSummarizer` (cross-repo AST-chunk retrieval + few-shot
examples, see `agentic_rag.py`) by also injecting the file-level signals
`few_shot_all_context` uses — module docstring / enclosing class (Java: outer
class too) / imports, plus an outline of sibling functions in the same file
(see `enrichers/file_context.py`). These come from a plain file parse, not an
extra agent call, and are placed right before the target function, after the
few-shot examples — closest in the prompt to the code it describes.
"""

from ..data import get_metadata_index
from ..enrichers.ast_chunks import Chunk
from ..enrichers.file_context import (
    _extract_func_name_from_code,
    extract_file_context,
    extract_file_outline,
    render_file_context,
)
from ..enrichers.file_context_java import _extract_method_name_from_code as _extract_java_method_name
from .agentic_rag import SUMMARIZER_FINAL_INSTRUCTION, SUMMARIZER_INSTRUCTION, AgenticRagSummarizer, clean_summary_reply
from .base import _call_with_rate_limit_retry


class AgenticRagAllContextSummarizer(AgenticRagSummarizer):
    """AgenticRagSummarizer + local file context (module doc/class/imports/outline)."""

    name = "agentic_rag_all_context"

    def __init__(
        self,
        *args,
        use_outer_context: bool = True,
        use_class_context: bool = True,
        max_imports: int = 0,
        max_file_chars: int = 4000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_outer_context = use_outer_context
        self.use_class_context = use_class_context
        self.max_imports = max_imports
        self.max_file_chars = max_file_chars

    # ------------------------------------------------------------ file context

    def _file_context_parts(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        blame_timestamp: str | None,
        blame_sha: str | None,
    ) -> list[str]:
        if language not in ("python", "java") or not project or not path:
            return []

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

        parts: list[str] = []
        rendered = render_file_context(ctx)
        if rendered:
            parts.append(rendered)

        func_name = _extract_java_method_name(code) if language == "java" else _extract_func_name_from_code(code)
        outline = extract_file_outline(
            project,
            path,
            exclude_func_name=func_name,
            language=language,
            max_chars=self.max_file_chars,
            cutoff_timestamp=blame_timestamp,
            sha=blame_sha,
        )
        if outline:
            parts.append(f"Other functions in file:\n{outline}")

        return parts

    # ---------------------------------------------------------------- prompt

    def _context_block(self, project: str | None, path: str | None, chunks: list[Chunk]) -> str:
        """Query-side context: repo/file lines, then the gatherer's cross-repo chunks."""
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
        if chunks:
            rendered = "\n".join(self._render_chunk(c) for c in chunks)
            parts.append(f"Relevant repository context:\n{rendered}")
        return "\n".join(parts)

    def build_prompt(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        chunks: list[Chunk],
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
    ) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        sections = [SUMMARIZER_INSTRUCTION]
        context = self._context_block(project, path, chunks)
        if context:
            sections.append(context)
        if examples:
            sections.append("Here are examples of well-summarized functions:")
            sections.extend(self._example_block(s) for s in examples)
        sections.extend(self._file_context_parts(code, language, project, path, blame_timestamp, blame_sha))
        sections.append(f"Now summarize the following function:\n\nCode:\n{code}")
        sections.append(SUMMARIZER_FINAL_INSTRUCTION)
        return "\n\n".join(sections)

    # ------------------------------------------------------------- pipeline

    async def async_summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        ground_truth: str | None = None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
    ) -> str:
        chunks: list[Chunk] = []
        if project:
            chunks = await self._gather_context(code, language, project, ground_truth=ground_truth)

        prompt = self.build_prompt(code, language, project, path, chunks, blame_timestamp, blame_sha)
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.summarizer_max_tokens,
            temperature=0.0,
        )
        return clean_summary_reply(response.choices[0].message.content or "")

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
        if ground_truths is None:
            ground_truths = [None] * n
        if blame_timestamps is None:
            blame_timestamps = [None] * n
        if blame_shas is None:
            blame_shas = [None] * n

        async def _gather():
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _one(code, lang, proj, path, url, gt, blame_ts, blame_sha):
                async with sem:
                    return await _call_with_rate_limit_retry(
                        lambda: self.async_summarize(
                            code, lang, project=proj, path=path, url=url, ground_truth=gt,
                            blame_timestamp=blame_ts, blame_sha=blame_sha,
                        )
                    )

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url, gt, blame_ts, blame_sha)
                for code, lang, proj, path, url, gt, blame_ts, blame_sha
                in zip(codes, languages, projects, paths, urls, ground_truths, blame_timestamps, blame_shas)
            ], desc="samples")

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        return {
            **super().params(),
            "use_outer_context": self.use_outer_context,
            "use_class_context": self.use_class_context,
            "max_imports": self.max_imports,
            "max_file_chars": self.max_file_chars,
        }
