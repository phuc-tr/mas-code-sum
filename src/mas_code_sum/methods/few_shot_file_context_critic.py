"""Few-shot file-context summarizer with a critic agent that reads the entire file.

Two-stage pipeline:
  1. Summarizer: generates an initial summary using the file-context few-shot prompt
     (completions API with <s> delimiters).
  2. Critic: refines the summary using a structured outline of all other
     functions/classes in the same file (chat completions API).

The file outline is used instead of raw source to keep prompts compact and structured,
which is important when working with smaller/weaker LLMs.
"""

from ..enrichers.file_context import _extract_func_name_from_code, extract_file_context, extract_file_outline, render_file_context
from ..evaluator import bleu as _bleu
from ..retrievers.base import BaseRetriever
from .base import (
    BaseSummarizer,
    CHATTY_CHECK,
    _CHATTY_MAX_RETRIES,
    extract_summary,
    is_chatty,
    make_clients,
    strip_code_fences,
)
from .few_shot_context_enriched import _build_block
from .zero_shot_context_enriched import _get_metadata_index



def _docs_from_outline(outline: str) -> list[str]:
    import re
    return [m.group(1) for m in (re.match(r'^\s+"(.+)"$', line) for line in outline.splitlines()) if m]


def _score(candidate: str, example_ref: str, outline_refs: list[str]) -> float:
    import statistics
    example_score = _bleu([example_ref], candidate)[0]
    if not outline_refs:
        return example_score
    outline_score = statistics.mean(_bleu([ref], candidate)[0] for ref in outline_refs)
    return (example_score + outline_score) / 2


def _pick_best(initial: str, refined: str, example_refs: list[str], outline_refs: list[str]) -> tuple[str, str]:
    """Return (best_summary, source) where source is 'critic' or 'initial'."""
    top1 = example_refs[0]
    if _score(refined, top1, outline_refs) > _score(initial, top1, outline_refs):
        return refined, "critic"
    return initial, "initial"


CRITIC_PROMPT = """\
You are a critic reviewing a code summary.

Repository: {repo}
Repository description: {about}
File: {path}

--- OTHER FUNCTIONS IN THE SAME FILE ---
{file_outline}

--- CODE TO SUMMARIZE ---
{code}

--- INITIAL SUMMARY ---
{initial_summary}

--- TASK ---
Rewrite the summary in one sentence. Requirements:
- Make minimal changes — only fix what is necessary
- Start with a verb, singular or plural depending on the style used in the other functions above (e.g. "Returns"/"Return", "Loads"/"Load")
- Match the documentation style of the other functions above
- Output only the improved summary, no explanation

Improved summary:"""


_STAGE1_MODEL = "meta-llama/Meta-Llama-3.1-8B"
_STAGE1_HF_PROVIDER = "featherless-ai"


class FewShotFileContextCriticSummarizer(BaseSummarizer):
    """File-context few-shot summarizer with a critic agent.

    Stage 1 (summarizer): completions API via featherless-ai + Meta-Llama-3.1-8B (base).
    Stage 2 (critic): chat completions API refines the summary using an outline
    of all other functions/classes in the same file.
    """

    name = "few_shot_file_context_critic"

    def __init__(
        self,
        critic_model: str = "meta-llama/llama-3.1-8b-instruct",
        retriever: BaseRetriever = None,
        example_paths: bool = False,
        use_module_doc: bool = True,
        use_class_context: bool = True,
        max_imports: int = 0,
        max_file_chars: int = 4000,
        backend: str = "openrouter",
        hf_provider: str = "featherless-ai",
    ):
        self.model = _STAGE1_MODEL
        self.critic_model = critic_model
        self.retriever = retriever
        self.example_paths = example_paths
        self.use_module_doc = use_module_doc
        self.use_class_context = use_class_context
        self.max_imports = max_imports
        self.max_file_chars = max_file_chars
        self.backend = backend
        self.hf_provider = hf_provider
        self.max_concurrency = 10
        self._client, self._async_client = make_clients("huggingface", _STAGE1_HF_PROVIDER)
        self._critic_client, self._critic_async_client = make_clients(backend, hf_provider)

    # ------------------------------------------------------------------
    # Stage 1: few-shot file-context summarizer (completions API)
    # ------------------------------------------------------------------

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

        parts: list[str] = []
        if project:
            parts.append(f"Repository: {project}")
        if about:
            parts.append(f"Repository description: {about}")
        if path:
            parts.append(f"File: {path}")

        if language in ("python", "java") and project and path:
            ctx = extract_file_context(
                project, path, code=code, max_imports=self.max_imports, language=language
            )
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

    def _example_refs(self, code: str, language: str, project: str | None, path: str | None) -> list[str]:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        return [" ".join(s["docstring_tokens"]) for s in examples]

    def build_prompt(self, code: str, language: str, project: str | None, path: str | None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        blocks = [self._example_block(s) for s in examples]
        blocks.append(self._query_block(code, language, project, path))
        return "\n\n".join(blocks)

    def _run_stage1(self, code: str, language: str, project: str | None, path: str | None) -> str:
        prompt = self.build_prompt(code, language, project, path)
        for _ in range(_CHATTY_MAX_RETRIES + 1):
            response = self._client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
            raw = response.choices[0].text or ""
            comment = strip_code_fences(extract_summary(raw))
            if not CHATTY_CHECK or not is_chatty(comment):
                return comment
        return comment

    async def _run_stage1_async(self, code: str, language: str, project: str | None, path: str | None) -> str:
        prompt = self.build_prompt(code, language, project, path)
        for _ in range(_CHATTY_MAX_RETRIES + 1):
            response = await self._async_client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
            raw = response.choices[0].text or ""
            comment = strip_code_fences(extract_summary(raw))
            if not CHATTY_CHECK or not is_chatty(comment):
                return comment
        return comment

    # ------------------------------------------------------------------
    # Stage 2: critic (chat completions API)
    # ------------------------------------------------------------------

    def _run_critic(
        self,
        initial_summary: str,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
    ) -> tuple[str, str]:
        about: str | None = None
        if project:
            about = _get_metadata_index().get(project, {}).get("about")

        func_name = _extract_func_name_from_code(code)
        file_outline = ""
        if project and path:
            file_outline = extract_file_outline(
                project,
                path,
                exclude_func_name=func_name,
                language=language,
                max_chars=self.max_file_chars,
            )

        critic_prompt = CRITIC_PROMPT.format(
            repo=project or "",
            about=about or "",
            path=path or "",
            file_outline=file_outline or "(not available)",
            code=code,
            initial_summary=initial_summary,
        )
        response = self._critic_client.chat.completions.create(
            model=self.critic_model,
            messages=[{"role": "user", "content": critic_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content or ""), file_outline

    async def _run_critic_async(
        self,
        initial_summary: str,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
    ) -> tuple[str, str]:
        about: str | None = None
        if project:
            about = _get_metadata_index().get(project, {}).get("about")

        func_name = _extract_func_name_from_code(code)
        file_outline = ""
        if project and path:
            file_outline = extract_file_outline(
                project,
                path,
                exclude_func_name=func_name,
                language=language,
                max_chars=self.max_file_chars,
            )

        critic_prompt = CRITIC_PROMPT.format(
            repo=project or "",
            about=about or "",
            path=path or "",
            file_outline=file_outline or "(not available)",
            code=code,
            initial_summary=initial_summary,
        )
        response = await self._critic_async_client.chat.completions.create(
            model=self.critic_model,
            messages=[{"role": "user", "content": critic_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content or ""), file_outline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _summarize_one(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        url: str | None,
        ground_truth: str | None,
    ) -> tuple[str, str, str, str]:
        """Return (prediction, source, initial, refined) where source is 'critic' or 'initial'."""
        initial = await self._run_stage1_async(code, language, project, path)
        refined, file_outline = await self._run_critic_async(initial, code, language, project, path)
        example_refs = self._example_refs(code, language, project, path)
        outline_refs = _docs_from_outline(file_outline)
        pred, source = _pick_best(initial, refined, example_refs, outline_refs)
        return pred, source, initial, refined

    def summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        ground_truth: str | None = None,
    ) -> str:
        initial = self._run_stage1(code, language, project, path)
        refined, file_outline = self._run_critic(initial, code, language, project, path)
        example_refs = self._example_refs(code, language, project, path)
        outline_refs = _docs_from_outline(file_outline)
        pred, _ = _pick_best(initial, refined, example_refs, outline_refs)
        return pred

    async def async_summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        ground_truth: str | None = None,
    ) -> str:
        pred, _, _initial, _refined = await self._summarize_one(code, language, project, path, url, ground_truth)
        return pred

    def summarize_batch(
        self,
        codes: list[str],
        languages: list[str],
        projects: list[str | None] | None = None,
        paths: list[str | None] | None = None,
        urls: list[str | None] | None = None,
        ground_truths: list[str | None] | None = None,
    ) -> list[str]:
        """Batch summarization with optional per-sample ground truths for oracle selection.

        Populates self.last_sources with 'critic' or 'initial' for each prediction.
        """
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

        async def _gather():
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _one(code, lang, proj, path, url, gt):
                async with sem:
                    return await self._summarize_one(code, lang, proj, path, url, gt)

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url, gt)
                for code, lang, proj, path, url, gt
                in zip(codes, languages, projects, paths, urls, ground_truths)
            ], desc="samples")

        results: list[tuple[str, str, str, str]] = list(asyncio.run(_gather()))
        if results:
            preds, sources, initials, refineds = zip(*results)
        else:
            preds, sources, initials, refineds = [], [], [], []
        self.last_sources: list[str] = list(sources)
        self.last_initials: list[str] = list(initials)
        self.last_refineds: list[str] = list(refineds)
        return list(preds)

    def params(self) -> dict:
        return {
            "stage1_model": self.model,
            "stage1_backend": f"huggingface/{_STAGE1_HF_PROVIDER}",
            "critic_model": self.critic_model,
            "critic_backend": self.backend,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "example_paths": self.example_paths,
            "use_module_doc": self.use_module_doc,
            "use_class_context": self.use_class_context,
            "max_imports": self.max_imports,
            "max_file_chars": self.max_file_chars,
        }
