"""Agentic RAG summarizer: a context-gatherer agent feeding a few-shot summarizer.

Two agents per sample:
  1. Gatherer — the target repo is chunked with AST-aware chunking
     (astchunk, see `enrichers/ast_chunks.py`), BM25 pre-ranks candidate
     chunks against the target code, and — when `use_filter=True` (default)
     — an LLM agent (chat call) inspects the candidates and decides which
     ones actually help explain the target function, keeping up to
     `n_context` of the `n_candidates` BM25 hits. When `use_filter=False`,
     this LLM call is skipped entirely and all `n_candidates` BM25 hits are
     passed straight through as context (no narrowing).
  2. Summarizer — an instructed chat call: a "You are..." preamble, retrieved
     few-shot examples, the query's repo/file info plus the gatherer's chunks
     as context, and a closing instruction to output only the summary.

Prompt structure:
  You are ... (task instruction)
  Repository: {repo}
  Repository description: {about}
  File: {path}
  Relevant repository context:
  {filepath} (lines a-b):
  {chunk}
  ...
  Here are examples ...
  <example blocks: Repository/File/Code/Summary: {docstring}>
  Now summarize ...
  Code:
  {code}
  Output only the one-sentence summary ...

Leakage control: candidate chunks are dropped by content, not by file — any
chunk containing the target function's own code is excluded (so other chunks
from the same file remain eligible), and — since repos contain duplicated
functions and files move between the dataset snapshot and HEAD — any chunk
containing the reference docstring itself is dropped too (the runner supplies
`ground_truths`; they are used only for this exclusion and never shown to
either agent).

Thinking models: `<think>...</think>` blocks are stripped from both agents'
outputs before parsing, and the gatherer's token budget is sized so the answer
survives a reasoning preamble.
"""

import re

from ..data import get_metadata_index
from ..enrichers.ast_chunks import Chunk, get_chunk_index
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_clients, strip_code_fences

GATHERER_PROMPT = """\
You are a context-gathering agent helping to document a function. Below is the \
target function, followed by numbered code chunks from elsewhere in the same \
repository.

Target function:
{code}

Candidate chunks:
{chunks}

Select the chunks that genuinely help explain what the target function does \
(e.g. definitions of things it calls, related classes, usage sites). End your \
reply with a single line containing only the chunk numbers, comma-separated \
(e.g. "1, 3"), or "NONE" if no chunk is relevant.
"""

SUMMARIZER_INSTRUCTION = """\
You are an expert code documentation assistant. Your task is to write a short, \
one-sentence summary describing what a given function does, matching the style \
and tone of the examples below.
"""

SUMMARIZER_FINAL_INSTRUCTION = """\
Output only the one-sentence summary of the function above. Do not output any \
explanation, reasoning, or extra text."""

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (and a dangling open block)."""
    text = _THINK_RE.sub("", text)
    # Unclosed <think> (budget ran out mid-reasoning): drop everything after it.
    if "<think>" in text:
        text = text[: text.index("<think>")]
    return text.strip()


def clean_summary_reply(raw: str) -> str:
    """Normalize an instructed chat model's summary reply to a single sentence.

    Strips think blocks and code fences, then — since even instructed models
    sometimes prefix a label or add trailing chatter — keeps the first
    non-empty line and drops a leading "Summary:".
    """
    raw = strip_code_fences(strip_think(raw))
    first_line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
    return re.sub(r"^summary\s*:\s*", "", first_line, flags=re.IGNORECASE)


class AgenticRagSummarizer(BaseSummarizer):
    """Few-shot ICL summarizer whose query block carries agent-gathered AST chunks."""

    name = "agentic_rag"

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        retriever: BaseRetriever = None,
        example_paths: bool = False,
        use_repo: bool = True,
        backend: str = "featherless",
        gatherer_model: str | None = None,
        use_filter: bool = True,
        n_candidates: int = 10,
        n_context: int = 4,
        max_chunk_size: int = 1200,
        max_chunk_chars: int = 1500,
        gatherer_max_tokens: int = 2048,
        summarizer_max_tokens: int = 2048,
    ):
        self.model = model
        self.retriever = retriever
        self.example_paths = example_paths
        self.use_repo = use_repo
        self.backend = backend
        self.gatherer_model = gatherer_model or model
        self.use_filter = use_filter  # if False, skip the gatherer agent and use all BM25 candidates as-is
        self.n_candidates = n_candidates  # BM25 pre-ranked chunks shown to the gatherer
        self.n_context = n_context  # max chunks the gatherer may keep
        self.max_chunk_size = max_chunk_size  # astchunk max non-whitespace chars per chunk
        self.max_chunk_chars = max_chunk_chars  # truncation cap when rendering a chunk
        # Token budgets are roomy because thinking models reason before answering.
        self.gatherer_max_tokens = gatherer_max_tokens
        self.summarizer_max_tokens = summarizer_max_tokens
        _, self._async_client = make_clients(backend)

    # ------------------------------------------------------------------ gatherer

    def _render_chunk(self, chunk: Chunk) -> str:
        content = chunk.content
        if len(content) > self.max_chunk_chars:
            content = content[: self.max_chunk_chars].rstrip() + "\n..."
        return f"{chunk.filepath} (lines {chunk.start_line}-{chunk.end_line}):\n{content}"

    @staticmethod
    def _parse_selection(raw: str, n_candidates: int) -> list[int]:
        """Parse the gatherer's reply into 0-based candidate indices."""
        raw = strip_think(raw)
        # The instruction asks for the selection on the final line.
        lines = [l for l in raw.splitlines() if l.strip()]
        answer = lines[-1] if lines else ""
        if "none" in answer.lower():
            return []
        picked: list[int] = []
        for m in re.finditer(r"\d+", answer):
            idx = int(m.group()) - 1
            if 0 <= idx < n_candidates and idx not in picked:
                picked.append(idx)
        return picked

    async def _gather_context(
        self, code: str, language: str, project: str, ground_truth: str | None = None
    ) -> list[Chunk]:
        index = get_chunk_index(project, language, max_chunk_size=self.max_chunk_size)
        candidates = index.query(code, k=self.n_candidates, exclude_text=ground_truth)
        if not candidates:
            return []

        if not self.use_filter:
            return candidates

        rendered = "\n\n".join(
            f"[{i + 1}] {self._render_chunk(c)}" for i, c in enumerate(candidates)
        )
        response = await self._async_client.chat.completions.create(
            model=self.gatherer_model,
            messages=[{"role": "user", "content": GATHERER_PROMPT.format(code=code, chunks=rendered)}],
            max_tokens=self.gatherer_max_tokens,
            temperature=0.0,
        )
        reply = response.choices[0].message.content or ""
        picked = self._parse_selection(reply, len(candidates))
        return [candidates[i] for i in picked[: self.n_context]]

    # ---------------------------------------------------------------- summarizer

    def _example_block(self, s: dict) -> str:
        code = " ".join(s["code_tokens"])
        docstring = " ".join(s["docstring_tokens"])
        repo = s.get("repo")
        about: str | None = None
        if repo and self.use_repo:
            about = get_metadata_index().get(repo, {}).get("about")
        path = s.get("path") if self.example_paths else None

        parts: list[str] = []
        if repo:
            parts.append(f"Repository: {repo}")
        if about:
            parts.append(f"Repository description: {about}")
        if path:
            parts.append(f"File: {path}")
        parts.append(f"Code:\n{code}")
        parts.append(f"Summary: {docstring}")
        return "\n".join(parts)

    def _context_block(self, project: str | None, path: str | None, chunks: list[Chunk]) -> str:
        """Query-side context (repo/description/file/gathered chunks), without the code."""
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

    def build_prompt(self, code: str, language: str, project: str | None, path: str | None, chunks: list[Chunk]) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        sections = [SUMMARIZER_INSTRUCTION]
        context = self._context_block(project, path, chunks)
        if context:
            sections.append(context)
        if examples:
            sections.append("Here are examples of well-summarized functions:")
            sections.extend(self._example_block(s) for s in examples)
        sections.append(f"Now summarize the following function:\n\nCode:\n{code}")
        sections.append(SUMMARIZER_FINAL_INSTRUCTION)
        return "\n\n".join(sections)

    async def async_summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        ground_truth: str | None = None,
    ) -> str:
        chunks: list[Chunk] = []
        if project:
            chunks = await self._gather_context(code, language, project, ground_truth=ground_truth)

        prompt = self.build_prompt(code, language, project, path, chunks)
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
    ) -> list[str]:
        import asyncio
        from tqdm.asyncio import tqdm as atqdm

        from .base import _call_with_rate_limit_retry

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
                    return await _call_with_rate_limit_retry(
                        lambda: self.async_summarize(code, lang, project=proj, path=path, url=url, ground_truth=gt)
                    )

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url, gt)
                for code, lang, proj, path, url, gt
                in zip(codes, languages, projects, paths, urls, ground_truths)
            ], desc="samples")

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        return {
            "model": self.model,
            "gatherer_model": self.gatherer_model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "example_paths": self.example_paths,
            "use_repo": self.use_repo,
            "backend": self.backend,
            "use_filter": self.use_filter,
            "n_candidates": self.n_candidates,
            "n_context": self.n_context,
            "max_chunk_size": self.max_chunk_size,
            "max_chunk_chars": self.max_chunk_chars,
            "gatherer_max_tokens": self.gatherer_max_tokens,
            "summarizer_max_tokens": self.summarizer_max_tokens,
        }
