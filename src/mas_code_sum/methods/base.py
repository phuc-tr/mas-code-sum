"""Abstract base class for summarization methods."""

import asyncio
import re
from abc import ABC, abstractmethod


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences and triple-quotes from LLM output."""
    text = text.strip()
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = re.sub(r'^"""\n?', "", text)
    text = re.sub(r'\n?"""$', "", text)
    return text.strip()


class BaseSummarizer(ABC):
    """All experiment methods must implement this interface."""

    name: str  # used as MLflow run tag and artifact prefix

    @abstractmethod
    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        """Generate a one-line summary for the given code snippet."""
        ...

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        """Async version of summarize. Default: run summarize in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.summarize(code, language, project=project, path=path, url=url))

    def summarize_batch(
        self,
        codes: list[str],
        languages: list[str],
        projects: list[str | None] | None = None,
        paths: list[str | None] | None = None,
        urls: list[str | None] | None = None,
    ) -> list[str]:
        """Summarize a batch of code snippets concurrently via async_summarize."""
        n = len(codes)
        if projects is None:
            projects = [None] * n
        if paths is None:
            paths = [None] * n
        if urls is None:
            urls = [None] * n

        max_concurrency = getattr(self, "max_concurrency", 32)

        async def _gather():
            sem = asyncio.Semaphore(max_concurrency)

            async def _one(code, lang, proj, path, url):
                async with sem:
                    return await self.async_summarize(code, lang, project=proj, path=path, url=url)

            return await asyncio.gather(*[
                _one(code, lang, proj, path, url)
                for code, lang, proj, path, url in zip(codes, languages, projects, paths, urls)
            ])

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        """Return method hyperparameters to log in MLflow."""
        return {}
