"""Abstract base class for summarization methods."""

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod

from openai import AsyncOpenAI, OpenAI, RateLimitError
from tqdm.asyncio import tqdm as atqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
_LLM_TIMEOUT = 60.0
_LLM_MAX_RETRIES = 3

_RATE_LIMIT_INITIAL_WAIT = 5.0
_RATE_LIMIT_MAX_RETRIES = 6


async def _call_with_rate_limit_retry(coro_factory):
    """Call *coro_factory* and await the result, retrying on RateLimitError."""
    wait = _RATE_LIMIT_INITIAL_WAIT
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except RateLimitError:
            if attempt == _RATE_LIMIT_MAX_RETRIES:
                raise
            logging.warning("Rate limited; retrying in %.0fs (attempt %d/%d)", wait, attempt + 1, _RATE_LIMIT_MAX_RETRIES)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 300)

def extract_summary(raw: str) -> str:
    end = raw.find("</s>")
    return raw[:end].strip() if end != -1 else raw.split("\n")[0].strip()


def make_openai_clients() -> tuple[OpenAI, AsyncOpenAI]:
    """Create sync and async OpenAI clients pointed at OpenRouter."""
    kwargs = dict(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


def make_featherless_clients() -> tuple[OpenAI, AsyncOpenAI]:
    """Create sync and async OpenAI clients pointed at Featherless API."""
    kwargs = dict(
        api_key=os.environ["FEATHERLESS_API_KEY"],
        base_url=FEATHERLESS_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


def make_clients(backend: str = "featherless"):
    """Return OpenAI-compatible clients for the specified backend."""
    if backend == "openrouter":
        return make_openai_clients()
    if backend == "featherless":
        return make_featherless_clients()
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'openrouter' or 'featherless'.")


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences, triple-quotes, and <think> blocks from LLM output."""
    text = text.strip()
    # Strip <think>...</think> block (may start with </think> if opening tag was truncated)
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^</think>\s*", "", text)
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
                    return await _call_with_rate_limit_retry(
                        lambda: self.async_summarize(code, lang, project=proj, path=path, url=url)
                    )

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url)
                for code, lang, proj, path, url in zip(codes, languages, projects, paths, urls)
            ], desc="samples")

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        """Return method hyperparameters to log in MLflow."""
        return {}
