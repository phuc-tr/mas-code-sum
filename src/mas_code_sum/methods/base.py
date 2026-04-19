"""Abstract base class for summarization methods."""

import asyncio
import os
import re
from abc import ABC, abstractmethod

from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as atqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_LLM_TIMEOUT = 60.0
_LLM_MAX_RETRIES = 3


def make_openai_clients() -> tuple[OpenAI, AsyncOpenAI]:
    """Create sync and async OpenAI clients pointed at OpenRouter with shared timeout/retry config."""
    kwargs = dict(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Local HuggingFace backend — thin wrappers that mimic the OpenAI client API
# ---------------------------------------------------------------------------

class _LocalTextResponse:
    """Mimics openai.types.Completion."""
    def __init__(self, text: str):
        self.choices = [type("_Choice", (), {"text": text})()]


class _LocalChatResponse:
    """Mimics openai.types.ChatCompletion."""
    def __init__(self, content: str):
        msg = type("_Msg", (), {"content": content})()
        self.choices = [type("_Choice", (), {"message": msg})()]


class _LocalSyncCompletions:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def create(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0, **_) -> _LocalTextResponse:
        do_sample = temperature > 0
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            return_full_text=False,
        )
        return _LocalTextResponse(outputs[0]["generated_text"])


class _LocalSyncChatCompletions:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def create(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.0, **_) -> _LocalChatResponse:
        do_sample = temperature > 0
        outputs = self._pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
        )
        content = outputs[0]["generated_text"][-1]["content"]
        return _LocalChatResponse(content)


class _LocalSyncChat:
    def __init__(self, pipeline):
        self.completions = _LocalSyncChatCompletions(pipeline)


class _LocalSyncClient:
    """Sync local client. Has .completions and .chat.completions interfaces."""
    def __init__(self, pipeline):
        self.completions = _LocalSyncCompletions(pipeline)
        self.chat = _LocalSyncChat(pipeline)


class _LocalAsyncCompletions:
    def __init__(self, sync: _LocalSyncCompletions):
        self._sync = sync

    async def create(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0, **_) -> _LocalTextResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync.create(prompt, max_tokens=max_tokens, temperature=temperature))


class _LocalAsyncChatCompletions:
    def __init__(self, sync: _LocalSyncChatCompletions):
        self._sync = sync

    async def create(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.0, **_) -> _LocalChatResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync.create(messages, max_tokens=max_tokens, temperature=temperature))


class _LocalAsyncChat:
    def __init__(self, sync_chat: _LocalSyncChat):
        self.completions = _LocalAsyncChatCompletions(sync_chat.completions)


class _LocalAsyncClient:
    """Async local client. Has .completions and .chat.completions interfaces."""
    def __init__(self, sync_client: _LocalSyncClient):
        self.completions = _LocalAsyncCompletions(sync_client.completions)
        self.chat = _LocalAsyncChat(sync_client.chat)


def make_local_clients(model_id: str, device: str = "cuda") -> tuple[_LocalSyncClient, _LocalAsyncClient]:
    """Load a HuggingFace text-generation pipeline and return sync/async local clients.

    The returned clients expose the same .completions.create() and
    .chat.completions.create() interfaces as the OpenAI client, so methods
    can be switched to local GPU inference with no other code changes.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        device: device_map value passed to the pipeline (e.g. "cuda", "cpu", "auto").
    """
    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        device_map=device,
        torch_dtype="auto",
    )
    sync = _LocalSyncClient(pipe)
    return sync, _LocalAsyncClient(sync)


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
                    return await self.async_summarize(code, lang, project=proj, path=path, url=url)

            return await atqdm.gather(*[
                _one(code, lang, proj, path, url)
                for code, lang, proj, path, url in zip(codes, languages, projects, paths, urls)
            ], desc="samples")

        return list(asyncio.run(_gather()))

    def params(self) -> dict:
        """Return method hyperparameters to log in MLflow."""
        return {}
