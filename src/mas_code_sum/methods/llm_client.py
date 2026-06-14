"""LLM client factories and retry logic for OpenAI-compatible backends."""

import asyncio
import logging
import os

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, OpenAI, RateLimitError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
_LLM_TIMEOUT = 60.0
_LLM_MAX_RETRIES = 3

_RATE_LIMIT_INITIAL_WAIT = 5.0
_RATE_LIMIT_MAX_RETRIES = 6


def _is_capacity_error(exc: InternalServerError) -> bool:
    return exc.status_code == 503 or "capacity" in str(exc).lower()


async def _call_with_rate_limit_retry(coro_factory):
    """Call *coro_factory* and await the result, retrying on transient API errors."""
    wait = _RATE_LIMIT_INITIAL_WAIT
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except (RateLimitError, InternalServerError, APIConnectionError) as exc:
            if isinstance(exc, InternalServerError) and not _is_capacity_error(exc):
                raise
            if attempt == _RATE_LIMIT_MAX_RETRIES:
                raise
            logging.warning("%s; retrying in %.0fs (attempt %d/%d)", type(exc).__name__, wait, attempt + 1, _RATE_LIMIT_MAX_RETRIES)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 300)


def make_openai_clients() -> tuple[OpenAI, AsyncOpenAI]:
    kwargs = dict(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


def make_featherless_clients() -> tuple[OpenAI, AsyncOpenAI]:
    kwargs = dict(
        api_key=os.environ["FEATHERLESS_API_KEY"],
        base_url=FEATHERLESS_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


def make_clients(backend: str = "featherless") -> tuple[OpenAI, AsyncOpenAI]:
    if backend == "openrouter":
        return make_openai_clients()
    if backend == "featherless":
        return make_featherless_clients()
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'openrouter' or 'featherless'.")
