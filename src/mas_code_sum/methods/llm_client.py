"""LLM client factories and retry logic for OpenAI-compatible backends."""

import asyncio
import logging
import os
import threading

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, OpenAI, RateLimitError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

# Restrict OpenRouter requests to these upstream providers only (no fallback to others).
OPENROUTER_ALLOWED_PROVIDERS = ["Novita", "DeepInfra", "openai", "azure"]
_LLM_TIMEOUT = 60.0
_LLM_MAX_RETRIES = 3

_RATE_LIMIT_INITIAL_WAIT = 5.0
_RATE_LIMIT_MAX_RETRIES = 6

# Per-backend concurrency ceilings. OpenRouter tolerates higher parallelism;
# Featherless rate-limits aggressively, so we stay conservative there.
_CONCURRENCY_BY_BACKEND = {
    "openrouter": 10,
    "featherless": 2,
}
_DEFAULT_CONCURRENCY = 2


def get_concurrency(backend: str | None) -> int:
    """Max concurrent in-flight requests for *backend*. Not user-configurable."""
    return _CONCURRENCY_BY_BACKEND.get(backend, _DEFAULT_CONCURRENCY)


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


class CostTracker:
    """Accumulates actual USD spend reported by OpenRouter's usage accounting.

    OpenRouter returns the exact, post-discount dollar cost of each generation
    in `response.usage.cost` when the request opts in via `extra_body={"usage":
    {"include": True}}`. Token-count-based estimates can't reproduce
    per-model/provider pricing (and promos), so this is the only accurate source.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0.0

    def add(self, amount: float) -> None:
        with self._lock:
            self._total += amount

    def reset(self) -> None:
        with self._lock:
            self._total = 0.0

    @property
    def total(self) -> float:
        with self._lock:
            return self._total


cost_tracker = CostTracker()


def _record_cost(response):
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    if cost is not None:
        cost_tracker.add(cost)
    return response


def _with_openrouter_extra_body(extra_body):
    extra_body = dict(extra_body or {})
    extra_body.setdefault("usage", {"include": True})
    provider = dict(extra_body.get("provider") or {})
    provider.setdefault("order", OPENROUTER_ALLOWED_PROVIDERS)
    provider.setdefault("allow_fallbacks", False)
    extra_body["provider"] = provider
    return extra_body


def _enable_usage_accounting(client: OpenAI, async_client: AsyncOpenAI) -> None:
    """Patch both clients' `chat.completions.create` and `completions.create` to
    request/record OpenRouter cost and restrict requests to the allowed providers."""

    for target_client in (client, async_client):
        orig_chat_create = target_client.chat.completions.create
        orig_create = target_client.completions.create
        is_async = target_client is async_client

        if is_async:
            async def chat_create(*args, _orig=orig_chat_create, **kwargs):
                kwargs["extra_body"] = _with_openrouter_extra_body(kwargs.get("extra_body"))
                return _record_cost(await _orig(*args, **kwargs))

            async def create(*args, _orig=orig_create, **kwargs):
                kwargs["extra_body"] = _with_openrouter_extra_body(kwargs.get("extra_body"))
                return _record_cost(await _orig(*args, **kwargs))
        else:
            def chat_create(*args, _orig=orig_chat_create, **kwargs):
                kwargs["extra_body"] = _with_openrouter_extra_body(kwargs.get("extra_body"))
                return _record_cost(_orig(*args, **kwargs))

            def create(*args, _orig=orig_create, **kwargs):
                kwargs["extra_body"] = _with_openrouter_extra_body(kwargs.get("extra_body"))
                return _record_cost(_orig(*args, **kwargs))

        target_client.chat.completions.create = chat_create
        target_client.completions.create = create


def make_openai_clients() -> tuple[OpenAI, AsyncOpenAI]:
    kwargs = dict(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    client, async_client = OpenAI(**kwargs), AsyncOpenAI(**kwargs)
    _enable_usage_accounting(client, async_client)
    return client, async_client


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
