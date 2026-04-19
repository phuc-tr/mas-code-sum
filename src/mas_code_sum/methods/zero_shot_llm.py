from .base import BaseSummarizer, make_local_clients, make_openai_clients, strip_code_fences

PROMPT_TEMPLATE = """\
Please generate a short comment in one sentence for the following function. Output only the summary, no explanation:

{code}
"""


class ZeroShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM via OpenRouter or a local HuggingFace model."""

    name = "zero_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", max_concurrency: int = 10, backend: str = "openrouter", device: str = "cuda"):
        self.model = model
        self.max_concurrency = max_concurrency
        self.backend = backend
        if backend == "local":
            self._client, self._async_client = make_local_clients(model, device)
        else:
            self._client, self._async_client = make_openai_clients()

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(code=code)}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(code=code)},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model, "backend": self.backend}
