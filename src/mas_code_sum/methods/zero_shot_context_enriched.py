"""Zero-shot summarizer enriched with repository context (name, about, README, file path)."""

from ..data import get_metadata_index
from .base import BaseSummarizer, make_clients, strip_code_fences

PROMPT_TEMPLATE = """\
You are summarizing a function from the repository "{repo}".

Repository description: {about}

File: {path}

Please generate a short comment in one sentence for the following function. Output only the summary, no explanation:

{code}
"""

PROMPT_TEMPLATE_NO_CONTEXT = """\
Please generate a short comment in one sentence for the following function. Output only the summary, no explanation:

{code}
"""


class ZeroShotContextEnrichedSummarizer(BaseSummarizer):
    """Zero-shot LLM summarizer enriched with repo name, description, README, and file path."""

    name = "zero_shot_context_enriched"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", backend: str = "featherless"):
        self.model = model
        self.backend = backend
        _, self._async_client = make_clients(backend)

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        if project:
            ctx = get_metadata_index().get(project, {"about": "N/A"})
            prompt = PROMPT_TEMPLATE.format(repo=project, about=ctx["about"], path=path or "unknown", code=code)
        else:
            prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(code=code)
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model}
