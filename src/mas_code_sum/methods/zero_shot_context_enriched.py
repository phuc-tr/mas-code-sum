"""Zero-shot summarizer enriched with repository context (name, about, README, file path)."""

import json
from pathlib import Path

from .base import BaseSummarizer, make_openai_clients, strip_code_fences

METADATA_PATH = Path(__file__).parents[3] / "dataset" / "repo_metadata" / "all_repo_metadata.json"

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


def _load_metadata_index() -> dict[str, dict]:
    """Load all_repo_metadata.json and index by repo name."""
    with open(METADATA_PATH) as f:
        data = json.load(f)
    index: dict[str, dict] = {}
    for entries in data.values():
        for entry in entries:
            index[entry["repo"]] = {
                "about": entry.get("about") or "N/A",
            }
    return index


_METADATA_INDEX: dict[str, dict] | None = None


def _get_metadata_index() -> dict[str, dict]:
    global _METADATA_INDEX
    if _METADATA_INDEX is None:
        _METADATA_INDEX = _load_metadata_index()
    return _METADATA_INDEX


class ZeroShotContextEnrichedSummarizer(BaseSummarizer):
    """Zero-shot LLM summarizer enriched with repo name, description, README, and file path."""

    name = "zero_shot_context_enriched"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", max_concurrency: int = 10):
        self.model = model
        self.max_concurrency = max_concurrency
        self._client, self._async_client = make_openai_clients()

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        if project:
            ctx = _get_metadata_index().get(project, {"about": "N/A"})
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

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        if project:
            ctx = _get_metadata_index().get(project, {"about": "N/A"})
            prompt = PROMPT_TEMPLATE.format(
                repo=project,
                about=ctx["about"],
                path=path or "unknown",
                code=code,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(code=code)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model}
