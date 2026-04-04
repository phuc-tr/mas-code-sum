"""Few-shot summarizer enriched with repository context (name, about, file path)."""

import json
import os
from pathlib import Path

from openai import OpenAI

from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, strip_code_fences
from .zero_shot_context_enriched import _get_metadata_index

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {docstring}"""

PROMPT_TEMPLATE = """\
You are summarizing a function from the repository "{repo}".

Repository description: {about}

File: {path}

Here are some examples of code summarization:

{examples}

Now summarize the following code in one sentence. Output only the summary, no explanation:

Code:
{code}
Summary:"""

PROMPT_TEMPLATE_NO_CONTEXT = """\
Here are some examples of code summarization:

{examples}

Now summarize the following code in one sentence. Output only the summary, no explanation:

Code:
{code}
Summary:"""


class FewShotContextEnrichedSummarizer(BaseSummarizer):
    """Few-shot LLM summarizer enriched with repo name, description, and file path."""

    name = "few_shot_context_enriched"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", retriever: BaseRetriever = None):
        self.model = model
        self.retriever = retriever
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        examples_str = "\n\n".join(example_blocks)

        if project:
            ctx = _get_metadata_index().get(project, {"about": "N/A"})
            prompt = PROMPT_TEMPLATE.format(
                repo=project,
                about=ctx["about"],
                path=path or "unknown",
                examples=examples_str,
                code=code,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(examples=examples_str, code=code)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n}
