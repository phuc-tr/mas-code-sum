"""Few-shot summarizer with a critic agent that refines the initial summary."""

import os

from openai import OpenAI

from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, strip_code_fences
from .zero_shot_context_enriched import _get_metadata_index

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {docstring}"""

SUMMARIZER_PROMPT = """\
You are summarizing a function from the repository "{repo}".

Repository description: {about}

File: {path}

Here are some examples of code summarization:

{examples}

Now summarize the following code in one sentence. Output only the summary, no explanation:

Code:
{code}
Summary:"""

CRITIC_PROMPT = """\
You are a critic reviewing a code summary from the repository "{repo}".

Repository description: {about}

File: {path}

Here are examples showing the expected summarization style and convention:

{examples}

The following code was summarized:

Code:
{code}

Initial summary: {summary}

Improve the summary to better match the style and convention of the examples above. Output only the improved summary in one sentence, no explanation:"""


class FewShotCriticSummarizer(BaseSummarizer):
    """Few-shot summarizer with a critic agent that refines the initial summary.

    Two-agent pipeline:
    1. Summarizer: generates an initial summary using few-shot examples and context.
    2. Critic: refines the summary to better match the style of the examples.
    """

    name = "few_shot_critic"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        critic_model: str | None = None,
        retriever: BaseRetriever = None,
    ):
        self.model = model
        self.critic_model = critic_model or model
        self.retriever = retriever
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str, project: str, path: str, url: str | None = None) -> str:
        ctx = _get_metadata_index()[project]
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        examples_str = "\n\n".join(example_blocks)

        # --- Summarizer agent ---
        summarizer_prompt = SUMMARIZER_PROMPT.format(
            repo=project,
            about=ctx["about"],
            path=path,
            examples=examples_str,
            code=code,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summarizer_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        initial_summary = strip_code_fences(response.choices[0].message.content)

        # --- Critic agent ---
        critic_prompt = CRITIC_PROMPT.format(
            repo=project,
            about=ctx["about"],
            path=path,
            examples=examples_str,
            code=code,
            summary=initial_summary,
        )
        critic_response = self._client.chat.completions.create(
            model=self.critic_model,
            messages=[{"role": "user", "content": critic_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(critic_response.choices[0].message.content)

    def params(self) -> dict:
        return {
            "model": self.model,
            "critic_model": self.critic_model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
        }
