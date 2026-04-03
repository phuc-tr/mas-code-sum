import os

from openai import OpenAI

from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_TEMPLATE = """\
Here are some examples of {language} functions and their summaries:

{examples}
Now summarize the following {language} function. \
Output only the summary, no explanation.

```{language}
{code}
```"""

EXAMPLE_TEMPLATE = "```{language}\n{code}\n```: {summary}"


class FewShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM with few-shot examples from a retriever."""

    name = "few_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", retriever: BaseRetriever = None):
        self.model = model
        self.retriever = retriever
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str, project: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project)
        examples_str = "\n\n".join(
            EXAMPLE_TEMPLATE.format(
                language=language,
                code=s["code"],
                summary=s["docstring"],
            )
            for s in examples
        )
        prompt = PROMPT_TEMPLATE.format(language=language, examples=examples_str, code=code)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def params(self) -> dict:
        return {"model": self.model, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n}
