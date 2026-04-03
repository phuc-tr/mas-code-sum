import os

from openai import OpenAI

from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, strip_code_fences

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {docstring}"""

FINAL_TEMPLATE = """\
Here are some examples of code summarization from the same project:

{examples}

Now summarize the following code in one sentence. Output only the summary, no explanation:

Code:
{code}
Summary:"""


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
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=s["code"], docstring=s["docstring"])
            for s in examples
        ]
        prompt = FINAL_TEMPLATE.format(
            examples="\n\n".join(example_blocks),
            code=code,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n}
