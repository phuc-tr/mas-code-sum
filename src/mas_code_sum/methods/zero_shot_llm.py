import os

from openai import OpenAI

from .base import BaseSummarizer, strip_code_fences

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_TEMPLATE = """\
Please generate a short comment in one sentence for the following function. Output only the summary, no explanation:

{code}
"""


class ZeroShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM via OpenRouter."""

    name = "zero_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.model = model
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str, project: str | None = None) -> str:
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
        return {"model": self.model}
