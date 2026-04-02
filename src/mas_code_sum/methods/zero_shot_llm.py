import os

from openai import OpenAI

from .base import BaseSummarizer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_TEMPLATE = """\
Summarize the following {language} function in one concise sentence. \
Output only the summary, no explanation.

```{language}
{code}
```"""


class ZeroShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM via OpenRouter."""

    name = "zero_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.model = model
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def summarize(self, code: str, language: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(language=language, code=code)},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def params(self) -> dict:
        return {"model": self.model}
