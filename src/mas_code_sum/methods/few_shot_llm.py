from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_local_clients, make_openai_clients, strip_code_fences

EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: <s>{docstring}</s>"""

QUERY_TEMPLATE = """\
Code:
{code}
Summary: <s>"""


class FewShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM with few-shot examples from a retriever."""

    name = "few_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", retriever: BaseRetriever = None, max_concurrency: int = 5, backend: str = "openrouter", device: str = "cuda"):
        self.model = model
        self.retriever = retriever
        self.max_concurrency = max_concurrency
        self.backend = backend
        if backend == "local":
            self._client, self._async_client = make_local_clients(model, device)
        else:
            self._client, self._async_client = make_openai_clients()

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        prompt = "\n\n".join(example_blocks) + "\n\n" + QUERY_TEMPLATE.format(code=code)
        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
        )
        raw = response.choices[0].text or ""
        end = raw.find("</s>")
        comment = raw[:end].strip() if end != -1 else raw.split("\n")[0].strip()
        return strip_code_fences(comment)

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        prompt = "\n\n".join(example_blocks) + "\n\n" + QUERY_TEMPLATE.format(code=code)
        response = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
        )
        raw = response.choices[0].text or ""
        end = raw.find("</s>")
        comment = raw[:end].strip() if end != -1 else raw.split("\n")[0].strip()
        return strip_code_fences(comment)

    def params(self) -> dict:
        return {"model": self.model, "backend": self.backend, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n}
