from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, make_clients, strip_code_fences

EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {open}{docstring}{close}"""

FINAL_TEMPLATE = """\
{examples}

Code:
{code}
Summary: {open}"""


def _closing(delimiter: str) -> str:
    """Closing counterpart of an opening delimiter: `<s>` -> `</s>`, otherwise
    the delimiter itself (empty stays empty)."""
    if delimiter.startswith("<") and delimiter.endswith(">"):
        return f"</{delimiter[1:]}"
    return delimiter


class FewShotLLMSummarizer(BaseSummarizer):
    """Summarize code using an LLM with few-shot examples from a retriever."""

    name = "few_shot_llm"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", retriever: BaseRetriever = None, backend: str = "featherless", delimiter: str | None = "<s>"):
        self.model = model
        self.retriever = retriever
        self.backend = backend
        # Wraps each summary as <s>...</s> so generation can be cut at the closing
        # tag. Set to null/"" in the config to prompt without any delimiter, in
        # which case the first output line is taken as the summary.
        self.delimiter = delimiter or ""
        _, self._async_client = make_clients(backend)

    def build_prompt(self, code: str, language: str, project: str | None = None, path: str | None = None) -> str:
        """Assemble the flat few-shot prompt (example blocks + query block)."""
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        open_tag = self.delimiter
        close_tag = _closing(self.delimiter)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(
                code=" ".join(s["code_tokens"]),
                docstring=" ".join(s["docstring_tokens"]),
                open=open_tag,
                close=close_tag,
            )
            for s in examples
        ]
        # rstrip: without a delimiter the prompt would end in "Summary: ", and a
        # trailing space skews tokenization for completion models.
        return FINAL_TEMPLATE.format(examples="\n\n".join(example_blocks), code=code, open=open_tag).rstrip()

    def parse_reply(self, raw: str) -> str:
        """Cut the model's continuation down to the single summary."""
        close_tag = _closing(self.delimiter)
        end = raw.find(close_tag) if close_tag else -1
        comment = raw[:end].strip() if end != -1 else raw.strip().split("\n")[0].strip()
        return strip_code_fences(comment)

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        prompt = self.build_prompt(code, language, project=project, path=path)
        close_tag = _closing(self.delimiter)
        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
            # Stop at the closing delimiter so generation ends where
            # `parse_reply` cuts, instead of running to `max_tokens` and having
            # the continuation thrown away after being billed.
            #
            # Only applied when there is a delimiter: without one, `parse_reply`
            # keeps the first line, and stopping at "\n" would return an empty
            # string whenever the model opens with a newline.
            **({"stop": [close_tag]} if close_tag else {}),
        )
        return self.parse_reply(response.choices[0].text or "")

    def params(self) -> dict:
        return {"model": self.model, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n, "backend": self.backend, "delimiter": self.delimiter or "none"}
