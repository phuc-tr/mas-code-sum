"""HyDE retriever — generate a hypothetical docstring, then retrieve by docstring similarity."""

from rank_bm25 import BM25Okapi

from ..data import load_samples
from ..methods.base import make_openai_clients, strip_code_fences
from .base import BaseRetriever

_PROMPT = """\
Write a one-line docstring summarising what the following {language} function does. \
Reply with only the docstring text, no code fences or quotes.

{code}"""


class HyDERetriever(BaseRetriever):
    """HyDE: generate a hypothetical docstring for the query, then retrieve by BM25 over docstring_tokens."""

    def __init__(self, n: int = 3, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.n = n
        self.model = model
        self._client, _ = make_openai_clients()
        self._samples: dict[str, list[dict]] = {}
        self._index: dict[str, BM25Okapi] = {}

    def _ensure_index(self, language: str) -> None:
        if language not in self._index:
            samples = load_samples(language, split="train")
            self._samples[language] = samples
            self._index[language] = BM25Okapi([s["docstring_tokens"] for s in samples])

    def _hypothetical_docstring(self, code: str, language: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": _PROMPT.format(language=language, code=code)}],
            max_tokens=64,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content or "")

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        self._ensure_index(language)
        k = n or self.n
        hypothesis = self._hypothetical_docstring(code, language)
        query = hypothesis.split()
        scores = self._index[language].get_scores(query)
        samples = self._samples[language]

        ranked = sorted(range(len(samples)), key=lambda i: scores[i], reverse=True)

        if project is not None:
            ranked = [i for i in ranked if samples[i]["repo"] == project]

        return [samples[i] for i in ranked[:k]]
