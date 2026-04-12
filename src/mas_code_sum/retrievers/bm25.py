"""BM25 retriever — ranks training samples by lexical similarity to query code."""

from rank_bm25 import BM25Okapi

from ..data import load_samples
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Retrieve n training samples ranked by BM25 score against the query code tokens."""

    def __init__(self, n: int = 3):
        self.n = n
        self._samples: dict[str, list[dict]] = {}
        self._index: dict[str, BM25Okapi] = {}

    def _ensure_index(self, language: str) -> None:
        if language not in self._index:
            samples = load_samples(language, split="train")
            self._samples[language] = samples
            self._index[language] = BM25Okapi([s["code_tokens"] for s in samples])

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        self._ensure_index(language)
        k = n or self.n
        query = code.split()
        scores = self._index[language].get_scores(query)
        samples = self._samples[language]

        ranked = sorted(range(len(samples)), key=lambda i: scores[i], reverse=True)

        if project is not None:
            ranked = [i for i in ranked if samples[i]["repo"] == project]

        return [samples[i] for i in ranked[:k]]
