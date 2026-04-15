"""BM25 retriever — ranks training samples by lexical similarity to query code."""

from rank_bm25 import BM25Okapi

from ..data import SAME_PROJECT_DIR, iter_same_project_samples, load_samples
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Retrieve n training samples ranked by BM25 score against the query code tokens."""

    def __init__(self, n: int = 3):
        self.n = n
        self._samples: dict[str, list[dict]] = {}
        self._index: dict[str, BM25Okapi] = {}

    def _ensure_index(self, key: str, samples: list[dict]) -> None:
        if key not in self._index:
            self._samples[key] = samples
            self._index[key] = BM25Okapi([s["code_tokens"] for s in samples])

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        # Use per-project index for same-project dataset; fall back to language-wide index.
        if project is not None and (SAME_PROJECT_DIR / project).is_dir():
            key = f"same_project:{project}"
            if key not in self._index:
                self._ensure_index(key, list(iter_same_project_samples(project, split="train")))
        else:
            key = language
            if key not in self._index:
                self._ensure_index(key, load_samples(language, split="train"))

        k = n or self.n
        query = code.split()
        scores = self._index[key].get_scores(query)
        samples = self._samples[key]
        ranked = sorted(range(len(samples)), key=lambda i: scores[i], reverse=True)

        if project is not None and key == language:
            ranked = [i for i in ranked if samples[i]["repo"] == project]

        return [samples[i] for i in ranked[:k]]
