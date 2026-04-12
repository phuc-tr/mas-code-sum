"""Random retriever — selects n samples at random from the training set."""

import random

from ..data import load_samples
from .base import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Retrieve n random samples from the training split of the given language."""

    def __init__(self, n: int = 3):
        self.n = n
        self._cache: dict[str, list[dict]] = {}

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        if language not in self._cache:
            self._cache[language] = load_samples(language, split="train")
        pool = self._cache[language]
        return random.sample(pool, min(n or self.n, len(pool)))
