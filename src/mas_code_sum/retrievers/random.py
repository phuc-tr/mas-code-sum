"""Random retriever — selects n samples at random from the training set."""

import random

from ..data import load_few_shot_samples, load_samples
from .base import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Retrieve n random samples from the training split of the given language.

    Args:
        n: number of examples to return
        pool: ``"train"`` uses the existing train split; ``"few_shots"`` uses
            the pool extracted from raw repo source files.
    """

    def __init__(self, n: int = 3, pool: str = "train"):
        self.n = n
        self.pool = pool
        self._cache: dict[str, list[dict]] = {}

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        if language not in self._cache:
            self._cache[language] = (
                load_few_shot_samples(language) if self.pool == "few_shots"
                else load_samples(language, split="train")
            )
        return random.sample(self._cache[language], min(n or self.n, len(self._cache[language])))
