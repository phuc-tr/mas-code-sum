"""Random retriever — selects n samples at random from the same project in the training set."""

import random
from collections import defaultdict

from ..data import load_samples
from .base import BaseRetriever


class RandomSameProjectRetriever(BaseRetriever):
    """Retrieve n random samples from the same project (repository_name) in the training split."""

    def __init__(self, n: int = 3):
        self.n = n
        self._cache: dict[str, dict[str, list[dict]]] = {}  # language -> project -> samples

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        if language not in self._cache:
            by_project: dict[str, list[dict]] = defaultdict(list)
            for sample in load_samples(language, split="train"):
                by_project[sample["repo"]].append(sample)
            self._cache[language] = dict(by_project)

        pool = self._cache[language].get(project, []) if project else []
        return random.sample(pool, min(n or self.n, len(pool)))
