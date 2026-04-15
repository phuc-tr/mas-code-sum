"""Random retriever — selects n samples at random from the same project's training split."""

import random

from ..data import iter_same_project_samples
from .base import BaseRetriever


class RandomSameProjectRetriever(BaseRetriever):
    """Retrieve n random samples from the same project's training split (same-project dataset)."""

    def __init__(self, n: int = 3):
        self.n = n
        self._cache: dict[str, list[dict]] = {}  # project -> samples

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        if project is None:
            return []
        if project not in self._cache:
            self._cache[project] = list(iter_same_project_samples(project, split="train"))
        pool = self._cache[project]
        return random.sample(pool, min(n or self.n, len(pool)))
