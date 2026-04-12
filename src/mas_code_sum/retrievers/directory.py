"""Directory-proximity retriever — ranks training samples by file path closeness."""

from collections import defaultdict
from pathlib import PurePosixPath

from ..data import load_samples
from .base import BaseRetriever


def _common_prefix_len(a: str, b: str) -> int:
    """Number of shared path components from the left."""
    parts_a = PurePosixPath(a).parts
    parts_b = PurePosixPath(b).parts
    count = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            count += 1
        else:
            break
    return count


class DirectoryRetriever(BaseRetriever):
    """Retrieve n training samples whose file path is closest to the query path.

    Closeness is measured by the number of leading path components shared with
    the query path (longer common prefix = closer).  Samples from the same
    project are required; if ``path`` is not supplied the retriever falls back
    to returning samples in dataset order.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self._cache: dict[str, dict[str, list[dict]]] = {}  # language -> project -> samples

    def _ensure_cache(self, language: str) -> None:
        if language not in self._cache:
            by_project: dict[str, list[dict]] = defaultdict(list)
            for sample in load_samples(language, split="train"):
                by_project[sample["repo"]].append(sample)
            self._cache[language] = dict(by_project)

    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None, path: str | None = None) -> list[dict]:
        self._ensure_cache(language)
        k = n or self.n
        pool = self._cache[language].get(project, []) if project else []

        if not pool:
            return []

        if path is None:
            return pool[:k]

        ranked = sorted(pool, key=lambda s: _common_prefix_len(path, s["path"]), reverse=True)
        return ranked[:k]
