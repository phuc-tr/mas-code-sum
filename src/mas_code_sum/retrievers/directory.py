"""Directory-proximity retriever — ranks training samples by file path closeness."""

from collections import defaultdict
from pathlib import PurePosixPath

from ..data import load_few_shot_samples, load_samples
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
    """Retrieve samples whose file path is closest to the query path.

    Closeness is measured by the number of leading path components shared with
    the query path (longer common prefix = closer).  Samples from the same
    project are required; if ``path`` is not supplied the retriever falls back
    to returning samples in dataset order.

    Args:
        n: number of examples to return
        pool: ``"train"`` uses the existing train split; ``"few_shots"`` uses
            the pool extracted from raw repo source files.
    """

    def __init__(self, n: int = 3, pool: str = "train"):
        self.n = n
        self.pool = pool
        self._cache: dict[str, dict[str, list[dict]]] = {}  # language -> project -> samples

    def _ensure_cache(self, language: str) -> None:
        if language not in self._cache:
            by_project: dict[str, list[dict]] = defaultdict(list)
            source = (
                load_few_shot_samples(language) if self.pool == "few_shots"
                else load_samples(language, split="train")
            )
            for sample in source:
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
