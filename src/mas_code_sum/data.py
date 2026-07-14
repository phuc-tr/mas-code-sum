"""Dataset loading utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

_BASE_DATASET_DIR = Path(__file__).parents[2] / "dataset"
LANGUAGES = ["python", "java", "javascript", "go", "php", "ruby"]

_METADATA_PATH = _BASE_DATASET_DIR / "repo_metadata" / "all_repo_metadata.json"
_METADATA_INDEX: dict[str, dict] | None = None


def get_metadata_index() -> dict[str, dict]:
    """Return repo metadata indexed by repo name (lazy-loaded singleton).

    `all_repo_metadata.json` is a generated artifact under the gitignored
    `dataset/` dir and may not be present in every environment. It only
    supplies the optional "Repository description" prompt line, so a missing
    file degrades to no descriptions rather than failing the run.
    """
    global _METADATA_INDEX
    if _METADATA_INDEX is None:
        if not _METADATA_PATH.exists():
            _METADATA_INDEX = {}
            return _METADATA_INDEX
        with open(_METADATA_PATH) as f:
            data = json.load(f)
        index: dict[str, dict] = {}
        for entries in data.values():
            for entry in entries:
                index[entry["repo"]] = {"about": entry.get("about") or "N/A"}
        _METADATA_INDEX = index
    return _METADATA_INDEX


def _dataset_dir(dataset: str) -> Path:
    return _BASE_DATASET_DIR / dataset


def iter_samples(language: str, split: str = "test", dataset: str = "full") -> Iterator[dict]:
    """Yield samples from dataset/{dataset}/{language}/{split}.jsonl."""
    path = _dataset_dir(dataset) / language / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_flat_samples(split: str, dataset: str) -> Iterator[dict]:
    """Yield samples from a flat dataset/{dataset}/{split}.jsonl."""
    path = _dataset_dir(dataset) / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_samples(language: str, split: str = "test", dataset: str = "full") -> list[dict]:
    flat_path = _dataset_dir(dataset) / f"{split}.jsonl"
    if flat_path.exists():
        return [s for s in _iter_flat_samples(split, dataset) if s.get("language") == language]
    return list(iter_samples(language, split, dataset))


def load_projects(
    languages: list[str],
    split: str = "test",
    max_samples_per_project: int | None = None,
    dataset: str = "full",
    projects: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Load samples grouped by repo (project) across the given languages.

    Args:
        languages: languages to load from
        split: dataset split to use
        max_samples_per_project: if set, cap the number of samples kept per project
        dataset: dataset subdirectory under dataset/ ("full" or "small")
        projects: if set, only include these repos

    Returns:
        dict mapping repo -> list of samples
    """
    project_filter = set(projects) if projects is not None else None
    result: dict[str, list[dict]] = defaultdict(list)

    flat_path = _dataset_dir(dataset) / f"{split}.jsonl"
    if flat_path.exists():
        lang_set = set(languages) if languages else None
        for sample in _iter_flat_samples(split, dataset):
            if lang_set is None or sample.get("language") in lang_set:
                if project_filter is None or sample["repo"] in project_filter:
                    result[sample["repo"]].append(sample)
    else:
        for language in languages:
            for sample in iter_samples(language, split, dataset):
                if project_filter is None or sample["repo"] in project_filter:
                    result[language].append(sample)

    if max_samples_per_project is not None:
        result = {repo: random.sample(samples, min(max_samples_per_project, len(samples))) for repo, samples in result.items()}

    return dict(result)


