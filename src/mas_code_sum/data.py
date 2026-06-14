"""Dataset loading utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

_BASE_DATASET_DIR = Path(__file__).parents[2] / "dataset"
LANGUAGES = ["python", "java", "javascript", "go", "php", "ruby"]
SAME_PROJECT_DIR = _BASE_DATASET_DIR / "Same-project"


def _versioned_dir(dataset_version: str) -> Path:
    return _BASE_DATASET_DIR / dataset_version


def iter_samples(language: str, split: str = "test", dataset_version: str = "v1") -> Iterator[dict]:
    """Yield samples from dataset/{dataset_version}/{language}/{split}.jsonl."""
    path = _versioned_dir(dataset_version) / language / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_flat_samples(split: str, dataset_version: str) -> Iterator[dict]:
    """Yield samples from a flat dataset/{dataset_version}/{split}.jsonl (e.g. v3)."""
    path = _versioned_dir(dataset_version) / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_same_project_samples(project: str, split: str = "test") -> Iterator[dict]:
    """Yield samples from dataset/Same-project/{project}/{split}.jsonl."""
    path = SAME_PROJECT_DIR / project / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                # Same-project uses "index" instead of "id"
                if "id" not in sample:
                    sample["id"] = sample["index"]
                yield sample


def load_samples(language: str, split: str = "test", dataset_version: str = "v1") -> list[dict]:
    flat_path = _versioned_dir(dataset_version) / f"{split}.jsonl"
    if flat_path.exists():
        return [s for s in _iter_flat_samples(split, dataset_version) if s.get("language") == language]
    return list(iter_samples(language, split, dataset_version))


def load_projects(
    languages: list[str],
    split: str = "test",
    max_samples_per_project: int | None = None,
    dataset_version: str = "v1",
    projects: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Load samples grouped by repo (project) across the given languages.

    Args:
        languages: languages to load from
        split: dataset split to use
        max_samples_per_project: if set, cap the number of samples kept per project
        dataset_version: versioned subdirectory under dataset/ (e.g. "v1", "v2")
        projects: if set, only include these repos

    Returns:
        dict mapping repo -> list of samples
    """
    project_filter = set(projects) if projects is not None else None
    result: dict[str, list[dict]] = defaultdict(list)

    flat_path = _versioned_dir(dataset_version) / f"{split}.jsonl"
    if flat_path.exists():
        lang_set = set(languages) if languages else None
        for sample in _iter_flat_samples(split, dataset_version):
            if lang_set is None or sample.get("language") in lang_set:
                if project_filter is None or sample["repo"] in project_filter:
                    result[sample["repo"]].append(sample)
    else:
        for language in languages:
            for sample in iter_samples(language, split, dataset_version):
                if project_filter is None or sample["repo"] in project_filter:
                    result[language].append(sample)

    if max_samples_per_project is not None:
        result = {repo: random.sample(samples, min(max_samples_per_project, len(samples))) for repo, samples in result.items()}

    return dict(result)


def load_same_project_projects(
    split: str = "test",
    max_samples_per_project: int | None = None,
    projects: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Load Same-project dataset samples grouped by project directory.

    Returns:
        dict mapping project name -> list of samples
    """
    result: dict[str, list[dict]] = {}

    for project_dir in sorted(SAME_PROJECT_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        if projects is not None and project_dir.name not in projects:
            continue
        samples = list(iter_same_project_samples(project_dir.name, split))
        if max_samples_per_project is not None:
            samples = random.sample(samples, min(max_samples_per_project, len(samples)))
        result[project_dir.name] = samples

    return result
