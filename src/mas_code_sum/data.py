"""Dataset loading utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

DATASET_DIR = Path(__file__).parents[2] / "dataset"
LANGUAGES = ["python", "java", "javascript", "go", "php", "ruby"]


def iter_samples(language: str, split: str = "test") -> Iterator[dict]:
    """Yield samples from dataset/{language}/{split}.jsonl."""
    path = DATASET_DIR / language / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_samples(language: str, split: str = "test") -> list[dict]:
    return list(iter_samples(language, split))


def load_projects(
    languages: list[str],
    split: str = "test",
    max_samples_per_project: int | None = None,
) -> dict[str, list[dict]]:
    """
    Load samples grouped by repo (project) across the given languages.

    Args:
        languages: languages to load from
        split: dataset split to use
        max_samples_per_project: if set, cap the number of samples kept per project

    Returns:
        dict mapping repo -> list of samples
    """
    projects: dict[str, list[dict]] = defaultdict(list)

    for language in languages:
        for sample in iter_samples(language, split):
            projects[sample["repo"]].append(sample)

    if max_samples_per_project is not None:
        projects = {repo: random.sample(samples, min(max_samples_per_project, len(samples))) for repo, samples in projects.items()}

    return dict(projects)
