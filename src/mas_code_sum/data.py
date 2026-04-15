"""Dataset loading utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

DATASET_DIR = Path(__file__).parents[2] / "dataset"
LANGUAGES = ["python", "java", "javascript", "go", "php", "ruby"]
SAME_PROJECT_DIR = DATASET_DIR / "Same-project"


def iter_samples(language: str, split: str = "test") -> Iterator[dict]:
    """Yield samples from dataset/{language}/{split}.jsonl."""
    path = DATASET_DIR / language / f"{split}.jsonl"
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
