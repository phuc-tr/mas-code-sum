"""Dataset loading utilities."""

import json
from pathlib import Path
from typing import Iterator

DATASET_DIR = Path(__file__).parents[2] / "dataset"
LANGUAGES = ["python", "java", "javascript", "go", "php", "ruby"]


def iter_samples(language: str, split: str = "test") -> Iterator[dict]:
    """Yield samples from dataset/{language}.jsonl filtered by split_name."""
    path = DATASET_DIR / f"{language}.jsonl"
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            if split is None or sample.get("split_name") == split:
                yield sample


def load_samples(language: str, split: str = "test", max_samples: int | None = None) -> list[dict]:
    samples = []
    for sample in iter_samples(language, split):
        samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break
    return samples
