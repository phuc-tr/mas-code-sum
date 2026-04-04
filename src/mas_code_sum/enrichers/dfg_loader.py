"""
Loads pre-computed DFG context from dataset/{language}/{split}_dfg.json.

Pre-computation:
    bash scripts/setup_dfg_parser.sh
    python scripts/precompute_dfg.py

Usage:
    loader = DFGLoader()
    ctx = loader.get(language="python", url="https://github.com/...")
    # returns the DFG context string, or "" if not available
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATASET_DIR = Path(__file__).parents[3] / "dataset"


class DFGLoader:
    """Lazy-loading cache of pre-computed DFG context strings, keyed by URL."""

    def __init__(self) -> None:
        # (language, split) → {url: dfg_text}
        self._cache: dict[tuple[str, str], dict[str, str]] = {}

    def _load(self, language: str, split: str) -> dict[str, str]:
        key = (language, split)
        if key not in self._cache:
            path = DATASET_DIR / language / f"{split}_dfg.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._cache[key] = json.load(f)
            else:
                self._cache[key] = {}
        return self._cache[key]

    def get(self, language: str, url: str, split: str = "test") -> str:
        """Return the pre-computed DFG context for a sample, or ''."""
        lookup = self._load(language, split)
        return lookup.get(url, "")

    def available(self, language: str, split: str = "test") -> bool:
        """Return True if a pre-computed DFG file exists for this language/split."""
        return (DATASET_DIR / language / f"{split}_dfg.json").exists()


# Module-level singleton — shared across summarizer instances
_LOADER: DFGLoader | None = None


def get_dfg_loader() -> DFGLoader:
    global _LOADER
    if _LOADER is None:
        _LOADER = DFGLoader()
    return _LOADER
