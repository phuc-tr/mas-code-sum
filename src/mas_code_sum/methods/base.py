"""Abstract base class for summarization methods."""

import re
from abc import ABC, abstractmethod


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences and triple-quotes from LLM output."""
    text = text.strip()
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = re.sub(r'^"""\n?', "", text)
    text = re.sub(r'\n?"""$', "", text)
    return text.strip()


class BaseSummarizer(ABC):
    """All experiment methods must implement this interface."""

    name: str  # used as MLflow run tag and artifact prefix

    @abstractmethod
    def summarize(self, code: str, language: str, project: str | None = None) -> str:
        """Generate a one-line summary for the given code snippet."""
        ...

    def params(self) -> dict:
        """Return method hyperparameters to log in MLflow."""
        return {}
