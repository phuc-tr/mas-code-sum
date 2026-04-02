"""Abstract base class for summarization methods."""

from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    """All experiment methods must implement this interface."""

    name: str  # used as MLflow run tag and artifact prefix

    @abstractmethod
    def summarize(self, code: str, language: str) -> str:
        """Generate a one-line summary for the given code snippet."""
        ...

    def params(self) -> dict:
        """Return method hyperparameters to log in MLflow."""
        return {}
