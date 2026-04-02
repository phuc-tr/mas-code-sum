"""Abstract base class for retrievers."""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Given a code snippet, return n relevant samples from the training set."""

    @abstractmethod
    def retrieve(self, code: str, language: str, n: int | None = None, project: str | None = None) -> list[dict]:
        """
        Return n training samples to use as few-shot examples.

        Each returned sample must have at least:
            - func_code_string
            - func_documentation_string
        """
        ...
