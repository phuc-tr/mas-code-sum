from .base import BaseSummarizer


class ExactCopySummarizer(BaseSummarizer):
    """Baseline: return the code unchanged as the summary."""

    name = "exact_copy"

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        return code
