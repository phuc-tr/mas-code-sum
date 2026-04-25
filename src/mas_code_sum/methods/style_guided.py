"""Few-shot summarizer with a data-derived per-project style guide."""

from __future__ import annotations

from ..retrievers.base import BaseRetriever
from ..style_guide import build_style_guide
from .base import BaseSummarizer, make_clients, strip_code_fences
from .zero_shot_context_enriched import _get_metadata_index

_EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {docstring}"""

_PROMPT_TEMPLATE = """\
You are summarizing a {language} function from the repository "{repo}".

Repository description: {about}

File: {path}

Style guide for summaries in this repository:
{style_guide}

Here are some examples of code summarization:

{examples}

Summarize the following code in one sentence, following the style guide above. \
Output only the summary, nothing else.

Code:
{code}
Summary:"""


class StyleGuidedSummarizer(BaseSummarizer):
    """Few-shot summarizer that injects an LLM-derived per-project style guide.

    The style guide is built once from the project's training docstrings and
    cached to disk. At inference time it is prepended to the prompt so the
    model matches the corpus's natural summarization conventions without any
    hand-written rules.
    """

    name = "style_guided"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        style_model: str | None = None,
        retriever: BaseRetriever | None = None,
        n_style_samples: int = 20,
        backend: str = "featherless",
    ):
        self.model = model
        self.style_model = style_model or model
        self.retriever = retriever
        self.n_style_samples = n_style_samples
        self.backend = backend
        self._client, _ = make_clients(backend)
        self._style_cache: dict[tuple[str, str], str] = {}

    def _get_style_guide(self, project: str, language: str) -> str:
        key = (project, language)
        if key not in self._style_cache:
            self._style_cache[key] = build_style_guide(
                project=project,
                language=language,
                model=self.style_model,
                n_samples=self.n_style_samples,
            )
        return self._style_cache[key]

    def summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
    ) -> str:
        ctx = _get_metadata_index().get(project, {"about": "N/A"})
        style_guide = self._get_style_guide(project, language)

        examples = self.retriever.retrieve(code, language, project=project, path=path)
        examples_str = "\n\n".join(
            _EXAMPLE_TEMPLATE.format(
                code=" ".join(s["code_tokens"]),
                docstring=" ".join(s["docstring_tokens"]),
            )
            for s in examples
        )

        prompt = _PROMPT_TEMPLATE.format(
            language=language,
            repo=project,
            about=ctx["about"],
            path=path or "unknown",
            style_guide=style_guide,
            examples=examples_str,
            code=code,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {
            "model": self.model,
            "style_model": self.style_model,
            "retriever": type(self.retriever).__name__,
            "n_shots": self.retriever.n,
            "n_style_samples": self.n_style_samples,
        }
