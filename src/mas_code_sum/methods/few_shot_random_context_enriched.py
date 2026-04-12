"""Few-shot summarizer with randomized repository context (name, about, file path from a different project)."""

import json
import os
import random

from openai import OpenAI

from ..data import DATASET_DIR
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, strip_code_fences
from .few_shot_context_enriched import EXAMPLE_TEMPLATE, PROMPT_TEMPLATE, PROMPT_TEMPLATE_NO_CONTEXT
from .zero_shot_context_enriched import _get_metadata_index

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_PATH_INDEX: dict[str, list[str]] | None = None


def _get_path_index() -> dict[str, list[str]]:
    """Build a project -> [paths] index from the train split of all languages."""
    global _PATH_INDEX
    if _PATH_INDEX is not None:
        return _PATH_INDEX
    index: dict[str, list[str]] = {}
    for lang_dir in DATASET_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        train_file = lang_dir / "train.jsonl"
        if not train_file.exists():
            continue
        with open(train_file) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                repo = sample["repo"]
                path = sample["path"]
                index.setdefault(repo, []).append(path)
    if not index:
        raise RuntimeError(f"Path index is empty — no train.jsonl files found under {DATASET_DIR}")
    _PATH_INDEX = index
    return _PATH_INDEX


class FewShotRandomContextEnrichedSummarizer(BaseSummarizer):
    """Few-shot LLM summarizer with repo/path context drawn from a randomly chosen different project."""

    name = "few_shot_random_context_enriched"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct", retriever: BaseRetriever = None):
        self.model = model
        self.retriever = retriever
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )

    def _random_context(self, project: str) -> tuple[str, str, str]:
        """Return (random_repo, about, random_path) from a project other than the given one."""
        meta_index = _get_metadata_index()
        path_index = _get_path_index()
        # Only consider projects that have both metadata and paths
        candidates = [k for k in meta_index if k != project and k in path_index]
        if not candidates:
            raise RuntimeError(f"No candidate projects found for random context (current project: {project})")
        random_project = random.choice(candidates)
        about = meta_index[random_project]["about"]
        path = random.choice(path_index[random_project])
        return random_project, about, path

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        example_blocks = [
            EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        examples_str = "\n\n".join(example_blocks)

        if project:
            repo, about, rnd_path = self._random_context(project)
            prompt = PROMPT_TEMPLATE.format(
                repo=repo,
                about=about,
                path=rnd_path,
                examples=examples_str,
                code=code,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_CONTEXT.format(examples=examples_str, code=code)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return strip_code_fences(response.choices[0].message.content)

    def params(self) -> dict:
        return {"model": self.model, "retriever": type(self.retriever).__name__, "n_shots": self.retriever.n}
