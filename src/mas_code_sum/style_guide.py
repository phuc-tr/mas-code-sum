"""Derive and cache per-project style guides from training docstrings."""

from __future__ import annotations

import os
import random
from pathlib import Path

from openai import OpenAI

from .data import load_samples

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CACHE_DIR = Path(__file__).parents[2] / ".cache" / "style_guides"

_STYLE_GUIDE_PROMPT = """\
Below are {n} one-sentence function summaries from the repository "{repo}" ({language}):

{docstrings}

Describe the stylistic patterns you observe across these summaries. Focus on:
- Sentence structure and grammatical form
- Verb tense and whether summaries start with an action verb
- Level of technical detail (do they name specific types, parameters, return values?)
- What information is consistently included or omitted
- Typical length and tone

Be concise (3-5 sentences). Output only the style description, no preamble."""


def build_style_guide(
    project: str,
    language: str,
    model: str = "meta-llama/llama-3.1-8b-instruct",
    n_samples: int = 50,
    seed: int = 42,
) -> str:
    """Return a natural-language style guide derived from the project's training docstrings.

    Results are cached to disk so the LLM call only happens once per project/language/model.
    """
    cache_key = f"{project.replace('/', '_')}__{language}__{model.replace('/', '_')}.txt"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        return cache_path.read_text().strip()

    samples = _sample_project_docstrings(project, language, n_samples, seed)
    if not samples:
        raise ValueError(f"No training samples found for project={project!r} language={language!r}")

    docstrings_str = "\n".join(f"- {d}" for d in samples)
    prompt = _STYLE_GUIDE_PROMPT.format(
        n=len(samples),
        repo=project,
        language=language,
        docstrings=docstrings_str,
    )

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    guide = response.choices[0].message.content.strip()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(guide)
    return guide


def _sample_project_docstrings(
    project: str, language: str, n: int, seed: int
) -> list[str]:
    rng = random.Random(seed)
    samples = load_samples(language, split="train")
    project_samples = [s for s in samples if s["repo"] == project]
    chosen = rng.sample(project_samples, min(n, len(project_samples)))
    return [" ".join(s["docstring_tokens"]) for s in chosen]
