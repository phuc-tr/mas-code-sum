"""Metagente-style multi-agent prompt optimization for code summarization.

Ports the "parallel optimizer" from https://github.com/MDEGroup/Metagente
(originally built for GitHub README summarization) to code -> docstring
summarization. Four agent roles, each a thin LLM wrapper:

  - Extractor: shortens a code snippet to its essential logic.
  - Summarizer: turns extracted code into a one-sentence summary, driven by
    an evolving prompt template.
  - Teacher: rewrites the summarizer prompt when its ROUGE-L score against
    the ground-truth docstring falls short of a threshold.
  - Prompt-Combine: merges the per-example best summarizer prompts (one per
    training sample) into a single, general-purpose prompt.

Training (over a small sample of the `train` split) runs once to produce a
final summarizer prompt; the learned prompt is cached to disk and then
reused for inference across the batch/async path shared with the rest of
the framework.

Per training example, the extractor/summarizer/teacher iteration chain is
inherently sequential (each teacher call depends on the previous
summarizer output). But the training examples are independent of each
other, so they are optimized concurrently (bounded by the same
per-backend `max_concurrency` semaphore used by `summarize_batch`), unlike
the original repo's `ParallelOptimizer` which — despite the name — loops
over examples one at a time.

Optionally, an in-context-learning (ICL) retriever can be attached
(`retriever=` — any `BaseRetriever`, e.g. `BM25Retriever`, following the
same convention as `few_shot_llm.py`) to prepend a few retrieved
code/summary examples ahead of the learned prompt at inference time. This
is purely an inference-time augmentation: it does not affect training or
the cached final prompt, so a cache produced without ICL remains valid and
reusable after enabling (or changing) ICL, and vice versa — no retraining
needed. See `_cache_key()`: the retriever/ICL settings are deliberately
excluded from the training-config fingerprint.
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import string
import threading
from datetime import datetime, timezone
from pathlib import Path

from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm as atqdm

from ..data import dataset_dir, load_samples
from ..retrievers.base import BaseRetriever
from .base import BaseSummarizer, _call_with_rate_limit_retry, make_clients, strip_code_fences

_log = logging.getLogger(__name__)

_PROMPT_VERSION = "v1"
_DEFAULT_CACHE_DIR = dataset_dir() / "metagente_cache"

EXTRACTOR_PROMPT = string.Template("""\
You are given a $language function and its surrounding code. Extract and shorten it to only the \
logic that is essential to understanding what the function does. Strip away irrelevant boilerplate: \
unrelated imports, logging statements, argument-validation noise, and decorators that do not affect \
behavior. Output only the shortened, code-relevant text. Do not add any explanation or output \
identifiers like "Here's the ..." or "Extracted code:".

Code:
$code
""")

INITIAL_SUMMARIZER_PROMPT = (
    "Please generate a short comment in one sentence for the following function. "
    "Output only the summary, no explanation:\n\n$extracted_text"
)

TEACHER_PROMPT = string.Template("""\
You are a professional Prompt Engineer. You are working on a system using a Large Language Model \
(LLM) to help developers automatically generate a short one-sentence comment describing what a \
function does, from an extracted version of its code. Your task is to modify and improve the current \
prompt of the LLM based on the result of testing on a data point that includes the extracted code and \
a ground truth docstring.

# Steps:
- **Analyze the data for testing**: Analyze the following extracted code and ground truth docstring:
<EXTRACTED_CODE>
$extracted_text
</EXTRACTED_CODE>

<GROUND_TRUTH_DOCSTRING>
$description
</GROUND_TRUTH_DOCSTRING>
- **Review the current result**: Review the generated summary using the extracted code and its \
ROUGE-L score against the ground truth docstring to identify improvements that could be made:
<GENERATED_SUMMARY>
$generated_about
</GENERATED_SUMMARY>
<ROUGE_L_SCORE>
$rouge_score
</ROUGE_L_SCORE>
- **Modify the current prompt**: Identify mistakes and lacking instructions in the current prompt from \
the result of the above review. You should preserve the current prompt as much as possible and only \
make small changes to the prompt based on the identified mistakes and lacking instructions.
<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>
The new prompt MUST still contain the literal placeholder "$$extracted_text" exactly once, since it \
will be filled in with the extracted code at generation time. As the new prompt will not include the \
ground truth docstring, DO NOT mention the ground truth docstring in the new prompt. DO NOT include \
any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any \
output identifiers like "Prompt:", "New Prompt", ... The output should only include a string \
representing the new prompt for the LLM.
""")

COMBINE_PROMPT = string.Template("""\
You are a professional Prompt Engineer. You are working on a system using a Large Language Model \
(LLM) to help developers automatically generate a short one-sentence comment describing what a \
function does, from an extracted version of its code. Your task is to combine several candidate \
prompts for the LLM into a final prompt.

# Steps:
- **Review all candidate prompts**: Analyze the following prompts to identify common parts to be \
included in the final prompt, and also include specific details or conditional key points from these \
prompts in the final prompt:
<CANDIDATE_PROMPTS>
$prompt_list
</CANDIDATE_PROMPTS>
- **Generate a final prompt**: Based on the common parts and conditional key points, generate a final \
prompt for the LLM. The final prompt MUST still contain the literal placeholder "$$extracted_text" \
exactly once.

# Output Format:
Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", \
... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a \
string representing the prompt for the LLM.
""")

ICL_EXAMPLE_TEMPLATE = """\
Code:
{code}
Summary: {docstring}"""

assert "$extracted_text" in INITIAL_SUMMARIZER_PROMPT, "INITIAL_SUMMARIZER_PROMPT must be substitutable with extracted_text"

_PLACEHOLDER_RE = re.compile(r"\$\{?extracted_text\}?")


class MetagenteSummarizer(BaseSummarizer):
    """Metagente-style (parallel optimizer) multi-agent prompt-tuning summarizer."""

    name = "metagente"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        teacher_model: str = "meta-llama/llama-3.3-70b-instruct",
        backend: str = "featherless",
        num_train_samples: int = 20,
        max_iterations: int = 15,
        rouge_threshold: float = 0.7,
        train_dataset: str = "full",
        train_split: str = "train",
        pool_languages: bool = True,
        cache_dir: str | Path | None = None,
        retriever: BaseRetriever | None = None,
    ):
        self.model = model
        self.teacher_model = teacher_model
        self.backend = backend
        self.num_train_samples = num_train_samples
        self.max_iterations = max_iterations
        self.rouge_threshold = rouge_threshold
        self.train_dataset = train_dataset
        self.train_split = train_split
        self.pool_languages = pool_languages
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.retriever = retriever

        _, self._async_client = make_clients(backend)
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self._lock = threading.Lock()
        self._final_prompt: dict[str, str] = {}
        self._train_meta: dict[str, dict] = {}
        self._languages_seen: set[str] = set()

    # ------------------------------------------------------------------
    # BaseSummarizer interface
    # ------------------------------------------------------------------

    def summarize_batch(
        self,
        codes: list[str],
        languages: list[str],
        projects: list[str | None] | None = None,
        paths: list[str | None] | None = None,
        urls: list[str | None] | None = None,
    ) -> list[str]:
        self._languages_seen |= set(languages)
        for lang in sorted(set(languages)):
            self._ensure_trained(lang)
        return super().summarize_batch(codes, languages, projects=projects, paths=paths, urls=urls)

    async def async_summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        key = self._train_key(language)
        final_prompt = self._final_prompt[key]

        extracted_text = await self._extractor_agent(code, language)
        rendered = string.Template(final_prompt).substitute(extracted_text=extracted_text)
        rendered = self._prepend_icl_examples(rendered, code, language, project, path)
        return await _call_with_rate_limit_retry(lambda: self._async_chat(self.model, rendered, temperature=0.0))

    def _prepend_icl_examples(self, rendered_prompt: str, code: str, language: str, project: str | None, path: str | None) -> str:
        """Prepend a few retrieved code/summary examples ahead of the learned
        prompt (in-context learning), if a retriever is configured. Purely an
        inference-time augmentation — does not touch the cached/trained
        prompt itself."""
        if self.retriever is None:
            return rendered_prompt
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        if not examples:
            return rendered_prompt
        example_blocks = [
            ICL_EXAMPLE_TEMPLATE.format(code=" ".join(s["code_tokens"]), docstring=" ".join(s["docstring_tokens"]))
            for s in examples
        ]
        icl_prefix = "Here are examples of code and their one-sentence summaries:\n\n" + "\n\n".join(example_blocks) + "\n\n---\n\n"
        return icl_prefix + rendered_prompt

    def params(self) -> dict:
        return {
            "model": self.model,
            "teacher_model": self.teacher_model,
            "backend": self.backend,
            "num_train_samples": self.num_train_samples,
            "max_iterations": self.max_iterations,
            "rouge_threshold": self.rouge_threshold,
            "pool_languages": self.pool_languages,
            "train_dataset": self.train_dataset,
            "retriever": type(self.retriever).__name__ if self.retriever else None,
            "icl_n": self.retriever.n if self.retriever else None,
        }

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def _train_key(self, language: str) -> str:
        return "__pooled__" if self.pool_languages else language

    def _ensure_trained(self, language: str) -> None:
        key = self._train_key(language)
        if key in self._final_prompt:
            return
        with self._lock:
            if key in self._final_prompt:
                return
            cache_key = self._cache_key(key)
            cached = self._load_cache(cache_key, key)
            if cached is not None:
                self._final_prompt[key] = cached["final_summarizer_prompt"]
                self._train_meta[key] = {**cached, "from_cache": True}
                print(f"metagente[{key}]: loaded cached prompt (cache_key={cache_key}), skipping training")
            else:
                prompt, meta = asyncio.run(self._train(language, key))
                meta["from_cache"] = False
                self._final_prompt[key] = prompt
                self._train_meta[key] = meta
                self._save_cache(cache_key, prompt, meta)
                print(f"metagente[{key}]: trained new prompt (cache_key={cache_key})")
            self._log_mlflow_artifacts(key)

    def _log_mlflow_artifacts(self, key: str) -> None:
        try:
            import mlflow

            mlflow.log_text(self._final_prompt[key], f"metagente/final_prompt_{key}.txt")
            mlflow.log_dict(self._train_meta[key], f"metagente/train_meta_{key}.json")
        except Exception:
            _log.debug("metagente: MLflow artifact logging skipped (no active run?).", exc_info=True)

    async def _train(self, language: str, key: str) -> tuple[str, dict]:
        train_samples = self._select_training_samples(language)
        if not train_samples:
            _log.warning("metagente: no training samples found for %r; using initial prompt unmodified.", key)
            return INITIAL_SUMMARIZER_PROMPT, {
                "num_train_samples": 0,
                "max_iterations": self.max_iterations,
                "rouge_threshold": self.rouge_threshold,
                "avg_best_rougeL": 0.0,
                "model": self.model,
                "teacher_model": self.teacher_model,
                "backend": self.backend,
                "language_key": key,
                "trained_at": _utc_now_iso(),
                "prompt_version": _PROMPT_VERSION,
            }

        n = len(train_samples)
        print(f"metagente[{key}]: starting training on {n} samples (max_iterations={self.max_iterations}, rouge_threshold={self.rouge_threshold:.2f}, max_concurrency={self.max_concurrency})")

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _optimize_one(idx: int, sample: dict) -> tuple[str, float]:
            async with sem:
                code = " ".join(sample["code_tokens"])
                description = " ".join(sample["docstring_tokens"])
                lang = sample["language"]

                extracted_text = await self._extractor_agent(code, lang)
                summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
                best_score, best_prompt = -1.0, summarizer_prompt

                for it in range(1, self.max_iterations + 1):
                    about = await self._summarizer_agent(summarizer_prompt, extracted_text)
                    score = self._rouge_l(about, description)
                    if score > best_score:
                        best_score, best_prompt = score, summarizer_prompt
                    print(f"metagente[{key}]: sample {idx}/{n} iter {it}/{self.max_iterations} rougeL={score:.3f} (best={best_score:.3f})")
                    if score >= self.rouge_threshold:
                        print(f"metagente[{key}]: sample {idx}/{n} converged at iter {it} (rougeL={score:.3f} >= {self.rouge_threshold:.2f})")
                        break
                    summarizer_prompt = await self._teacher_agent(extracted_text, description, about, score, summarizer_prompt)
                else:
                    print(f"metagente[{key}]: sample {idx}/{n} did not converge after {self.max_iterations} iterations (best rougeL={best_score:.3f})")

                # Deliberate deviation from upstream: always contribute a
                # prompt for this example (even if it never crossed
                # rouge_threshold), so every training example informs the
                # final combine step.
                return best_prompt, best_score

        results = await atqdm.gather(
            *[_optimize_one(idx, sample) for idx, sample in enumerate(train_samples, start=1)],
            desc=f"metagente[{key}] training",
        )
        data_prompt = [prompt for prompt, _ in results]
        per_example_scores = [score for _, score in results]

        avg_so_far = sum(per_example_scores) / len(per_example_scores)
        print(f"metagente[{key}]: finished per-sample optimization, avg best rougeL={avg_so_far:.3f}; combining {len(data_prompt)} prompts")

        final_prompt = await self._combine_agent(data_prompt)
        print(f"metagente[{key}]: training complete, final combined prompt ready")
        meta = {
            "num_train_samples": len(train_samples),
            "max_iterations": self.max_iterations,
            "rouge_threshold": self.rouge_threshold,
            "avg_best_rougeL": avg_so_far,
            "model": self.model,
            "teacher_model": self.teacher_model,
            "backend": self.backend,
            "language_key": key,
            "trained_at": _utc_now_iso(),
            "prompt_version": _PROMPT_VERSION,
        }
        return final_prompt, meta

    def _select_training_samples(self, language: str) -> list[dict]:
        rng = random.Random(0)
        if self.pool_languages:
            langs = sorted(self._languages_seen) or [language]
            per_lang_n = max(1, self.num_train_samples // len(langs))
            pooled: list[dict] = []
            for lang in langs:
                samples = load_samples(lang, split=self.train_split, dataset=self.train_dataset)
                pooled.extend(rng.sample(samples, min(per_lang_n, len(samples))))
            rng.shuffle(pooled)
            return pooled[: self.num_train_samples]

        samples = load_samples(language, split=self.train_split, dataset=self.train_dataset)
        return rng.sample(samples, min(self.num_train_samples, len(samples)))

    # ------------------------------------------------------------------
    # Agent helpers (async, shared by training and inference; concurrency
    # and retries are handled by the caller / _call_with_rate_limit_retry)
    # ------------------------------------------------------------------

    async def _extractor_agent(self, code: str, language: str) -> str:
        rendered = EXTRACTOR_PROMPT.substitute(code=code, language=language)
        return await _call_with_rate_limit_retry(lambda: self._async_chat(self.model, rendered, temperature=0.0, max_tokens=256))

    async def _summarizer_agent(self, summarizer_prompt: str, extracted_text: str) -> str:
        rendered = string.Template(summarizer_prompt).substitute(extracted_text=extracted_text)
        return await _call_with_rate_limit_retry(lambda: self._async_chat(self.model, rendered, temperature=0.0, max_tokens=128))

    async def _teacher_agent(self, extracted_text: str, description: str, generated_about: str, rouge_score: float, summarizer_prompt: str) -> str:
        rendered = TEACHER_PROMPT.substitute(
            extracted_text=extracted_text,
            description=description,
            generated_about=generated_about,
            rouge_score=f"{rouge_score:.3f}",
            summarizer_prompt=summarizer_prompt,
        )
        raw = await _call_with_rate_limit_retry(lambda: self._async_chat(self.teacher_model, rendered, temperature=0.7, max_tokens=512))
        return self._parse_answer(raw)

    async def _combine_agent(self, prompt_list: list[str]) -> str:
        formatted = "\n\n".join(f"Summarizer Prompt #{i}:\n{p}" for i, p in enumerate(prompt_list))
        rendered = COMBINE_PROMPT.substitute(prompt_list=formatted)
        raw = await _call_with_rate_limit_retry(lambda: self._async_chat(self.teacher_model, rendered, temperature=0.2, max_tokens=512))
        return self._parse_answer(raw)

    def _rouge_l(self, prediction: str, reference: str) -> float:
        return self._scorer.score(reference, prediction)["rougeL"].fmeasure

    def _parse_answer(self, text: str) -> str:
        """Ensure the LLM-produced prompt is a valid, substitutable Template
        containing the required $extracted_text placeholder."""
        text = text.strip()
        if not _PLACEHOLDER_RE.search(text):
            repaired, n = re.subn(r"\btext\b", "$extracted_text", text, count=1)
            if n == 1:
                _log.warning("metagente: LLM-produced prompt missing $extracted_text; repaired via 'text' substitution.")
                text = repaired
            else:
                _log.warning("metagente: LLM-produced prompt missing $extracted_text and no 'text' token to repair; appending placeholder.")
                text = text.rstrip() + "\n\n$extracted_text"

        try:
            string.Template(text).substitute(extracted_text="")
        except (KeyError, ValueError) as exc:
            _log.warning("metagente: repaired prompt still invalid (%s); falling back to INITIAL_SUMMARIZER_PROMPT.", exc)
            return INITIAL_SUMMARIZER_PROMPT
        return text

    # ------------------------------------------------------------------
    # Async inference helper
    # ------------------------------------------------------------------

    async def _async_chat(self, model: str, prompt: str, temperature: float, max_tokens: int = 128) -> str:
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return strip_code_fences(response.choices[0].message.content or "")

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _fingerprint(self, key: str) -> dict:
        return {
            "model": self.model,
            "teacher_model": self.teacher_model,
            "backend": self.backend,
            "train_dataset": self.train_dataset,
            "train_split": self.train_split,
            "language_key": key,
            "num_train_samples": self.num_train_samples,
            "max_iterations": self.max_iterations,
            "rouge_threshold": self.rouge_threshold,
            "prompt_version": _PROMPT_VERSION,
        }

    def _cache_key(self, key: str) -> str:
        blob = json.dumps(self._fingerprint(key), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _load_cache(self, cache_key: str, key: str) -> dict | None:
        path = self._cache_dir / f"{cache_key}.json"
        if not path.exists():
            self._warn_cache_miss(cache_key, key)
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("metagente: failed to read cache %s: %s; retraining.", path, exc)
            return None

    def _warn_cache_miss(self, cache_key: str, key: str) -> None:
        """No cache file for the current fingerprint. If OTHER cache files
        exist (likely from a run with slightly different training-affecting
        params — e.g. backend/max_iterations drifted between yaml configs),
        print a diff so the mismatch is obvious instead of silently
        retraining."""
        wanted = self._fingerprint(key)
        print(f"metagente[{key}]: no cache for {cache_key} ({self._cache_dir}); will train from scratch")
        others = sorted(self._cache_dir.glob("*.json")) if self._cache_dir.exists() else []
        if not others:
            return
        print(f"metagente[{key}]: {len(others)} other cache file(s) found under {self._cache_dir} — checking for a near-miss:")
        for path in others:
            try:
                with open(path) as f:
                    cached = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            diffs = {
                k: (wanted.get(k), cached.get(k))
                for k in wanted
                if k != "language_key" and wanted.get(k) != cached.get(k)
            }
            if diffs:
                print(f"    {path.name}: differs in {diffs}")
            else:
                print(f"    {path.name}: language_key={cached.get('language_key')!r} (matches everything else — check pool_languages/language mismatch)")

    def _save_cache(self, cache_key: str, prompt: str, meta: dict) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{cache_key}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"final_summarizer_prompt": prompt, **meta}, f, indent=2)
        tmp.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
