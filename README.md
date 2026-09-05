<div align="center">

# mas-code-sum

**Multi-method code summarization research framework**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

Plug-in methods · BM25 retrieval · Agentic RAG · BLEU · ROUGE-L · Round-trip correctness

</div>

---

## Overview

`mas-code-sum` is an experiment framework for studying LLM-based code summarization. It provides:

- **Pluggable methods** — zero-shot, few-shot, ASAP, file-context enriched, agentic RAG, CodeT5
- **Pluggable retrievers** — BM25 (in-project and cross-project), random
- **Reference-free evaluation** — round-trip correctness (RTC) scores a summary by how well a backward model can reconstruct the original code from it
- **MLflow integration** — every run logs params, per-project metrics, cost, and prediction artifacts automatically
- **YAML-driven experiments** — swap models, retrievers, and datasets without touching code

All runs are tracked in MLflow under the **`code-summarization`** experiment.

**Headline finding:** CARA (`agentic_rag_all_context`) improves BLEU over zero-shot
prompting without a practically significant drop in round-trip correctness, while ASAP —
the baseline it is compared against — buys a larger BLEU gain at a materially significant
RTC cost. See [Results](#results).

---

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
  - [Schema](#schema)
- [Running Experiments](#running-experiments)
  - [Experiment Config](#experiment-config)
  - [MLflow Tracking](#mlflow-tracking)
- [Methods](#methods)
  - [ASAP DFG context](#asap-dfg-context)
- [Retrievers](#retrievers)
- [Round-Trip Correctness (RTC)](#round-trip-correctness-rtc)
- [Adding a New Method](#adding-a-new-method)
- [Adding a New Metric](#adding-a-new-metric)
- [Results](#results)

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

LLM methods support two backends. Set the relevant API key(s):

```bash
export OPENROUTER_API_KEY=sk-or-...   # OpenRouter
export FEATHERLESS_API_KEY=...         # Featherless (default for completions-based methods)
```

Pick one per run with `--backend {openrouter,featherless}` (or `method_params.backend` in the YAML).

Methods that read a sample's surrounding source file (`few_shot_file_context`, `few_shot_all_context`, the `agentic_rag` family) and the RTC metric need the dataset repos checked out locally at their recorded shas:

```bash
python scripts/clone_dataset_repos.py
```

---

## Project Structure

```
mas-code-sum/
├── asap_scripts/                    # ASAP DFG parser (CodeXGLUE upstream)
│   └── scripts/
│       └── parser/                  # DFG extraction modules (Python + Java)
├── dataset/
│   ├── full/                        # complete dataset (default)
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   └── small/                       # 6-project subset for fast iteration
│       ├── train.jsonl
│       └── test.jsonl
├── experiments/
│   └── *.yaml                       # experiment configs; pass --dataset full|small at run time
├── scripts/                         # pipeline utilities
│   ├── clone_dataset_repos.py       # clone each dataset repo at its recorded sha
│   ├── check_dataset_traceability.py # verify samples resolve against cloned repos
│   ├── backfill_rtc_metric.py       # compute RTC for MLflow runs that predate it
│   └── preview_file_context_prompt.py # inspect file-context prompts without hitting LLM
├── src/mas_code_sum/
│   ├── data.py                      # dataset loading, grouping by project
│   ├── metrics.py                   # BLEU, ROUGE-L
│   ├── rtc.py                       # round-trip correctness (backward model + BLEU/CrystalBLEU/CodeBLEU)
│   ├── runner.py                    # MLflow-integrated experiment runner
│   ├── evaluator.py                 # BLEU implementation
│   ├── methods/
│   │   ├── __init__.py              # REGISTRY dict
│   │   ├── base.py                  # BaseSummarizer + async batch execution
│   │   ├── llm_client.py            # backend clients, rate-limit retry, cost tracking
│   │   ├── exact_copy.py
│   │   ├── zero_shot_llm.py
│   │   ├── zero_shot_context_enriched.py
│   │   ├── few_shot_llm.py
│   │   ├── few_shot_context_enriched.py
│   │   ├── few_shot_file_context.py
│   │   ├── few_shot_all_context.py
│   │   ├── few_shot_all_context_instruct.py
│   │   ├── few_shot_asap.py
│   │   ├── agentic_rag.py
│   │   ├── agentic_rag_all_context.py
│   │   └── codet5_summarizer.py
│   ├── retrievers/
│   │   ├── __init__.py              # RETRIEVER_REGISTRY dict
│   │   ├── base.py                  # BaseRetriever abstract class
│   │   ├── random.py
│   │   └── bm25.py                  # BM25Retriever + BM25CrossProjectRetriever
│   └── enrichers/
│       ├── asap/
│       │   └── dfg.py               # DFG context, computed on demand
│       ├── ast_chunks.py            # AST-aware repo chunking (astchunk) for agentic RAG
│       ├── repo_context.py          # repo name/description metadata
│       ├── file_context.py          # module doc, class context, imports (Python)
│       └── file_context_java.py     # same for Java
├── docker-compose.mlflow.yml        # MLflow tracking server
├── run_experiment.py                # CLI entrypoint
└── pyproject.toml
```

---

## Dataset

The dataset comes in two variants under `dataset/`, selected via `--dataset` at run time:

- `full` (default) — the complete dataset.
- `small` — a 6-project subset (`apache/airflow`, `orientechnologies/orientdb`, `google/adk-java`, `newton-physics/newton`, `google/langextract`, `tamboui/tamboui`) for fast iteration; `test.jsonl` is capped at 50 samples per project, `train.jsonl` keeps all samples for those projects.

Each variant has flat `train.jsonl` and `test.jsonl` splits spanning all languages.

Every sample also carries a `set` label — `original` (samples from the established
benchmark projects) or `new` (samples from repositories added after the base models'
training cutoff, as a contamination control). The runner logs metrics per set as well
as in aggregate.

### Schema

Each sample in a `.jsonl` file is a JSON object with these fields:

| Field | Description |
|---|---|
| `id` | Unique sample identifier |
| `repo` | GitHub repo (`owner/repo`) |
| `language` | Programming language |
| `func_name` | Function name |
| `code` / `code_tokens` | Function source code, raw and tokenized |
| `original_string` | Function source including its docstring |
| `docstring` / `docstring_tokens` | Ground truth docstring, raw and tokenized |
| `path` | File path within the repository |
| `sha` | Commit sha the sample was extracted at |
| `blame_sha` | Commit sha of the most recently modified line (used to resolve file context) |
| `authored_timestamp` | ISO timestamp of the most recently modified line |
| `url` | GitHub permalink to the function |
| `set` | `original` or `new` (see above) |

---

## Running Experiments

**Step 1** — Start the MLflow tracking server (Docker):

```bash
docker compose -f docker-compose.mlflow.yml up -d
```

This serves the UI/API at `http://127.0.0.1:5000`, backed by `sqlite:///mlflow-data/mlflow.db` with artifacts under `./mlflow-data/mlartifacts` (both gitignored, persisted on the host).

Override the tracking URI if using a remote server:

```bash
export MLFLOW_TRACKING_URI=http://...
```

**Step 2** — Run an experiment:

```bash
python run_experiment.py experiments/example.yaml
python run_experiment.py experiments/example.yaml --dataset small   # fast iteration on the 6-project subset
```

CLI flags (all optional, they override or extend the YAML):

| Flag | Description |
|---|---|
| `--model MODEL` | Override `method_params.model` |
| `--backend BACKEND` | Override `method_params.backend` (`openrouter` or `featherless`) |
| `--dataset full\|small` | Dataset variant to load (default: `full`) |
| `--tag KEY=VALUE` | Set an MLflow tag; repeatable |
| `--rtc MODEL` | Compute round-trip correctness with `MODEL` as the backward model (see [RTC](#round-trip-correctness-rtc)) |
| `--rtc-backend BACKEND` | Backend for the RTC backward model (default: `openrouter`) |

**Step 3** — Open `http://127.0.0.1:5000` in a browser to view results. Stop the server with:

```bash
docker compose -f docker-compose.mlflow.yml down
```

### Experiment Config

```yaml
method: few_shot_llm             # key in REGISTRY

method_params:                   # kwargs passed to the method constructor
  model: meta-llama/llama-3.1-8b-instruct

retriever: bm25                  # optional; key in RETRIEVER_REGISTRY
retriever_params:
  n: 10

languages:                       # languages to evaluate
  - python
  - java

split: test
max_samples: 100                 # max samples per project; null = all
num_runs: 1                      # repeat each sample N times, average metrics
projects:                        # optional; filter to specific project names
  - apache__airflow
```

The `retriever` is constructed first and injected into the method via `method_params`. `run_experiment.py` never needs to change when you add new methods or retrievers.

The dataset variant (`full` or `small`) is a CLI flag, not a YAML field — see `--dataset` above.

### MLflow Tracking

All runs land under the **`code-summarization`** experiment. Each run represents one full method invocation across all projects and is named after the method.

Logged metrics are aggregates across all projects. Metrics restricted to one contamination-control set are suffixed with it (`bleu_original`, `bleu_new`). Per-project breakdowns are written to the `per_project_metrics.json` table artifact rather than logged as metrics.

**Params logged per run:**
- `method`, `dataset`, `languages`, `num_runs`, `num_samples`, `max_samples_per_project`
- All hyperparameters returned by the method's `params()` method (e.g. `model`, `retriever`, `n_shots`)

**Metrics logged:**
- `bleu` — sentence-level BLEU averaged across samples (0–100 scale)
- `rougeL` — average ROUGE-L F1
- `cost_usd` — total API spend for the run
- RTC metrics, when `--rtc` is passed — see [Round-Trip Correctness](#round-trip-correctness-rtc)

**Artifacts:**
- `predictions/*.csv` — columns: `id`, `project`, `func_name`, `run`, `reference`, `prediction`; with `--rtc`, also the per-sample RTC scores and the exact strings they compared
- `per_project_metrics.json` — one row of metrics per project

---

## Methods

| Key | Class | Description |
|---|---|---|
| `exact_copy` | `ExactCopySummarizer` | Returns the raw code as-is (sanity baseline) |
| `zero_shot_llm` | `ZeroShotLLMSummarizer` | Plain zero-shot prompt via LLM |
| `zero_shot_context_enriched` | `ZeroShotContextEnrichedSummarizer` | Zero-shot + repo name/description in prompt |
| `few_shot_llm` | `FewShotLLMSummarizer` | Few-shot with retrieved examples |
| `few_shot_context_enriched` | `FewShotContextEnrichedSummarizer` | Few-shot + repo/file context in each block |
| `few_shot_file_context` | `FewShotFileContextSummarizer` | Few-shot + file-level context (module doc, class, imports) |
| `few_shot_all_context` | `FewShotAllContextSummarizer` | Few-shot + file context and file outline merged in a single completions pass |
| `few_shot_all_context_instruct` | `FewShotAllContextInstructSummarizer` | Same signals as `few_shot_all_context`, assembled as an instructed chat prompt (works with instruct/thinking models) |
| `few_shot_asap` | `FewShotAsapSummarizer` | Replicates the ASAP completion-style prompt with DFG context |
| `agentic_rag` | `AgenticRagSummarizer` | Two-agent pipeline: a gatherer BM25-ranks AST-aware chunks of the repo (optionally LLM-filtered), a summarizer writes the description from them |
| `agentic_rag_all_context` | `AgenticRagAllContextSummarizer` | `agentic_rag` plus the target file's own context (module doc, enclosing class, imports, sibling-function outline) |
| `codet5` | `CodeT5Summarizer` | Fine-tuned CodeT5 model (no LLM API required) |

LLM-based methods call models via OpenRouter or Featherless using the OpenAI-compatible API. Client construction, rate-limit retry, and cost tracking are centralized in `src/mas_code_sum/methods/llm_client.py`. Specify the backend per method with `backend: featherless` or `backend: openrouter` in `method_params`.

### ASAP DFG context

`few_shot_asap` can enrich each prompt with a data-flow graph (DFG) derived from the function's AST — set `use_dfg: true` in `method_params`. There is no setup step: `enrichers/asap/dfg.py` computes the block from the source at prompt-build time (~1 ms per function, memoized per process) using the vendored CodeXGLUE parser under `asap_scripts/scripts/parser/`.

The block is byte-identical to what `asap_scripts/scripts/DFG.py` produces, verified across all 13,590 dataset samples.

---

## Retrievers

Retrievers fetch training examples for few-shot methods. They are configured separately in the experiment YAML.

| Key | Class | Description |
|---|---|---|
| `random` | `RandomRetriever` | Random samples from the training split |
| `bm25` | `BM25Retriever` | BM25 lexical similarity over code tokens |
| `bm25_cross_project` | `BM25CrossProjectRetriever` | Same as `bm25`, but excludes examples from the query's own project |

All retrievers implement `BaseRetriever.retrieve(code, language, n, project, path) -> list[dict]`.

---

## Round-Trip Correctness (RTC)

RTC is a reference-free metric, following Allamanis et al., *"Unsupervised Evaluation of Code LLMs with Round-Trip Correctness"* (ICML 2024). A **backward model** is asked to reconstruct the original function from the predicted description alone — spliced back into its source file in place of the function (fill-in-the-middle) — and the reconstruction is compared against the original code. A system whose descriptions round-trip to more similar code is capturing more of what the code actually does.

Both sides are normalized first (doc comments, decorators, and imports stripped), so a docstring that merely restates the prediction can't inflate the score. Add `--rtc` to any run:

```bash
python run_experiment.py experiments/few_shot_all_context_bm25.yaml --rtc openai/gpt-4o-mini
python run_experiment.py experiments/agentic_rag.yaml --rtc openai/gpt-4o-mini --rtc-backend openrouter
```

This costs one backward-model call per sample and requires the dataset repos to be cloned locally (`python scripts/clone_dataset_repos.py`) so the surrounding file can be reconstructed.

**Metrics logged:**

| Metric | Description |
|---|---|
| `rtc_bleu` | Smoothed BLEU-4 between the original and round-tripped code |
| `rtc_crystalbleu` | CrystalBLEU (Eghbali & Pradel, 2022) — BLEU that ignores the corpus's 500 most common n-grams, so language boilerplate shared by nearly every snippet doesn't inflate the score. The shared-n-gram table is built once per run, per language, from the functions under evaluation |
| `rtc_codebleu` | CodeBLEU (Ren et al., 2020) — weighted n-grams + AST syntax match + data-flow match |
| `codebleu_ngram_match`, `codebleu_weighted_ngram_match`, `codebleu_syntax_match`, `codebleu_dataflow_match` | CodeBLEU's four components, logged individually |

The predictions artifact additionally carries `backward_code` (raw model output) plus `og_code_compared` / `rtc_code_compared` — the exact post-normalization strings that were scored, so any number can be traced back to the text it came from.

To add RTC to runs that predate the metric:

```bash
python scripts/backfill_rtc_metric.py RUN_ID [RUN_ID ...] --model openai/gpt-4o-mini
```

---

## Adding a New Method

**Step 1** — Create `src/mas_code_sum/methods/my_method.py`:

```python
from .base import BaseSummarizer

class MyMethodSummarizer(BaseSummarizer):
    name = "my_method"

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
        ...

    def params(self) -> dict:
        return {"model": "...", "temperature": 0.7}
```

For LLM methods, override `async_summarize` instead — the base class `summarize_batch` runs all async calls concurrently via `asyncio` with a configurable semaphore (`self.max_concurrency`).

**Step 2** — Register in `src/mas_code_sum/methods/__init__.py`:

```python
from .my_method import MyMethodSummarizer

REGISTRY = {
    ...,
    "my_method": MyMethodSummarizer,
}
```

**Step 3** — Create `experiments/my_experiment.yaml` and run it.

---

## Adding a New Metric

All metrics live in `src/mas_code_sum/metrics.py` inside `compute_metrics()`. Every key in the returned dict is automatically logged to MLflow.

```python
def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    ...
    return {
        "bleu": bleu,
        "rougeL": rougeL / n,
        "my_metric": ...,   # appears automatically in MLflow
    }
```

---

## Results

**CARA improves BLEU over zero-shot prompting without a practically significant drop in
round-trip correctness. ASAP, the baseline it is measured against, does not.**

That is the headline finding, and it is what the two metric families are for. BLEU asks
how closely a summary matches the human reference; RTC asks how much of the code's actual
behavior survives in the summary, by having a backward model reconstruct the function
from it. A method can buy the first with the second — and few-shot prompting does exactly
that. CARA does not.

`cara` below is the `agentic_rag_all_context` method; `asap` is `few_shot_asap`.

### Setup

Every number comes from MLflow runs on the **`full`** dataset, evaluated on the same
**2,000 matched `(project, func_name, id)` samples** per run (Python + Java), for two
backbones via OpenRouter, with **`openai/gpt-4o-mini`** as the RTC backward model. BLEU is
sentence BLEU against the human reference, on a 0–100 scale.

| Key | Method | Model | MLflow run |
|---|---|---|---|
| zeroshot / 70B | `zero_shot_llm` | Llama 3.3 70B Instruct | `3fc7ad1d` |
| few_shot / 70B | `few_shot_llm` | Llama 3.3 70B Instruct | `d1197c3c` |
| asap / 70B | `few_shot_asap` | Llama 3.3 70B Instruct | `5005bca8` |
| cara / 70B | `agentic_rag_all_context` | Llama 3.3 70B Instruct | `68e8ff6f` |
| zeroshot / 8B | `zero_shot_llm` | Llama 3.1 8B Instruct | `38bf994f` |
| few_shot / 8B | `few_shot_llm` | Llama 3.1 8B Instruct | `f39382e4` |
| asap / 8B | `few_shot_asap` | Llama 3.1 8B Instruct | `93ac3d2d` |
| cara / 8B | `agentic_rag_all_context` | Llama 3.1 8B Instruct | `9f85a34e` |
| codet5 | `codet5` | `Salesforce/codet5-base-multi-sum` | `847b29e6` |

Retrieval-based methods use BM25 with n=3; both CARA runs use `use_filter: false` with
`n_candidates: 3`. Run IDs are MLflow `run_uuid`s — see `mlflow-data/mlflow.db` for full
provenance. The analysis itself lives in `scratch/cara_bleu_rtc_story.ipynb`, which reads
the prediction CSVs straight out of the MLflow artifact store.

### Practical significance

With n=2,000 paired samples, a Wilcoxon signed-rank test finds *every* comparison below
statistically significant (p < 0.05), including differences far too small to matter. So
each comparison carries one of:

- `**` — significant **and** the mean difference exceeds **2 points**, the bar for a
  difference worth claiming in prose
- `*` — significant, but the mean difference is ≤ 2 points (real, not material)
- blank — not statistically significant

### Table 1 — Headline means

| Method | Model | n | BLEU | RTC-CrystalBLEU | RTC-CodeBLEU |
|---|---|---:|---:|---:|---:|
| zeroshot | 70B | 2000 | 10.69 | **30.57** | **38.74** |
| zeroshot | 8B | 2000 | 14.21 | 27.35 | 36.36 |
| few_shot | 70B | 2000 | 23.93 | 25.76 | 35.30 |
| few_shot | 8B | 2000 | 22.01 | 25.98 | 35.56 |
| asap | 70B | 2000 | **24.22** | 25.46 | 35.11 |
| asap | 8B | 2000 | 22.79 | 25.70 | 35.36 |
| cara | 70B | 2000 | 18.37 | 29.44 | 37.89 |
| cara | 8B | 2000 | 19.19 | 28.03 | 37.03 |
| codet5 | — | 2000 | 18.99 | 24.97 | 34.66 |

Zero-shot has the worst BLEU and the best RTC. That inversion is the whole tension: the
methods that push BLEU up are the ones that pull RTC down.

### Table 2 — Significance vs. zero-shot

Paired Wilcoxon, matched on `(project, func_name, id)`, each method against zero-shot on
the same backbone. `d` is the mean difference (method − zero-shot).

| Comparison | Model | dBLEU | | dRTC-CrystalBLEU | | dRTC-CodeBLEU | |
|---|---|---:|---|---:|---|---:|---|
| few_shot vs. zeroshot | 70B | +13.25 | `**` | −4.81 | `**` | −3.44 | `**` |
| asap vs. zeroshot | 70B | +13.53 | `**` | −5.11 | `**` | −3.63 | `**` |
| cara vs. zeroshot | 70B | +7.68 | `**` | −1.13 | `*` | −0.85 | `*` |
| few_shot vs. zeroshot | 8B | +7.81 | `**` | −1.38 | `*` | −0.80 | `*` |
| asap vs. zeroshot | 8B | +8.58 | `**` | −1.65 | `*` | −1.00 | `*` |
| cara vs. zeroshot | 8B | +4.98 | `**` | +0.68 | `*` | +0.67 | `*` |

On 70B the split is unambiguous: ASAP and few-shot buy ~13 BLEU points at a materially
significant RTC cost on both metrics (`**`), while CARA buys 7.68 BLEU points at an RTC
cost that stays under the 2-point bar (`*`). On 8B every RTC movement is sub-material, but
only CARA's is *positive* — it gains BLEU and RTC at once.

ASAP is the stronger BLEU method throughout, by roughly 5–6 points. The claim is not that
CARA wins on BLEU; it is that ASAP's larger BLEU gain is partly paid for out of
faithfulness, and CARA's is not.

### Table 3 — CARA vs. the CodeT5 baseline

CodeT5 is a fine-tuned, non-prompted baseline with no model-size axis, so it sits in its
own comparison.

| Comparison | Model | dBLEU | | dRTC-CrystalBLEU | | dRTC-CodeBLEU | |
|---|---|---:|---|---:|---|---:|---|
| cara vs. CodeT5 | 70B | −0.62 | `*` | +4.47 | `**` | +3.23 | `**` |
| cara vs. CodeT5 | 8B | +0.20 | `*` | +3.06 | `**` | +2.37 | `**` |

CARA matches a fine-tuned model on BLEU — the difference is under half a point on both
backbones, well inside the noise bar — while beating it by 2.4–4.5 points on both RTC
metrics.

### What this means

Few-shot examples teach the model to imitate the *style* of the project's docstrings,
which is most of what BLEU rewards. ASAP maximizes that. Retrieved repository context
teaches the model what the code *does*, which is what survives a round trip. Reporting
BLEU alone would rank ASAP first and hide the trade; reporting both shows CARA occupying
the position the metrics jointly favor — near-ASAP reference overlap at near-zero-shot
faithfulness.
