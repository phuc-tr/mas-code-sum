<div align="center">

# mas-code-sum

**Multi-method code summarization research framework**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

Plug-in methods · BM25 retrieval · BLEU · ROUGE-L · BERTScore

</div>

---

## Overview

`mas-code-sum` is an experiment framework for studying LLM-based code summarization. It provides:

- **Pluggable methods** — zero-shot, few-shot, ASAP, file-context enriched, CodeT5
- **Pluggable retrievers** — BM25, random
- **MLflow integration** — every run logs params, per-project metrics, and prediction artifacts automatically
- **YAML-driven experiments** — swap models, retrievers, and datasets without touching code

All runs are tracked in MLflow under the **`code-summarization`** experiment.

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
  - [ASAP Setup](#asap-setup)
- [Retrievers](#retrievers)
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

---

## Project Structure

```
mas-code-sum/
├── asap_scripts/                    # ASAP DFG parser (CodeXGLUE upstream)
│   ├── scripts/
│   │   ├── parser/                  # DFG extraction modules (Python + Java)
│   │   └── sitter-libs/             # tree-sitter grammar sources
│   ├── precompute_dfg.py            # generates dataset/{version}/{lang}/{split}_dfg.json
│   └── setup_dfg_parser.sh          # one-time parser build
├── dataset/
│   ├── v1/                          # default dataset version (includes DFG files)
│   │   ├── python/
│   │   │   ├── train.jsonl
│   │   │   ├── test.jsonl
│   │   │   ├── train_dfg.json       # pre-computed DFG context (ASAP)
│   │   │   └── test_dfg.json
│   │   └── java/
│   │       ├── train.jsonl
│   │       ├── test.jsonl
│   │       ├── train_dfg.json
│   │       └── test_dfg.json
│   └── v2/                          # alternate dataset version
│       ├── python/
│       └── java/
├── experiments/
│   ├── *.yaml                       # experiment configs (v1 by default)
│   └── v2/                          # configs targeting the v2 dataset
├── scripts/                         # pipeline utilities
│   ├── clone_dataset_repos.py       # clone each dataset repo at its recorded sha
│   ├── check_dataset_traceability.py # verify samples resolve against cloned repos
│   └── preview_file_context_prompt.py # inspect file-context prompts without hitting LLM
├── src/mas_code_sum/
│   ├── data.py                      # dataset loading, grouping by project
│   ├── metrics.py                   # BLEU, ROUGE-L, BERTScore
│   ├── runner.py                    # MLflow-integrated experiment runner
│   ├── evaluator.py                 # BLEU implementation
│   ├── methods/
│   │   ├── __init__.py              # REGISTRY dict
│   │   ├── base.py                  # BaseSummarizer + async batch execution
│   │   ├── exact_copy.py
│   │   ├── zero_shot_llm.py
│   │   ├── zero_shot_context_enriched.py
│   │   ├── few_shot_llm.py
│   │   ├── few_shot_context_enriched.py
│   │   ├── few_shot_file_context.py
│   │   ├── few_shot_all_context.py
│   │   ├── few_shot_asap.py
│   │   └── codet5_summarizer.py
│   ├── retrievers/
│   │   ├── __init__.py              # RETRIEVER_REGISTRY dict
│   │   ├── base.py                  # BaseRetriever abstract class
│   │   ├── random.py
│   │   └── bm25.py
│   └── enrichers/
│       ├── asap/
│       │   └── dfg_loader.py        # loads pre-computed DFG context from dataset/
│       ├── file_context.py          # module doc, class context, imports (Python)
│       └── file_context_java.py     # same for Java
├── run_experiment.py                # CLI entrypoint
└── pyproject.toml
```

---

## Dataset

The dataset is organized into versioned subdirectories under `dataset/`. The default version is `v1`, which includes pre-computed DFG context files used by the ASAP method.

Each version contains one directory per language (`python`, `java`) with `train.jsonl` and `test.jsonl` splits.

### Schema

Each sample in a `.jsonl` file is a JSON object with these fields:

| Field | Description |
|---|---|
| `id` | Unique sample identifier |
| `repo` | GitHub repo (`owner/repo`) |
| `language` | Programming language |
| `func_name` | Function name |
| `code_tokens` | Tokenized function source code |
| `docstring_tokens` | Tokenized ground truth docstring |
| `path` | File path within the repository |
| `url` | GitHub permalink to the function |
| `latest_blame_timestamp` | ISO timestamp of the most recently modified line |

---

## Running Experiments

**Step 1** — Start the MLflow tracking server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Override the tracking URI if using a remote server:

```bash
export MLFLOW_TRACKING_URI=http://...
```

**Step 2** — Run an experiment:

```bash
python run_experiment.py experiments/example.yaml
```

**Step 3** — Open the MLflow UI to view results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Experiment Config

```yaml
method: few_shot_llm             # key in REGISTRY

method_params:                   # kwargs passed to the method constructor
  model: meta-llama/llama-3.1-8b-instruct

retriever: bm25                  # optional; key in RETRIEVER_REGISTRY
retriever_params:
  n: 10

dataset_version: v1              # dataset subdirectory (default: v1)

languages:                       # language subdirectories to evaluate
  - python
  - java

split: test
max_samples: 100                 # max samples per project; null = all
num_runs: 1                      # repeat each sample N times, average metrics
projects:                        # optional; filter to specific project names
  - apache__airflow
```

The `retriever` is constructed first and injected into the method via `method_params`. `run_experiment.py` never needs to change when you add new methods or retrievers.

### MLflow Tracking

All runs land under the **`code-summarization`** experiment. Each run represents one full method invocation across all projects and is named after the method.

Per-project metrics are logged with a `{project}/` prefix. Aggregate metrics (no prefix) summarise across all projects.

**Params logged per run:**
- `method`, `dataset`, `dataset_version`, `languages`, `split`, `num_runs`, `num_samples`, `max_samples_per_project`
- All hyperparameters returned by the method's `params()` method (e.g. `model`, `retriever`, `n_shots`)

**Metrics logged:**
- `bleu` — sentence-level BLEU averaged across samples (0–100 scale)
- `rougeL` — average ROUGE-L F1
- `bertscore_f1` — average BERTScore F1 (roberta-large)

**Artifacts:**
- `predictions/*.csv` — columns: `id`, `project`, `func_name`, `run`, `reference`, `prediction`

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
| `few_shot_asap` | `FewShotAsapSummarizer` | Replicates the ASAP completion-style prompt with DFG context |
| `codet5` | `CodeT5Summarizer` | Fine-tuned CodeT5 model (no LLM API required) |

LLM-based methods call models via OpenRouter or Featherless using the OpenAI-compatible API. The backend and client construction are centralized in `src/mas_code_sum/methods/base.py`. Specify the backend per method with `backend: featherless` or `backend: openrouter` in `method_params`.

### ASAP Setup

`few_shot_asap` enriches each prompt with a data-flow graph (DFG) derived from the function's AST. The DFG context is pre-computed and stored in `dataset/v1/{language}/{split}_dfg.json` — these files are already committed, so no setup is needed for a standard run.

To regenerate the DFG files (e.g. after adding new dataset samples):

**Step 1 — Build the tree-sitter parser** (one-time):

```bash
bash asap_scripts/setup_dfg_parser.sh
```

This clones the required tree-sitter grammar repos into `asap_scripts/scripts/sitter-libs/` and compiles the parser shared library.

**Step 2 — Pre-compute DFG context**:

```bash
python asap_scripts/precompute_dfg.py
```

Options:

```bash
python asap_scripts/precompute_dfg.py --languages python java --splits train test
```

Output: `dataset/v1/{language}/{split}_dfg.json` — a `{url: dfg_text}` mapping loaded at runtime by `enrichers/asap/dfg_loader.py`.

---

## Retrievers

Retrievers fetch training examples for few-shot methods. They are configured separately in the experiment YAML.

| Key | Class | Description |
|---|---|---|
| `random` | `RandomRetriever` | Random samples from the training split |
| `bm25` | `BM25Retriever` | BM25 lexical similarity over code tokens |

All retrievers implement `BaseRetriever.retrieve(code, language, n, project, path) -> list[dict]`.

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
        "bertscore_f1": bertscore_f1,
        "my_metric": ...,   # appears automatically in MLflow
    }
```

---

## Results

### Per-project BLEU — Llama 3.1 8B (base), BM25 n=10

| Project | CodeT5 | FewShot LLM | ASAP | All Context |
|---|---:|---:|---:|---:|
| apache/airflow | 18.88 | 27.91 | 24.05 | **33.39** |
| vaexio/vaex | 18.13 | 25.02 | 23.78 | **30.25** |
| Qiskit/qiskit-terra | 21.21 | 27.48 | 24.81 | **34.08** |
| PyCQA/pylint | 19.80 | 21.89 | 20.95 | **23.61** |
| h2oai/h2o-3 | 16.36 | 22.87 | 21.45 | **26.78** |
| oblac/jodd | 18.16 | 20.67 | 25.34 | **30.89** |
| orientechnologies/orientdb | 18.97 | 27.24 | 20.81 | **32.59** |
| real-logic/aeron | 17.11 | 22.44 | 20.03 | **27.49** |
| spring-projects/spring-security | 16.01 | 19.85 | 19.91 | **25.71** |
| wildfly/wildfly | 16.06 | 21.54 | 18.21 | **25.98** |
| **Aggregate** | 18.07 | 23.69 | 21.93 | **29.08** |

### Aggregate BLEU — all models

| Method | Model | n | BLEU |
|---|---|---:|---:|
| `few_shot_all_context` | Llama 3.1 8B (base) | 10 | **29.08** |
| `few_shot_all_context` | Llama 3.3 70B Instruct | 10 | 27.49 |
| `few_shot_all_context` | Qwen3 8B | 10 | 25.05 |
| `few_shot_all_context` | Llama 3.1 8B Instruct | 10 | 25.00 |
| `few_shot_llm` | Llama 3.1 8B (base) | 10 | 23.69 |
| `few_shot_asap` | Llama 3.1 8B (base) | 3 | 21.93 |
| `few_shot_asap` | Llama 3.1 8B Instruct | 3 | 20.67 |
| `few_shot_llm` | Llama 3.1 8B Instruct | 10 | 20.66 |
| `codet5` | CodeT5-base-multi-sum | — | 18.07 |
