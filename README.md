# mas-code-sum

A framework for experimenting with code summarization methods and tracking results with MLflow.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
  - [Standard Dataset](#standard-dataset)
  - [Same-project Dataset](#same-project-dataset)
  - [Schema](#schema)
- [Running Experiments](#running-experiments)
  - [Experiment Config](#experiment-config)
  - [MLflow Tracking](#mlflow-tracking)
- [Methods](#methods)
- [Retrievers](#retrievers)
- [Adding a New Method](#adding-a-new-method)
- [Adding a New Metric](#adding-a-new-metric)

---

## Project Structure

```
mas-code-sum/
├── dataset/
│   ├── collect_data.py          # Step 1: pull data from HuggingFace + git blame
│   ├── split_jsonl_by_blame.py  # Step 2: split into train/test by blame timestamp
│   ├── python/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── Same-project/            # same-project dataset (one subdir per project)
│   │   └── {project}/
│   │       ├── train.jsonl
│   │       └── test.jsonl
│   └── repos/                   # cloned repos (used by file_context enricher)
├── experiments/
│   └── *.yaml                   # experiment configs
├── scripts/                     # one-off analysis and backfill scripts
├── src/mas_code_sum/
│   ├── data.py                  # dataset loading, grouping by project
│   ├── metrics.py               # BLEU, ROUGE-L, BERTScore
│   ├── runner.py                # MLflow-integrated experiment runner
│   ├── evaluator.py             # BLEU implementation
│   ├── style_guide.py           # style guide construction helper
│   ├── methods/
│   │   ├── __init__.py          # REGISTRY dict
│   │   ├── base.py              # BaseSummarizer + async batch execution
│   │   ├── exact_copy.py
│   │   ├── zero_shot_llm.py
│   │   ├── zero_shot_context_enriched.py
│   │   ├── few_shot_llm.py
│   │   ├── few_shot_context_enriched.py
│   │   ├── few_shot_file_context.py
│   │   ├── few_shot_critic.py
│   │   ├── few_shot_asap.py
│   │   ├── codet5_summarizer.py
│   │   └── style_guided.py
│   ├── retrievers/
│   │   ├── __init__.py          # RETRIEVER_REGISTRY dict
│   │   ├── base.py              # BaseRetriever abstract class
│   │   ├── random.py
│   │   ├── random_same_project.py
│   │   ├── bm25.py
│   │   ├── directory.py
│   │   └── hyde.py
│   └── enrichers/
│       ├── dfg_loader.py        # data-flow graph features
│       ├── identifier_extractor.py
│       ├── file_context.py      # module doc, class context, imports (Python)
│       └── file_context_java.py # same for Java
├── run_experiment.py            # CLI entrypoint
└── pyproject.toml
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

LLM methods call models via [OpenRouter](https://openrouter.ai). Set the API key before running:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

MLflow tracking URI defaults to `http://127.0.0.1:5000`. Override with:

```bash
export MLFLOW_TRACKING_URI=http://...
```

---

## Dataset

### Standard Dataset

`dataset/collect_data.py` builds the raw dataset from scratch. It:

1. Loads the [`code-search-net/code_search_net`](https://huggingface.co/datasets/code-search-net/code_search_net) test split from HuggingFace.
2. Picks repositories that have between 30–50 samples per language (5 repos per language, seeded for reproducibility).
3. For each repo, runs `git blame` on every function to record the `latest_blame_timestamp`.
4. Writes one `{language}.jsonl` file per language under `dataset/`.

```bash
cd dataset && python collect_data.py
```

`dataset/split_jsonl_by_blame.py` then splits each file per repo by blame recency:

- The **N=30 most recently modified** functions per repo → `test.jsonl`
- The rest → `train.jsonl`

```bash
cd dataset && python split_jsonl_by_blame.py
```

### Same-project Dataset

The `dataset/Same-project/` directory holds a separate benchmark where the train and test sets come from the same repositories (used to study in-project retrieval). Each subdirectory is one project and contains its own `train.jsonl` / `test.jsonl`.

To use this dataset in an experiment, set `dataset: same-project` in the config (and omit `languages`).

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

```bash
python run_experiment.py experiments/example.yaml
```

Then open the MLflow UI:

```bash
mlflow ui
```

### Experiment Config

```yaml
method: few_shot_context_enriched   # key in REGISTRY

method_params:                       # kwargs passed to the method constructor
  model: meta-llama/llama-3.1-8b-instruct

retriever: bm25                      # optional; key in RETRIEVER_REGISTRY
retriever_params:
  n: 10

dataset: standard                    # "standard" (language-based) or "same-project"

languages:                           # required when dataset=standard
  - python
  - java

split: test
max_samples: 100                     # max samples per project; null = all
num_runs: 1                          # repeat each sample N times, average metrics
projects:                            # optional; filter to specific project names
  - apache__airflow
```

The `retriever` is constructed first and injected into the method via `method_params`. `run_experiment.py` never needs to change when you add new methods or retrievers.

### MLflow Tracking

All runs land under the **`code-summarization`** experiment. Each run represents one full method invocation across all projects and is named after the method.

Per-project metrics are logged with a `{project}/` prefix. Aggregate metrics (no prefix) summarise across all projects.

**Params logged per run:**
- `method`, `dataset`, `languages`, `split`, `num_runs`, `num_samples`, `max_samples_per_project`
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
| `few_shot_critic` | `FewShotCriticSummarizer` | Generate then self-critique and refine |
| `few_shot_asap` | `FewShotAsapSummarizer` | Replicates the ASAP completion-style prompt |
| `codet5` | `CodeT5Summarizer` | Fine-tuned CodeT5 model (no LLM API required) |
| `style_guided` | `StyleGuidedSummarizer` | Few-shot with a project-level style guide derived from training summaries |

LLM-based methods call models via OpenRouter using the OpenAI-compatible API. The base URL and client construction are centralized in `src/mas_code_sum/methods/base.py`.

---

## Retrievers

Retrievers fetch training examples for few-shot methods. They are configured separately in the experiment YAML.

| Key | Class | Description |
|---|---|---|
| `random` | `RandomRetriever` | Random samples from the training split |
| `random_same_project` | `RandomSameProjectRetriever` | Random samples from the same project |
| `bm25` | `BM25Retriever` | BM25 lexical similarity over code tokens |
| `directory` | `DirectoryRetriever` | Examples from the same directory as the query |
| `hyde` | `HyDERetriever` | HyDE: generate a hypothetical docstring, retrieve by BM25 over docstring tokens |

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
