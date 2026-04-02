# mas-code-sum

A framework for experimenting with code summarization methods and tracking results with MLflow.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
  - [Data Collection](#data-collection)
  - [Train/Test Split](#traintest-split)
  - [Schema](#schema)
- [Running Experiments](#running-experiments)
  - [Experiment Config](#experiment-config)
  - [MLflow Tracking](#mlflow-tracking)
- [Adding a New Method](#adding-a-new-method)
- [Adding a New Metric](#adding-a-new-metric)

---

## Project Structure

```
mas-code-sum/
├── dataset/
│   ├── collect_data.py          # Step 1: pull data from HuggingFace + git blame
│   ├── split_jsonl_by_blame.py  # Step 2: split into train/test by blame timestamp
│   ├── python.jsonl             # raw collected samples per language
│   ├── python/
│   │   ├── train.jsonl
│   │   └── test.jsonl           # ← used by experiments
│   └── ...                      # same structure for java, go, etc.
├── experiments/
│   └── example.yaml             # experiment config
├── src/mas_code_sum/
│   ├── data.py                  # dataset loading, grouping by project
│   ├── metrics.py               # BLEU + ROUGE scoring
│   ├── runner.py                # MLflow-integrated experiment runner
│   └── methods/
│       ├── __init__.py          # method registry (REGISTRY dict)
│       ├── base.py              # BaseSummarizer abstract class
│       └── exact_copy.py        # example baseline method
├── run_experiment.py            # CLI entrypoint
└── pyproject.toml
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

---

## Dataset

### Data Collection

`dataset/collect_data.py` builds the raw dataset from scratch. It:

1. Loads the [`code-search-net/code_search_net`](https://huggingface.co/datasets/code-search-net/code_search_net) test split from HuggingFace.
2. Picks repositories that have between 30–50 samples per language (5 repos per language, seeded for reproducibility).
3. For each repo, runs `git blame` on every function to record the `latest_blame_timestamp` — the last time that code was touched.
4. Writes one `{language}.jsonl` file per language under `dataset/`.

```bash
cd dataset
python collect_data.py
```

Requires `git` to be installed. Repos are cloned temporarily and deleted after blame is collected.

### Train/Test Split

`dataset/split_jsonl_by_blame.py` reads the raw `.jsonl` files and splits them per repository by blame recency:

- The **N=30 most recently modified** functions per repo → `test.jsonl`
- The rest → `train.jsonl`

This simulates a realistic evaluation scenario where the test set contains newer code than the training set.

```bash
cd dataset
python split_jsonl_by_blame.py
```

Output is written to `dataset/{language}/train.jsonl` and `dataset/{language}/test.jsonl`.

### Schema

Each sample in a `.jsonl` file is a JSON object with the following fields:

| Field | Description |
|---|---|
| `repository_name` | GitHub repo (`owner/repo`) — used to group samples into projects |
| `language` | Programming language |
| `func_name` | Function name |
| `func_code_string` | Full function source code |
| `func_documentation_string` | Ground truth docstring/summary |
| `func_code_url` | GitHub permalink to the function |
| `latest_blame_timestamp` | ISO timestamp of the most recently modified line in the function |
| `split_name` | `"train"` or `"test"` |

---

## Running Experiments

```bash
python run_experiment.py experiments/example.yaml
```

Then open the MLflow UI to browse results:

```bash
mlflow ui --backend-store-uri mlruns/
```

### Experiment Config

Each experiment is defined by a YAML file in `experiments/`:

```yaml
method: zero_shot_llm                  # key in REGISTRY (see src/mas_code_sum/methods/__init__.py)

method_params:                         # passed as kwargs to the method constructor (optional)
  model: meta-llama/llama-3.1-8b-instruct

languages:                             # which languages to load samples from
  - python
  - java
  - javascript
  - go
  - php
  - ruby

split: test                            # which split to evaluate on (train or test)
max_samples: 100                       # max samples per project; set to null for all
```

To run a different method or model, create a new YAML pointing at it. All runs always land in the same MLflow experiment.

### MLflow Tracking

All runs are collected under a single MLflow experiment: **`code-summarization`**.

Each **run** represents one `(method, project)` pair and is named `{method}/{project}` (e.g. `zero_shot_llm/ekzhu/datasketch`). This means you can filter runs by the `method` param in the UI and compare the same project across different methods side by side.

Every run logs:

**Params**
- `method` — method name
- `project` — repository name
- `language` — programming language of the project
- `split` — dataset split used
- `num_samples` — number of samples evaluated
- `max_samples_per_project` — cap from the config
- any hyperparameters returned by the method's `params()` method (e.g. `model`)

**Metrics**
- `bleu` — corpus BLEU with smoothing
- `rouge1`, `rouge2`, `rougeL` — average F1 across samples

**Artifacts**
- `predictions/` — a CSV with columns `func_name`, `reference`, `prediction`

---

## Adding a New Method

**Step 1** — Create `src/mas_code_sum/methods/my_method.py`:

```python
from .base import BaseSummarizer

class MyMethodSummarizer(BaseSummarizer):
    name = "my_method"

    def summarize(self, code: str, language: str) -> str:
        # generate and return a summary string
        ...

    def params(self) -> dict:
        # return any hyperparameters to log in MLflow
        return {"model": "...", "temperature": 0.7}
```

The interface is minimal:
- `name` — string identifier, used as a label in MLflow
- `summarize(code, language) -> str` — given source code and its language, return a summary
- `params() -> dict` — optional; return hyperparameters to log

**Step 2** — Register it in `src/mas_code_sum/methods/__init__.py`:

```python
from .exact_copy import ExactCopySummarizer
from .my_method import MyMethodSummarizer

REGISTRY = {
    "exact_copy": ExactCopySummarizer,
    "my_method": MyMethodSummarizer,
}
```

**Step 3** — Create an experiment config in `experiments/my_experiment.yaml` and run it.

`run_experiment.py` never needs to change.

---

## Adding a New Metric

All metrics are computed in `src/mas_code_sum/metrics.py` inside `compute_metrics()`, which takes lists of predictions and references and returns a flat `dict[str, float]`. Every key in that dict is automatically logged to MLflow.

To add a new metric, extend the returned dict:

```python
from some_library import compute_codebleu

def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    ...
    codebleu = compute_codebleu(references, predictions)

    return {
        "bleu": bleu,
        "rouge1": rouge1 / n,
        "rouge2": rouge2 / n,
        "rougeL": rougeL / n,
        "codebleu": codebleu,   # ← new metric appears automatically in MLflow
    }
```
