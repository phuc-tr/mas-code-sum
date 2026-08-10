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

Methods that read a sample's surrounding source file (`few_shot_file_context`, `few_shot_all_context`, the `agentic_rag` family) and the RTC metric need the dataset repos checked out locally at their recorded shas:

```bash
python scripts/clone_dataset_repos.py
```

---

## Project Structure

```
mas-code-sum/
├── asap_scripts/                    # ASAP DFG parser (CodeXGLUE upstream)
│   ├── scripts/
│   │   ├── parser/                  # DFG extraction modules (Python + Java)
│   │   └── sitter-libs/             # tree-sitter grammar sources
│   ├── precompute_dfg.py            # generates dataset/{full,small}/{split}_dfg.json
│   └── setup_dfg_parser.sh          # one-time parser build
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
│       │   └── dfg_loader.py        # loads pre-computed DFG context from dataset/
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

### ASAP Setup

`few_shot_asap` enriches each prompt with a data-flow graph (DFG) derived from the function's AST. The DFG context is pre-computed and stored in `dataset/full/{language}/{split}_dfg.json` — these files are already committed, so no setup is needed for a standard run.

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

Output: `dataset/full/{language}/{split}_dfg.json` — a `{url: dfg_text}` mapping loaded at runtime by `enrichers/asap/dfg_loader.py`.

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

All numbers below come from MLflow runs on the **`full`** dataset (2,000 test samples, Python + Java, 10 `original` + 10 `new` projects), summarizer **Llama 3.1 8B Instruct** via OpenRouter, BM25 retrieval with n=3, RTC backward model **`openai/gpt-4o-mini`**.

### Aggregate

| Method | BLEU | ROUGE-L | RTC BLEU | RTC CrystalBLEU | RTC CodeBLEU | Cost |
|---|---:|---:|---:|---:|---:|---:|
| `few_shot_asap` | **22.84** | **0.42** | 31.32 | 27.95 | 35.06 | $0.41 |
| `agentic_rag_all_context` | 21.43 | **0.42** | **33.64** | **30.14** | **36.66** | $0.50 |
| `few_shot_all_context_instruct` | 19.50 | 0.40 | 33.44 | — | 36.65 | $0.42 |
| `zero_shot_llm` | 15.37 | 0.35 | 32.77 | 29.34 | 35.97 | $0.34 |

The two metric families disagree, which is the point of having both: ASAP's DFG-style prompt wins on n-gram overlap with the reference docstring, while the context-gathering methods produce descriptions that round-trip back to more faithful code. `zero_shot_llm` scores far worse against references but nearly as well under RTC — a sign that much of the reference-based gap is stylistic conformity to the project's docstring conventions rather than semantic content.

CrystalBLEU is blank where a run predates the metric; rerun with `--rtc` to fill it in.

### Contamination control — `original` vs `new` projects

`new` projects were added after the base models' training cutoff. Scores are consistently *higher* there, so the gap is not evidence of memorization on `original`; the newer repos simply have more formulaic docstrings.

| Method | BLEU (original) | BLEU (new) | RTC BLEU (original) | RTC BLEU (new) |
|---|---:|---:|---:|---:|
| `few_shot_asap` | **21.11** | **24.57** | 30.97 | 31.67 |
| `agentic_rag_all_context` | 20.47 | 22.40 | **32.83** | **34.45** |
| `few_shot_all_context_instruct` | 18.66 | 20.35 | 32.49 | 34.39 |
| `zero_shot_llm` | 14.64 | 16.10 | 32.08 | 33.45 |

### Per-project BLEU

| Project | Set | `zero_shot_llm` | `few_shot_all_context_instruct` | `agentic_rag_all_context` | `few_shot_asap` |
|---|---|---:|---:|---:|---:|
| apache/airflow | original | 13.16 | 17.04 | 17.61 | **20.11** |
| vaexio/vaex | original | 16.30 | 26.24 | **26.28** | 20.26 |
| Qiskit/qiskit-terra | original | 16.13 | 20.36 | **25.84** | 23.17 |
| PyCQA/pylint | original | 15.91 | 17.39 | 18.23 | **21.55** |
| h2oai/h2o-3 | original | 15.01 | 15.82 | 18.28 | **20.84** |
| oblac/jodd | original | 14.25 | 23.74 | **26.27** | 25.36 |
| orientechnologies/orientdb | original | 12.78 | 15.08 | 15.74 | **18.14** |
| real-logic/aeron | original | 15.47 | 22.66 | 25.64 | **27.24** |
| spring-projects/spring-security | original | 14.70 | 14.18 | 15.96 | **17.70** |
| wildfly/wildfly | original | 12.64 | 14.08 | 14.85 | **16.74** |
| google/adk-java | new | 24.07 | 36.15 | **36.91** | 35.31 |
| google/langextract | new | 16.12 | 15.51 | 16.68 | **21.88** |
| newton-physics/newton | new | 15.48 | 16.23 | 19.87 | **27.40** |
| tamboui/tamboui | new | 20.52 | 37.64 | **41.87** | 31.92 |
| MemPalace/mempalace | new | 12.25 | 13.87 | 15.33 | **16.08** |
| agentscope-ai/QwenPaw | new | 14.45 | 15.50 | 16.20 | **20.42** |
| floci-io/floci | new | 12.63 | 14.22 | 16.24 | **16.87** |
| iflytek/astron-agent | new | 13.01 | 17.45 | 18.98 | **32.16** |
| opendataloader-project/opendataloader-pdf | new | 18.14 | 20.33 | 23.20 | **23.89** |
| vllm-project/vllm-omni | new | 14.30 | 16.60 | 18.70 | **19.73** |

Per-project metrics are not logged as MLflow metrics — they land in the `per_project_metrics.json` table artifact on each run.

### Ablations — `small` dataset

Run on the 6-project subset (300 samples), so these are directional only and not comparable to the tables above.

| Method | Model | Config | BLEU | RTC BLEU |
|---|---|---|---:|---:|
| `few_shot_asap` | Llama 3.1 8B Instruct | n=3 | **22.86** | 30.34 |
| `agentic_rag_all_context` | Llama 3.1 8B Instruct | no filter, 5 chunks | 21.69 | 34.09 |
| `agentic_rag_all_context` | Llama 3.1 8B Instruct | LLM filter, 3 of 5 chunks | 20.33 | 33.66 |
| `few_shot_asap` | Llama 3.3 70B Instruct | n=3 | 20.11 | 28.11 |
| `few_shot_all_context_instruct` | Llama 3.1 8B Instruct | n=3 | 19.43 | 32.92 |
| `agentic_rag` | Llama 3.1 8B Instruct | no filter, few-shot | 18.87 | 33.80 |
| `agentic_rag_all_context` | Llama 3.3 70B Instruct | no filter, 5 chunks | 18.74 | 34.56 |
| `agentic_rag` | Llama 3.1 8B Instruct | LLM filter, few-shot | 17.47 | 33.60 |
| `zero_shot_llm` | Llama 3.1 8B Instruct | — | 15.55 | 26.95 |
| `agentic_rag` | Llama 3.1 8B Instruct | LLM filter, no few-shot | 11.85 | **35.91** |

Two consistent findings across the ablations: the LLM chunk filter does not pay for itself (it costs roughly 2x and scores slightly lower than passing all BM25 hits through), and dropping few-shot examples collapses BLEU while *improving* RTC — without examples the model stops imitating docstring style and describes the code more literally.
