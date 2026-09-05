"""Experiment runner with MLflow tracking."""

import csv
import inspect
import io
import os
import tempfile
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.openai.autolog()

EXPERIMENT_NAME = "code-summarization"

from .data import load_projects
from .methods.llm_client import cost_tracker
from .metrics import compute_metrics
from .methods.base import BaseSummarizer
from .rtc import DEFAULT_BACKWARD_BACKEND, compute_rtc_scores_sync


def run_experiment(
    method: BaseSummarizer,
    languages: list[str] | None = None,
    max_samples: int | None = None,
    num_runs: int = 1,
    projects: list[str] | None = None,
    dataset: str = "full",
    tags: dict[str, str] | None = None,
    cli_command: str | None = None,
    rtc_model: str | None = None,
    rtc_backend: str = DEFAULT_BACKWARD_BACKEND,
) -> None:
    """
    Run a summarization experiment across all projects found in the given languages.

    All runs land in the single "code-summarization" MLflow experiment.
    Each run represents one method invocation, named after the method.
    Per-project metrics are logged as rows in the "per_project_metrics.json"
    table artifact (one row per project, one column per metric); aggregate
    metrics are logged as top-level run metrics.

    Each sample is summarized `num_runs` times and metrics are averaged across
    runs to reduce variance from stochastic generation.

    `cli_command`, if given, is stored as the "cli_command" tag so a run can be
    reproduced exactly from the MLflow UI.

    `rtc_model`, if given, additionally computes round-trip correctness (see
    `rtc.py`) using this model for the backward (description -> code) pass,
    logged as the `rtc_bleu` metric alongside `bleu`/`rougeL`. Left as None,
    no backward-pass calls are made and no `rtc_bleu` metric is logged.
    """
    if not languages:
        raise ValueError("'languages' is required.")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading projects for languages: {languages} (dataset {dataset})...")
    projects = load_projects(languages, max_samples_per_project=max_samples, dataset=dataset, projects=projects)
    print(f"Found {len(projects)} projects.")

    artifact_rows: list[tuple[dict, int, str, str, dict | None]] = []

    all_samples: list[dict] = []
    all_references: list[str] = []
    # per-sample predictions accumulated across runs: index -> list[str]
    all_predictions_by_run: list[list[str]] = []
    # per-sample rtc detail dicts accumulated across runs: index -> list[dict] (only populated if rtc_model is set)
    all_rtc_detail_by_run: list[list[dict]] = []

    _batch_supports_gt = "ground_truths" in inspect.signature(method.summarize_batch).parameters
    _batch_supports_blame = "blame_timestamps" in inspect.signature(method.summarize_batch).parameters
    _batch_supports_blame_sha = "blame_shas" in inspect.signature(method.summarize_batch).parameters
    _batch_supports_func_names = "func_names" in inspect.signature(method.summarize_batch).parameters

    with mlflow.start_run(run_name=method.name):
        cost_tracker.reset()
        mlflow.log_params({
            "method": method.name,
            "dataset": dataset,
            "languages": str(languages),
            "max_samples_per_project": max_samples or "all",
            "num_runs": num_runs,
            **method.params(),
        })
        mlflow.set_tag("method", method.name)
        mlflow.set_tag("dataset", dataset)
        if cli_command:
            mlflow.set_tag("cli_command", cli_command)
        if tags:
            mlflow.set_tags(tags)
        if rtc_model:
            mlflow.log_params({"rtc_model": rtc_model, "rtc_backend": rtc_backend})

        if hasattr(method, "validate_projects"):
            method.validate_projects(list(projects.keys()))

        for project, samples in projects.items():
            references = [" ".join(s["docstring_tokens"]) for s in samples]
            codes = [" ".join(s["code_tokens"]) for s in samples]
            langs = [s["language"] for s in samples]
            paths = [s.get("path") for s in samples]
            urls = [s.get("url") for s in samples]
            blame_timestamps = [s.get("authored_timestamp") for s in samples]
            blame_shas = [s.get("blame_sha") for s in samples]
            # Dataset-qualified 'Class.method' — disambiguates same-named methods
            # across nested/sibling classes when resolving the enclosing class.
            func_names = [s.get("func_name") for s in samples]

            # Collect predictions across all runs for this project
            project_run_predictions: list[list[str]] = []
            project_run_rtc_detail: list[list[dict]] = []
            for run_idx in range(num_runs):
                batch_kwargs = dict(
                    codes=codes,
                    languages=langs,
                    projects=[project] * len(samples),
                    paths=paths,
                    urls=urls,
                )
                if _batch_supports_gt:
                    batch_kwargs["ground_truths"] = references
                if _batch_supports_blame:
                    batch_kwargs["blame_timestamps"] = blame_timestamps
                if _batch_supports_blame_sha:
                    batch_kwargs["blame_shas"] = blame_shas
                if _batch_supports_func_names:
                    batch_kwargs["func_names"] = func_names
                preds = method.summarize_batch(**batch_kwargs)
                project_run_predictions.append(preds)
                rtc_detail = compute_rtc_scores_sync(samples, preds, rtc_model, rtc_backend) if rtc_model else None
                if rtc_detail is not None:
                    project_run_rtc_detail.append(rtc_detail)
                for i, (sample, pred, ref) in enumerate(zip(samples, preds, references)):
                    artifact_rows.append((sample, run_idx, pred, ref, rtc_detail[i] if rtc_detail else None))

            # Average metrics across runs
            run_metrics = [
                compute_metrics(preds, references)
                for preds in project_run_predictions
            ]
            if rtc_model:
                for m, detail in zip(run_metrics, project_run_rtc_detail):
                    for key in _RTC_METRIC_KEYS:
                        m[key] = sum(d[key] for d in detail) / len(detail)
            avg_metrics = {
                k: sum(m[k] for m in run_metrics) / num_runs
                for k in run_metrics[0]
            }
            print(f"  [{project}] {avg_metrics}")
            mlflow.log_table(
                {"project": [project], **{k: [v] for k, v in avg_metrics.items()}},
                artifact_file="per_project_metrics.json",
            )

            all_samples.extend(samples)
            all_references.extend(references)
            all_predictions_by_run.extend(zip(*project_run_predictions))
            if rtc_model:
                all_rtc_detail_by_run.extend(zip(*project_run_rtc_detail))

        # Aggregate across all samples: average metrics per run, then average across runs
        aggregate_run_metrics = [
            compute_metrics([preds[run_idx] for preds in all_predictions_by_run], all_references)
            for run_idx in range(num_runs)
        ]
        if rtc_model:
            for run_idx, m in enumerate(aggregate_run_metrics):
                for key in _RTC_METRIC_KEYS:
                    scores = [s[run_idx][key] for s in all_rtc_detail_by_run]
                    m[key] = sum(scores) / len(scores)
        aggregate = {
            k: sum(m[k] for m in aggregate_run_metrics) / num_runs
            for k in aggregate_run_metrics[0]
        }
        print(f"  [aggregate] {aggregate}")
        mlflow.log_metrics(aggregate)
        mlflow.log_param("num_samples", len(all_samples))

        # Per-set metrics (for datasets with a 'set' field)
        set_values = [s.get("set") for s in all_samples]
        known_sets = {v for v in set_values if v is not None}
        for set_name in sorted(known_sets):
            indices = [i for i, v in enumerate(set_values) if v == set_name]
            set_refs = [all_references[i] for i in indices]
            set_run_metrics = [
                compute_metrics([all_predictions_by_run[i][run_idx] for i in indices], set_refs)
                for run_idx in range(num_runs)
            ]
            if rtc_model:
                for run_idx, m in enumerate(set_run_metrics):
                    for key in _RTC_METRIC_KEYS:
                        scores = [all_rtc_detail_by_run[i][run_idx][key] for i in indices]
                        m[key] = sum(scores) / len(scores)
            set_metrics = {
                f"{k}_{set_name}": sum(m[k] for m in set_run_metrics) / num_runs
                for k in set_run_metrics[0]
            }
            print(f"  [set={set_name}] {set_metrics}")
            mlflow.log_metrics(set_metrics)
        _log_predictions_artifact(artifact_rows)
        mlflow.log_metric("cost_usd", cost_tracker.total)


_RTC_METRIC_KEYS = [
    "rtc_bleu", "rtc_crystalbleu", "rtc_codebleu", "codebleu_ngram_match",
    "codebleu_weighted_ngram_match", "codebleu_syntax_match", "codebleu_dataflow_match",
]
_RTC_FIELDS = [*_RTC_METRIC_KEYS, "backward_code", "og_code_compared", "rtc_code_compared"]


def _log_predictions_artifact(rows: list[tuple[dict, int, str, str, dict | None]]) -> None:
    """Write a CSV of (id, project, func_name, run, reference, prediction) -- plus,
    when RTC was computed, rtc_bleu and the exact strings BLEU compared -- and log
    it as an MLflow artifact."""
    has_rtc = any(detail is not None for *_, detail in rows)
    fieldnames = ["id", "project", "func_name", "run", "reference", "prediction"]
    if has_rtc:
        fieldnames += _RTC_FIELDS

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for sample, run_idx, pred, ref, rtc_detail in rows:
        row = {
            "id": sample["id"],
            "project": sample["repo"],
            "func_name": sample["func_name"],
            "run": run_idx,
            "reference": ref,
            "prediction": pred,
        }
        if has_rtc:
            row.update({k: (rtc_detail or {}).get(k, "") for k in _RTC_FIELDS})
        writer.writerow(row)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="predictions")
    Path(tmp_path).unlink()
