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
from .metrics import compute_metrics
from .methods.base import BaseSummarizer


def run_experiment(
    method: BaseSummarizer,
    languages: list[str] | None = None,
    max_samples: int | None = None,
    num_runs: int = 1,
    projects: list[str] | None = None,
    dataset: str = "full",
    tags: dict[str, str] | None = None,
    cli_command: str | None = None,
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
    """
    if not languages:
        raise ValueError("'languages' is required.")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading projects for languages: {languages} (dataset {dataset})...")
    projects = load_projects(languages, max_samples_per_project=max_samples, dataset=dataset, projects=projects)
    print(f"Found {len(projects)} projects.")

    artifact_rows: list[tuple[dict, int, str, str]] = []

    all_samples: list[dict] = []
    all_references: list[str] = []
    # per-sample predictions accumulated across runs: index -> list[str]
    all_predictions_by_run: list[list[str]] = []

    _batch_supports_gt = "ground_truths" in inspect.signature(method.summarize_batch).parameters
    _batch_supports_blame = "blame_timestamps" in inspect.signature(method.summarize_batch).parameters
    _batch_supports_blame_sha = "blame_shas" in inspect.signature(method.summarize_batch).parameters

    with mlflow.start_run(run_name=method.name):
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

        if hasattr(method, "validate_projects"):
            method.validate_projects(list(projects.keys()))

        for project, samples in projects.items():
            references = [" ".join(s["docstring_tokens"]) for s in samples]
            codes = [" ".join(s["code_tokens"]) for s in samples]
            langs = [s["language"] for s in samples]
            paths = [s.get("path") for s in samples]
            urls = [s.get("url") for s in samples]
            blame_timestamps = [s.get("latest_blame_timestamp") for s in samples]
            blame_shas = [s.get("blame_sha") for s in samples]

            # Collect predictions across all runs for this project
            project_run_predictions: list[list[str]] = []
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
                preds = method.summarize_batch(**batch_kwargs)
                project_run_predictions.append(preds)
                for sample, pred, ref in zip(samples, preds, references):
                    artifact_rows.append((sample, run_idx, pred, ref))

            # Average metrics across runs
            run_metrics = [
                compute_metrics(preds, references)
                for preds in project_run_predictions
            ]
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

        # Aggregate across all samples: average metrics per run, then average across runs
        aggregate_run_metrics = [
            compute_metrics([preds[run_idx] for preds in all_predictions_by_run], all_references)
            for run_idx in range(num_runs)
        ]
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
            set_metrics = {
                f"{k}_{set_name}": sum(m[k] for m in set_run_metrics) / num_runs
                for k in set_run_metrics[0]
            }
            print(f"  [set={set_name}] {set_metrics}")
            mlflow.log_metrics(set_metrics)
        _log_predictions_artifact(artifact_rows)


def _log_predictions_artifact(rows: list[tuple[dict, int, str, str]]) -> None:
    """Write a CSV of (id, project, func_name, run, reference, prediction) and log as MLflow artifact."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "project", "func_name", "run", "reference", "prediction"])
    writer.writeheader()
    for sample, run_idx, pred, ref in rows:
        writer.writerow({
            "id": sample["id"],
            "project": sample["repo"],
            "func_name": sample["func_name"],
            "run": run_idx,
            "reference": ref,
            "prediction": pred,
        })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="predictions")
    Path(tmp_path).unlink()
