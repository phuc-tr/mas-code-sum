"""Experiment runner with MLflow tracking."""

import csv
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
    languages: list[str],
    max_samples: int | None = None,
    num_runs: int = 1,
) -> None:
    """
    Run a summarization experiment across all projects found in the given languages.

    All runs land in the single "code-summarization" MLflow experiment.
    Each run represents one method invocation, named after the method.
    Per-project metrics are logged with a "{project}/" prefix; aggregate
    metrics (no prefix) summarise across all projects.

    Each sample is summarized `num_runs` times and metrics are averaged across
    runs to reduce variance from stochastic generation.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading projects for languages: {languages}...")
    projects = load_projects(languages, max_samples_per_project=max_samples)
    print(f"Found {len(projects)} projects.")

    # artifact rows: (sample, run_idx, prediction, reference)
    artifact_rows: list[tuple[dict, int, str, str]] = []

    all_samples: list[dict] = []
    all_references: list[str] = []
    # per-sample predictions accumulated across runs: index -> list[str]
    all_predictions_by_run: list[list[str]] = []

    with mlflow.start_run(run_name=method.name):
        mlflow.log_params({
            "method": method.name,
            "languages": str(languages),
            "max_samples_per_project": max_samples or "all",
            "num_runs": num_runs,
            **method.params(),
        })
        mlflow.set_tag("method", method.name)

        for project, samples in projects.items():
            references = [" ".join(s["docstring_tokens"]) for s in samples]
            codes = [" ".join(s["code_tokens"]) for s in samples]
            langs = [s["language"] for s in samples]
            paths = [s.get("path") for s in samples]
            urls = [s.get("url") for s in samples]

            # Collect predictions across all runs for this project
            project_run_predictions: list[list[str]] = []
            for run_idx in range(num_runs):
                preds = method.summarize_batch(
                    codes=codes,
                    languages=langs,
                    projects=[project] * len(samples),
                    paths=paths,
                    urls=urls,
                )
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
            mlflow.log_metrics({f"{project}/{k}": v for k, v in avg_metrics.items()})

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
        _log_predictions_artifact(artifact_rows)


def _log_predictions_artifact(rows: list[tuple[dict, int, str, str]]) -> None:
    """Write a CSV of (id, project, func_name, run, reference, prediction) and log as MLflow artifact."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "project", "func_name", "run", "reference", "prediction"])
    writer.writeheader()
    for sample, run_idx, pred, ref in rows:
        writer.writerow({"id": sample["id"], "project": sample["repo"], "func_name": sample["func_name"], "run": run_idx, "reference": ref, "prediction": pred})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="predictions")
    Path(tmp_path).unlink()
