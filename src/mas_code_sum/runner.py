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
    split: str = "test",
    max_samples: int | None = None,
) -> None:
    """
    Run a summarization experiment across all projects found in the given languages.

    All runs land in the single "code-summarization" MLflow experiment.
    Each run represents one method invocation, named after the method.
    Per-project metrics are logged with a "{project}/" prefix; aggregate
    metrics (no prefix) summarise across all projects.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading projects for languages: {languages}...")
    projects = load_projects(languages, split=split, max_samples_per_project=max_samples)
    print(f"Found {len(projects)} projects.")

    all_samples: list[dict] = []
    all_predictions: list[str] = []
    all_references: list[str] = []

    with mlflow.start_run(run_name=method.name):
        mlflow.log_params({
            "method": method.name,
            "languages": str(languages),
            "split": split,
            "num_projects": len(projects),
            "max_samples_per_project": max_samples or "all",
            **method.params(),
        })
        mlflow.set_tag("projects", ",".join(sorted(projects)))

        for project, samples in projects.items():
            references = [" ".join(s["docstring_tokens"]) for s in samples]
            predictions = [method.summarize(" ".join(s["code_tokens"]), s["language"], project=project) for s in samples]

            metrics = compute_metrics(predictions, references)
            print(f"  [{project}] {metrics}")
            mlflow.log_metrics({f"{project}/{k}": v for k, v in metrics.items()})

            all_samples.extend(samples)
            all_predictions.extend(predictions)
            all_references.extend(references)

        aggregate = compute_metrics(all_predictions, all_references)
        print(f"  [aggregate] {aggregate}")
        mlflow.log_metrics(aggregate)
        mlflow.log_param("num_samples", len(all_samples))
        _log_predictions_artifact(all_samples, all_predictions, all_references)


def _log_predictions_artifact(samples: list[dict], predictions: list[str], references: list[str]) -> None:
    """Write a CSV of (project, func_name, reference, prediction) and log as MLflow artifact."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["project", "func_name", "reference", "prediction"])
    writer.writeheader()
    for sample, pred, ref in zip(samples, predictions, references):
        writer.writerow({"project": sample["repo"], "func_name": sample["func_name"], "reference": ref, "prediction": pred})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="predictions")
    Path(tmp_path).unlink()
