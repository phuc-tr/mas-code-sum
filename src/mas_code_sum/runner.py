"""Experiment runner with MLflow tracking."""

import csv
import io
import tempfile
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("mlruns/")

from .data import load_projects
from .metrics import compute_metrics
from .methods.base import BaseSummarizer


def run_experiment(
    method: BaseSummarizer,
    experiment_name: str,
    languages: list[str],
    split: str = "test",
    max_samples: int | None = None,
) -> None:
    """
    Run a summarization experiment across all projects found in the given languages.

    One MLflow experiment per call (named by experiment_name).
    One MLflow run per project (repository_name), with language logged as a param.
    max_samples caps the number of samples per project.
    """
    mlflow.set_experiment(experiment_name)

    print(f"Loading projects for languages: {languages}...")
    projects = load_projects(languages, split=split, max_samples_per_project=max_samples)
    print(f"Found {len(projects)} projects.")

    for project, samples in projects.items():
        # A project's samples may technically span languages, so log all unique ones.
        project_languages = list({s["language"] for s in samples})

        references = [s["func_documentation_string"] for s in samples]
        predictions = [method.summarize(s["func_code_string"], s["language"]) for s in samples]

        metrics = compute_metrics(predictions, references)
        print(f"  [{project}] {metrics}")

        with mlflow.start_run(run_name=project):
            mlflow.log_params({
                "method": method.name,
                "project": project,
                "language": project_languages[0] if len(project_languages) == 1 else str(project_languages),
                "split": split,
                "num_samples": len(samples),
                "max_samples_per_project": max_samples or "all",
                **method.params(),
            })
            mlflow.log_metrics(metrics)
            _log_predictions_artifact(samples, predictions, references)


def _log_predictions_artifact(samples: list[dict], predictions: list[str], references: list[str]) -> None:
    """Write a CSV of (func_name, reference, prediction) and log as MLflow artifact."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["func_name", "reference", "prediction"])
    writer.writeheader()
    for sample, pred, ref in zip(samples, predictions, references):
        writer.writerow({"func_name": sample["func_name"], "reference": ref, "prediction": pred})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path="predictions")
    Path(tmp_path).unlink()
