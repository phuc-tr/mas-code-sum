"""Experiment runner with MLflow tracking."""

import csv
import io
import tempfile
from pathlib import Path

import os

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
    Each run represents one (method, project) pair, named "{method}/{project}",
    making cross-method comparison easy within the MLflow UI.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading projects for languages: {languages}...")
    projects = load_projects(languages, split=split, max_samples_per_project=max_samples)
    print(f"Found {len(projects)} projects.")

    for project, samples in projects.items():
        project_languages = list({s["language"] for s in samples})
        references = [s["func_documentation_string"] for s in samples]

        with mlflow.start_run(run_name=f"{method.name}/{project}"):
            mlflow.log_params({
                "method": method.name,
                "project": project,
                "language": project_languages[0] if len(project_languages) == 1 else str(project_languages),
                "split": split,
                "num_samples": len(samples),
                "max_samples_per_project": max_samples or "all",
                **method.params(),
            })

            predictions = [method.summarize(s["func_code_string"], s["language"], project=project) for s in samples]

            metrics = compute_metrics(predictions, references)
            print(f"  [{project}] {metrics}")
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
