"""Experiment runner with MLflow tracking."""

import csv
import io
import tempfile
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("mlruns/")

from .data import load_samples
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
    Run a summarization experiment across the given languages.

    One MLflow experiment per call (named by experiment_name).
    One MLflow run per language, logging params + metrics + predictions artifact.

    Args:
        method: a BaseSummarizer instance
        experiment_name: MLflow experiment name (from the YAML config)
        languages: list of language names to evaluate
        split: dataset split to use
        max_samples: cap samples per language (useful for quick iterations)
    """
    mlflow.set_experiment(experiment_name)

    for language in languages:
        print(f"  [{language}] loading samples...")
        samples = load_samples(language, split=split, max_samples=max_samples)

        references = [s["func_documentation_string"] for s in samples]
        predictions = [method.summarize(s["func_code_string"], language) for s in samples]

        metrics = compute_metrics(predictions, references)
        print(f"  [{language}] {metrics}")

        with mlflow.start_run(run_name=language):
            mlflow.log_params({
                "method": method.name,
                "language": language,
                "split": split,
                "num_samples": len(samples),
                "max_samples": max_samples or "all",
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
