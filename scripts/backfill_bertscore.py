"""Backfill BERTScore (roberta-large) F1 into existing MLflow runs.

For each run in the "code-summarization" experiment:
  1. Downloads the predictions CSV artifact.
  2. Computes per-project and aggregate BERTScore F1.
  3. Logs the new metrics back to the same run.

Skips runs that already have the aggregate `bertscore_f1` metric logged.
"""

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import pandas as pd
from bert_score import score as bert_score
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

EXPERIMENT_NAME = "code-summarization"
BERT_MODEL = "roberta-large"


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, F1 = bert_score(predictions, references, model_type=BERT_MODEL, verbose=False)
    return F1.mean().item()


def backfill_run(client: mlflow.MlflowClient, run: mlflow.entities.Run) -> None:
    run_id = run.info.run_id
    run_name = run.data.tags.get("mlflow.runName", run_id)

    # Skip if already computed
    if "bertscore_f1" in run.data.metrics:
        print(f"  [{run_name}] already has bertscore_f1, skipping")
        return

    # Find the predictions artifact
    artifacts = client.list_artifacts(run_id, "predictions")
    csv_artifacts = [a for a in artifacts if a.path.endswith(".csv")]
    if not csv_artifacts:
        print(f"  [{run_name}] no predictions CSV found, skipping")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, csv_artifacts[0].path, tmp_dir)
        df = pd.read_csv(local_path)

    required = {"project", "reference", "prediction"}
    if not required.issubset(df.columns):
        print(f"  [{run_name}] CSV missing columns {required - set(df.columns)}, skipping")
        return

    # Average predictions across runs (if num_runs > 1, take first run only for scoring)
    # Use run=0 if the column exists, otherwise use all rows
    if "run" in df.columns:
        df = df[df["run"] == 0]

    new_metrics: dict[str, float] = {}

    # Per-project
    for project, group in df.groupby("project"):
        preds = group["prediction"].tolist()
        refs = group["reference"].tolist()
        score = compute_bertscore(preds, refs)
        new_metrics[f"{project}/bertscore_f1"] = score

    # Aggregate
    new_metrics["bertscore_f1"] = compute_bertscore(
        df["prediction"].tolist(), df["reference"].tolist()
    )

    client.log_batch(
        run_id,
        metrics=[mlflow.entities.Metric(k, v, 0, 0) for k, v in new_metrics.items()],
    )
    print(f"  [{run_name}] logged bertscore_f1={new_metrics['bertscore_f1']:.4f} "
          f"({len(new_metrics) - 1} projects)")


def main() -> None:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"Found {len(runs)} runs in '{EXPERIMENT_NAME}'")

    for run in runs:
        backfill_run(client, run)

    print("Done.")


if __name__ == "__main__":
    main()
