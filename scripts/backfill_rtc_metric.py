"""Backfill the `rtc_bleu` metric onto existing MLflow runs.

For each run, downloads its `predictions/*.csv` artifact, joins the original
`code`/`language`/`repo`/`path`/`blame_sha` back in from `dataset/full/test.jsonl`
(by `id` -- `dataset/small` is a strict id-subset of `dataset/full`, so this
works regardless of which dataset variant the run itself used), computes
round-trip correctness via `mas_code_sum.rtc`, and logs the resulting
`rtc_bleu` metric (overall, and `rtc_bleu_{run}` per summarize_batch run if
the predictions artifact has more than one `run` value) onto the run.

Usage:
    python scripts/backfill_rtc_metric.py RUN_ID [RUN_ID ...] --model openai/gpt-4o-mini
    python scripts/backfill_rtc_metric.py RUN_ID --model openai/gpt-4o-mini --backend openrouter
"""

import argparse
import json
import pathlib
import time

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

from mas_code_sum.data import dataset_dir
from mas_code_sum.rtc import DEFAULT_BACKWARD_BACKEND, compute_rtc_scores_sync

load_dotenv()


def _load_code_index() -> dict[int, dict]:
    index: dict[int, dict] = {}
    with open(dataset_dir() / "full" / "test.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            index[sample["id"]] = {
                "code": sample["code"],
                "language": sample["language"],
                "repo": sample["repo"],
                "path": sample["path"],
                "blame_sha": sample.get("blame_sha"),
            }
    return index


def backfill_run(client: MlflowClient, run_id: str, model: str, backend: str, code_index: dict[int, dict]) -> None:
    print(f"[{run_id}] downloading predictions artifact...")
    local_dir = client.download_artifacts(run_id, "predictions")
    csvs = list(pathlib.Path(local_dir).glob("*.csv"))
    assert len(csvs) == 1, f"expected exactly one predictions csv, found {csvs}"
    df = pd.read_csv(csvs[0])

    for field in ["code", "language", "repo", "path", "blame_sha"]:
        df[field] = df["id"].map(lambda i, field=field: code_index.get(i, {}).get(field))
    missing = df["code"].isna().sum()
    if missing:
        print(f"[{run_id}] {missing}/{len(df)} rows with no matching code in dataset/full/test.jsonl (dropping)")
    df = df.dropna(subset=["code"]).reset_index(drop=True)

    metric_keys = ["rtc_bleu", "rtc_codebleu"]
    run_values = sorted(df["run"].unique())
    per_run_means: dict[int, dict[str, float]] = {}
    all_detail: list[dict] = []
    for run_value in run_values:
        sub = df[df["run"] == run_value].reset_index(drop=True)
        samples = sub[["code", "language", "repo", "path", "blame_sha"]].to_dict("records")
        predictions = sub["prediction"].tolist()
        print(f"[{run_id}] run={run_value}: computing rtc_bleu/rtc_codebleu for {len(samples)} samples with backward model {model}...")
        detail = compute_rtc_scores_sync(samples, predictions, model, backend)
        per_run_means[run_value] = {key: sum(d[key] for d in detail) / len(detail) for key in metric_keys}
        all_detail.extend(detail)

    metrics = {key: sum(d[key] for d in all_detail) / len(all_detail) for key in metric_keys}
    if len(run_values) > 1:
        for run_value, means in per_run_means.items():
            metrics.update({f"{key}_run{run_value}": mean for key, mean in means.items()})

    print(f"[{run_id}] logging metrics: {metrics}")
    now_ms = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[mlflow.entities.Metric(key=k, value=v, timestamp=now_ms, step=0) for k, v in metrics.items()],
    )
    client.log_param(run_id, "rtc_model", model)
    client.log_param(run_id, "rtc_backend", backend)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="MLflow run IDs to backfill")
    parser.add_argument("--model", required=True, help="Backward model to use for the RTC round-trip")
    parser.add_argument("--backend", default=DEFAULT_BACKWARD_BACKEND, help="Backend for the backward model")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    code_index = _load_code_index()

    for run_id in args.run_ids:
        backfill_run(client, run_id, args.model, args.backend, code_index)


if __name__ == "__main__":
    main()
