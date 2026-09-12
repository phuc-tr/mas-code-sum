"""Backfill the `chrf` and `meteor` metrics onto existing MLflow runs.

Both metrics now come from `mas_code_sum.metrics`, so runs scored by the runner
and runs backfilled here get identical numbers. This script exists for runs that
finished before those metrics were wired in:

  chrf   -- chrF (Popovic, 2015), character n-grams up to order 6, beta=2 (the
            standard weighting, as in sacreBLEU's default).
  meteor -- the official CMU METEOR 1.5 jar, run with -norm. Not nltk's
            `meteor_score`, whose untuned METEOR 1.0 parameters (beta=3.0 vs
            1.5's 0.2) score roughly twice as high on this data.

Both are logged on a 0-100 scale, matching `bleu`. Following the runner, each is
logged overall and per dataset `set` (`chrf_new`, `meteor_original`, ...), with
the `set` joined back in from `dataset/full/test.jsonl` by `id`; runs whose
predictions carry more than one `run` value also get `_run{N}` variants, as in
`backfill_rtc_metric.py`.

Because chrF strips whitespace and METEOR's -norm tokenizes punctuation itself,
neither metric is affected by the references' spaced-punctuation style ("sub -
agents ." vs "sub-agents.") that few-shot methods mimic and others do not.

METEOR 1.5 needs Java and the jar, which is too large to vendor; see
`mas_code_sum.metrics` for where it is expected and how to fetch it.

Usage:
    python scripts/backfill_chrf_meteor.py RUN_ID [RUN_ID ...]
    python scripts/backfill_chrf_meteor.py RUN_ID --dry-run     # print, log nothing
"""

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from mas_code_sum.data import load_samples  # noqa: E402
from mas_code_sum.metrics import CHRF_BETA, chrf, meteor, meteor_jar  # noqa: E402

# `dataset/small` is a strict id-subset of `dataset/full`, so the full split
# resolves the `set` of every row regardless of which variant a run used.
SET_INDEX_LANGUAGES = ["python", "java"]

METEOR_IMPL = "meteor-1.5"


def _load_set_index() -> dict[int, str]:
    """id -> dataset set name ('new' / 'original'), for the per-set metrics."""
    index: dict[int, str] = {}
    for language in SET_INDEX_LANGUAGES:
        for sample in load_samples(language, "test", "full"):
            if sample.get("set") is not None:
                index[sample["id"]] = sample["set"]
    return index


def backfill_run(
    client: MlflowClient,
    run_id: str,
    set_index: dict[int, str],
    dry_run: bool,
) -> None:
    print(f"[{run_id}] downloading predictions artifact...")
    local_dir = client.download_artifacts(run_id, "predictions")
    csvs = list(pathlib.Path(local_dir).glob("*.csv"))
    assert len(csvs) == 1, f"expected exactly one predictions csv, found {csvs}"
    df = pd.read_csv(csvs[0])
    df["set"] = df["id"].map(set_index.get)

    # A model that returned nothing reads back as NaN. That is a failed summary,
    # not a missing sample: score it as the empty string (0 on both metrics) so
    # the mean still covers every row the run was asked to produce.
    for field in ["reference", "prediction"]:
        blank = int(df[field].isna().sum())
        if blank:
            print(f"[{run_id}] {blank}/{len(df)} rows have an empty `{field}` (scored as 0)")
        df[field] = df[field].fillna("")

    print(f"[{run_id}] scoring {len(df)} rows (chrF beta={CHRF_BETA:g}, {METEOR_IMPL})...")
    df["chrf"] = [
        chrf(ref, pred) for ref, pred in zip(df["reference"], df["prediction"])
    ]
    df["meteor"] = meteor(df["reference"].tolist(), df["prediction"].tolist())

    metrics = {key: float(df[key].mean()) for key in ["chrf", "meteor"]}

    # Per-set, matching the runner's `{metric}_{set}` keys.
    missing = int(df["set"].isna().sum())
    if missing:
        print(f"[{run_id}] {missing}/{len(df)} rows have no `set` in dataset/full/test.jsonl")
    for set_name, sub in df.dropna(subset=["set"]).groupby("set"):
        for key in ["chrf", "meteor"]:
            metrics[f"{key}_{set_name}"] = float(sub[key].mean())

    # Per summarize_batch run, matching backfill_rtc_metric.py.
    run_values = sorted(df["run"].unique())
    if len(run_values) > 1:
        for run_value, sub in df.groupby("run"):
            for key in ["chrf", "meteor"]:
                metrics[f"{key}_run{run_value}"] = float(sub[key].mean())

    print(f"[{run_id}] metrics: { {k: round(v, 4) for k, v in metrics.items()} }")
    if dry_run:
        print(f"[{run_id}] --dry-run: nothing logged")
        return

    now_ms = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[
            mlflow.entities.Metric(key=k, value=v, timestamp=now_ms, step=0)
            for k, v in metrics.items()
        ],
    )
    client.log_param(run_id, "meteor_impl", METEOR_IMPL)
    client.log_param(run_id, "chrf_beta", CHRF_BETA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="MLflow run IDs to backfill")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, log nothing")
    args = parser.parse_args()

    # `mas_code_sum.metrics` skips METEOR when the jar is missing, which is right
    # for a scoring run but not here: a backfill that silently logged only chrF
    # would look like it had done its job.
    if meteor_jar() is None:
        parser.error("METEOR 1.5 jar not found -- set METEOR_JAR (see mas_code_sum.metrics)")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    set_index = _load_set_index()

    for run_id in args.run_ids:
        backfill_run(client, run_id, set_index, args.dry_run)


if __name__ == "__main__":
    main()
