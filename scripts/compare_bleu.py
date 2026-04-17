"""Pairwise metric comparison across all finished MLflow runs.

Usage:
    uv run python scripts/compare_bleu.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.mas_code_sum.evaluator import bleu as _bleu

MLFLOW_URI = "http://127.0.0.1:5000"
ALPHA = 0.05
METRICS = ["bleu"]

console = Console(width=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_label(row: pd.Series) -> str:
    return str(row.get("params.method") or row.get("tags.mlflow.runName") or "unknown")


def load_predictions(client: mlflow.tracking.MlflowClient, run_id: str) -> pd.DataFrame:
    artifacts = client.list_artifacts(run_id, path="predictions")
    if not artifacts:
        return pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifacts[0].path, tmp)
        return pd.read_csv(local)


def per_sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample BLEU and ROUGE-L using run 0 only."""
    if "run" in df.columns:
        df = df[df["run"] == 0].copy()
    else:
        df = df.copy()

    # Drop rows with missing predictions or references (failed/empty generations)
    df = df.dropna(subset=["prediction", "reference"])
    df = df[df["prediction"].astype(str).str.strip() != ""]

    df["bleu"] = [
        _bleu([ref], pred)[0] * 100
        for pred, ref in zip(df["prediction"], df["reference"])
    ]

    # Always use (project, func_name) so indices are comparable across runs
    # regardless of whether an "id" column exists (schema changed over time).
    return df.groupby(["project", "func_name"])[METRICS].mean()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_URI)
runs = mlflow.search_runs(experiment_names=["code-summarization"], output_format="pandas")
runs = runs[runs["status"] == "FINISHED"]
if "tags.version" in runs.columns:
    runs = runs[runs["tags.version"] == "v2"]
runs = runs.reset_index(drop=True)
console.print(f"[dim]{len(runs)} finished runs (version=v2)[/dim]\n")

client = mlflow.tracking.MlflowClient()
scores: dict[str, pd.DataFrame] = {}  # run_id -> DataFrame(index=sample_id, cols=metrics)
labels: dict[str, str] = {}

for _, row in runs.iterrows():
    run_id = row["run_id"]
    label = run_label(row)
    preds = load_predictions(client, run_id)
    if preds.empty:
        console.print(f"[yellow]  skip {label!r} — no predictions artifact[/yellow]")
        continue
    scores[run_id] = per_sample_metrics(preds)
    labels[run_id] = label
    n = len(scores[run_id])
    console.print(f"  [dim]loaded[/dim] {label!r}  [dim]({n} samples)[/dim]")

run_ids = list(scores.keys())

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

means = {rid: scores[rid].mean() for rid in run_ids}
ranked = sorted(run_ids, key=lambda r: means[r]["bleu"], reverse=True)
rank_idx = {rid: i + 1 for i, rid in enumerate(ranked)}

console.print()
summary = Table(title="Summary", box=box.SIMPLE_HEAD)
summary.add_column("#", style="dim", justify="right")
summary.add_column("Method")
for m in METRICS:
    summary.add_column(m, justify="right")
summary.add_column("N", justify="right", style="dim")

for rid in ranked:
    summary.add_row(
        str(rank_idx[rid]),
        labels[rid],
        *[f"{means[rid][m]:.3f}" for m in METRICS],
        str(len(scores[rid])),
    )

console.print(summary)

# ---------------------------------------------------------------------------
# Per-project comparison vs CodeT5 baseline
# ---------------------------------------------------------------------------

baseline_id = next(
    (
        rid for rid in run_ids
        if str(runs.loc[runs["run_id"] == rid, "params.method"].iloc[0]).lower() == "codet5"
    ),
    None,
)
if baseline_id is None:
    console.print("\n[red]No CodeT5 baseline run found (params.method=codet5).[/red]")
    sys.exit(0)

baseline_label = labels[baseline_id]
baseline_scores = scores[baseline_id]
non_baseline_ids = [rid for rid in ranked if rid != baseline_id]

# One shared p-value per (non-baseline method, metric), computed across all
# paired samples from every project combined.
p_values: dict[str, dict[str, float]] = {}
for rid in non_baseline_ids:
    shared = baseline_scores.index.intersection(scores[rid].index)
    pv: dict[str, float] = {}
    if len(shared) < 2:
        pv = {m: float("nan") for m in METRICS}
    else:
        for m in METRICS:
            x = scores[rid].loc[shared, m].values
            y = baseline_scores.loc[shared, m].values
            d = x.mean() - y.mean()
            try:
                _, p = stats.wilcoxon(x, y, alternative="greater" if d >= 0 else "less")
            except ValueError:
                p = 1.0
            pv[m] = p
    p_values[rid] = pv

ordered = [baseline_id, *non_baseline_ids]

projects = sorted({
    proj
    for rid in ordered
    for proj in scores[rid].index.get_level_values(0).unique()
})


def project_mean(df: pd.DataFrame, project: str, metric: str) -> float | None:
    try:
        sub = df.xs(project, level=0)
    except KeyError:
        return None
    return None if sub.empty else float(sub[metric].mean())


def metric_col(rid: str, metric: str) -> str:
    return labels[rid] if len(METRICS) == 1 else f"{labels[rid]} {metric}"


def pvalue_col(rid: str, metric: str) -> str:
    return f"{labels[rid]} p" if len(METRICS) == 1 else f"{labels[rid]} p({metric})"


records: list[dict[str, object]] = []
for proj in projects:
    record: dict[str, object] = {"Project": proj}
    for rid in ordered:
        for m in METRICS:
            record[metric_col(rid, m)] = project_mean(scores[rid], proj, m)
            if rid != baseline_id:
                record[pvalue_col(rid, m)] = p_values[rid][m]
    records.append(record)

overall_record: dict[str, object] = {"Project": "Overall"}
for rid in ordered:
    for m in METRICS:
        overall_record[metric_col(rid, m)] = float(scores[rid][m].mean())
        if rid != baseline_id:
            overall_record[pvalue_col(rid, m)] = p_values[rid][m]
records.append(overall_record)


def best_rid_for(record: dict[str, object], metric: str) -> str | None:
    candidates = [
        (rid, record[metric_col(rid, metric)])
        for rid in ordered
        if record[metric_col(rid, metric)] is not None
    ]
    return max(candidates, key=lambda x: x[1])[0] if candidates else None


def format_records(bold_open: str, bold_close: str) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for record in records:
        best = {m: best_rid_for(record, m) for m in METRICS}
        row: dict[str, str] = {"Project": str(record["Project"])}
        for rid in ordered:
            for m in METRICS:
                v = record[metric_col(rid, m)]
                if v is None:
                    row[metric_col(rid, m)] = "—"
                else:
                    s = f"{v:.4f}"
                    if rid == best[m]:
                        s = f"{bold_open}{s}{bold_close}"
                    row[metric_col(rid, m)] = s
                if rid != baseline_id:
                    p = record[pvalue_col(rid, m)]
                    row[pvalue_col(rid, m)] = "—" if pd.isna(p) else f"{p:.4f}"
        formatted.append(row)
    return formatted


df_out = pd.DataFrame(records)
csv_path = Path(__file__).parents[1] / "compare_bleu_per_project.csv"
md_path = csv_path.with_suffix(".md")
html_path = csv_path.with_suffix(".html")
df_out.to_csv(csv_path, index=False, float_format="%.4f")

md_df = pd.DataFrame(format_records("**", "**"))
md_path.write_text(md_df.to_markdown(index=False))

html_df = pd.DataFrame(format_records("<strong>", "</strong>"))
html_df.to_html(html_path, index=False, border=1, escape=False)


console.print(f"\n[bold]Per-project metrics — baseline: {baseline_label}[/bold]")

table = Table(box=box.SIMPLE_HEAD)
table.add_column("Project", no_wrap=True)
for rid in ordered:
    tag = f"#{rank_idx[rid]}" + (" (base)" if rid == baseline_id else "")
    table.add_column(f"{tag}\nBLEU", justify="right")
    if rid != baseline_id:
        table.add_column(f"#{rank_idx[rid]}\np(BLEU)", justify="right")

for i, record in enumerate(records):
    is_overall = i == len(records) - 1
    best = {m: best_rid_for(record, m) for m in METRICS}
    project_label = "[bold]Overall[/bold]" if is_overall else str(record["Project"])
    cells: list[str] = [project_label]
    for rid in ordered:
        for m in METRICS:
            v = record[metric_col(rid, m)]
            if v is None:
                cells.append("[dim]—[/dim]")
            else:
                s = f"{v:.3f}"
                if rid == best[m]:
                    s = f"[bold]{s}[/bold]"
                cells.append(s)
            if rid != baseline_id:
                p = record[pvalue_col(rid, m)]
                if pd.isna(p):
                    cells.append("[dim]—[/dim]")
                else:
                    star = "*" if p < ALPHA else ""
                    cells.append(f"{p:.4f}{star}")
    table.add_row(*cells)

console.print(table)
console.print(
    f"[dim]p-values: one-sided Wilcoxon signed-rank vs {baseline_label} "
    f"on all paired samples across projects (single test per method/metric, "
    f"not per project). * p < {ALPHA} (no correction).[/dim]"
)
console.print(f"[dim]exported:[/dim] {csv_path}")
console.print(f"[dim]exported:[/dim] {md_path}")
console.print(f"[dim]exported:[/dim] {html_path}")
