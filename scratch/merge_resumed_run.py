"""Merge a resumed run's per-project metrics with those of the interrupted run.

Both runs log `per_project_metrics.json` (one row per project). Every metric the
runner computes is a per-sample mean and every project holds the same number of
samples, so the aggregate and per-set numbers are plain unweighted means over the
per-project rows -- exactly what the runner would have produced in one pass.

Usage:
    python scratch/merge_resumed_run.py <failed_run_id> <resume_run_id>
"""

import json
import sys
from pathlib import Path

from src.mas_code_sum.data import load_projects

ARTIFACT_ROOT = Path("mlflow-data/mlartifacts/1")


def load_rows(run_id: str) -> dict[str, dict[str, float]]:
    path = ARTIFACT_ROOT / run_id / "artifacts" / "per_project_metrics.json"
    table = json.loads(path.read_text())
    cols = table["columns"]
    return {row[0]: dict(zip(cols[1:], row[1:])) for row in table["data"]}


def mean_over(rows: dict[str, dict[str, float]], projects: list[str]) -> dict[str, float]:
    keys = next(iter(rows.values())).keys()
    return {k: sum(rows[p][k] for p in projects) / len(projects) for k in keys}


def main() -> None:
    failed_id, resume_id = sys.argv[1], sys.argv[2]

    rows = {**load_rows(failed_id), **load_rows(resume_id)}

    projects = load_projects(["python", "java"], dataset="full")
    missing = set(projects) - set(rows)
    if missing:
        raise SystemExit(f"Still missing per-project metrics for: {sorted(missing)}")

    sizes = {len(s) for s in projects.values()}
    if len(sizes) != 1:
        raise SystemExit(
            f"Projects have differing sample counts {sorted(sizes)}; an unweighted "
            "mean over per-project rows would not equal the pooled aggregate."
        )

    sets: dict[str, list[str]] = {}
    for name, samples in projects.items():
        values = {s.get("set") for s in samples}
        if len(values) != 1:
            raise SystemExit(f"Project {name} spans multiple sets {values}; cannot split.")
        sets.setdefault(values.pop(), []).append(name)

    all_projects = list(projects)
    aggregate = mean_over(rows, all_projects)

    print(f"[aggregate] (n={len(all_projects)} projects)")
    for k, v in aggregate.items():
        print(f"  {k:32s} {v:.4f}")

    for set_name in sorted(s for s in sets if s is not None):
        print(f"\n[set={set_name}] (n={len(sets[set_name])} projects)")
        for k, v in mean_over(rows, sets[set_name]).items():
            print(f"  {k}_{set_name:32s} {v:.4f}")

    out = Path("scratch/merged_metrics.json")
    out.write_text(json.dumps(
        {
            "source_runs": {"interrupted": failed_id, "resumed": resume_id},
            "per_project": rows,
            "aggregate": aggregate,
            "per_set": {s: mean_over(rows, p) for s, p in sets.items() if s is not None},
        },
        indent=2,
    ))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
