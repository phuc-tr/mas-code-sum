"""Show intention classification statistics grouped by project (repo)."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID_INTENTIONS = ["What", "Why", "How-to-use-it", "How-it-is-done", "Property"]

dataset_root = Path(__file__).parent.parent / "dataset"
files = [
    dataset_root / "python" / "train.jsonl",
    dataset_root / "python" / "test.jsonl",
    dataset_root / "java" / "train.jsonl",
    dataset_root / "java" / "test.jsonl",
]

# repo -> intention -> count
stats: dict[str, Counter] = defaultdict(Counter)
missing_intention = 0

for path in files:
    if not path.exists():
        print(f"Skipping missing file: {path}", file=sys.stderr)
        continue
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "intention" not in record:
            missing_intention += 1
            continue
        repo = record.get("repo", "<unknown>")
        stats[repo][record["intention"]] += 1

if missing_intention:
    print(f"Warning: {missing_intention} records missing 'intention' field\n", file=sys.stderr)

col_w = max(len(r) for r in stats) + 2
cell_w = 14  # "  123 (45.6%)"


def fmt_cell(n: int, total: int) -> str:
    pct = n / total * 100 if total else 0
    return f"{n} ({pct:.1f}%)".rjust(cell_w)


header = f"{'Project':<{col_w}}" + "".join(f"{h:>{cell_w}}" for h in VALID_INTENTIONS) + f"{'TOTAL':>{cell_w}}"
print(header)
print("-" * len(header))

totals = Counter()
for repo in sorted(stats):
    counts = stats[repo]
    total = sum(counts.values())
    row = f"{repo:<{col_w}}" + "".join(fmt_cell(counts.get(h, 0), total) for h in VALID_INTENTIONS) + f"{total:>{cell_w}}"
    print(row)
    totals += counts

print("-" * len(header))
grand = sum(totals.values())
print(f"{'TOTAL':<{col_w}}" + "".join(fmt_cell(totals.get(h, 0), grand) for h in VALID_INTENTIONS) + f"{grand:>{cell_w}}")
