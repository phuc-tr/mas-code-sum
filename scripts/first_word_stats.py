"""Show summary first-word statistics grouped by project (repo)."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOP_N = 30  # how many first words to show as rows

# Set to a subset to filter; leave empty to include all projects
PROJECTS = [
    # "PyCQA/pylint",
    # "Qiskit/qiskit-terra",
    "apache/airflow",
    # "h2oai/h2o-3",
    # "oblac/jodd",
    # "orientechnologies/orientdb",
    # "real-logic/aeron",
    # "spring-projects/spring-security",
    # "vaexio/vaex",
    # "wildfly/wildfly",
]

dataset_root = Path(__file__).parent.parent / "dataset"
files = [
    dataset_root / "python" / "train.jsonl",
    dataset_root / "python" / "test.jsonl",
    dataset_root / "java" / "train.jsonl",
    dataset_root / "java" / "test.jsonl",
]

# repo -> first_word -> count
stats: dict[str, Counter] = defaultdict(Counter)
global_word_counts: Counter = Counter()

for path in files:
    if not path.exists():
        print(f"Skipping missing file: {path}", file=sys.stderr)
        continue
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tokens = record.get("docstring_tokens")
        if not tokens:
            raise ValueError(f"Record {record.get('id')} has no docstring_tokens")
        first_word = tokens[0].lower()
        repo = record.get("repo", "<unknown>")
        if PROJECTS and repo not in PROJECTS:
            continue
        stats[repo][first_word] += 1
        global_word_counts[first_word] += 1

# Top N first words globally become rows; projects become columns
top_words = [w for w, _ in global_word_counts.most_common(TOP_N)]
repos = PROJECTS if PROJECTS else sorted(stats)
repo_totals = {repo: sum(stats[repo].values()) for repo in repos}
grand = sum(repo_totals.values())

row_w = max(len(w) for w in top_words + ["other", "TOTAL"]) + 2
cell_w = 14


def fmt_cell(n: int, total: int) -> str:
    pct = n / total * 100 if total else 0
    return f"{n} ({pct:.1f}%)".rjust(cell_w)


header = f"{'First word':<{row_w}}" + "".join(f"{r:>{cell_w}}" for r in repos) + f"{'TOTAL':>{cell_w}}"
print(header)
print("-" * len(header))

for word in top_words:
    word_total = global_word_counts[word]
    row = (
        f"{word:<{row_w}}"
        + "".join(fmt_cell(stats[repo].get(word, 0), repo_totals[repo]) for repo in repos)
        + fmt_cell(word_total, grand)
    )
    print(row)

# "other" row
other_per_repo = {repo: repo_totals[repo] - sum(stats[repo].get(w, 0) for w in top_words) for repo in repos}
other_total = grand - sum(global_word_counts[w] for w in top_words)
print(
    f"{'other':<{row_w}}"
    + "".join(fmt_cell(other_per_repo[repo], repo_totals[repo]) for repo in repos)
    + fmt_cell(other_total, grand)
)

print("-" * len(header))
print(
    f"{'TOTAL':<{row_w}}"
    + "".join(f"{repo_totals[repo]:>{cell_w}}" for repo in repos)
    + f"{grand:>{cell_w}}"
)
