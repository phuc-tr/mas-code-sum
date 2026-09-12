"""Compare chunk retention under the two temporal-filter policies.

One blame pass per repo (cached under dataset/blame_cache/), then both
policies scored from the cached dates.
"""
import json
import sys

sys.path.insert(0, "src")
from mas_code_sum.enrichers.ast_chunks import get_chunk_index  # noqa: E402
from mas_code_sum.enrichers.blame_cutoff import (  # noqa: E402
    get_chunk_dates,
    get_repo_cutoffs,
)

langs = {}
for line in open("dataset/full/test.jsonl"):
    row = json.loads(line)
    langs[row["repo"]] = row["language"]

cutoffs = get_repo_cutoffs()
print(f"{'repo':44s} {'total':>7s} {'start':>7s} {'touched':>8s} {'gain':>7s}")
tot = start_tot = touched_tot = 0
for repo, lang in sorted(langs.items()):
    chunks = get_chunk_index(repo, lang, temporal_filter=False).chunks
    dates = get_chunk_dates(repo, lang, chunks, 1200)
    cutoff = cutoffs[repo]
    vals = [dates[k] for c in chunks
            if (k := (c.filepath, c.start_line, c.end_line)) in dates]
    start = sum(1 for first, _ in vals if first <= cutoff)
    touched = sum(1 for _, last in vals if last <= cutoff)
    tot += len(chunks); start_tot += start; touched_tot += touched
    print(f"{repo:44s} {len(chunks):7d} {start:7d} {touched:8d} "
          f"{start / touched if touched else float('inf'):6.1f}x", flush=True)
print(f"{'TOTAL':44s} {tot:7d} {start_tot:7d} {touched_tot:8d} "
      f"{start_tot / touched_tot:6.1f}x")
