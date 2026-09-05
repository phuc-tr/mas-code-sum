"""Temporal (anti-leakage) filter for repository chunks.

Repo checkouts sit at HEAD, but every test sample was authored at some point
in the past. Code written *after* a sample was authored could not have
informed its docstring, so feeding it to the summarizer as "repository
context" leaks future information into the evaluation.

Cutoff
------
Two levels, so that one shared index can still be filtered per sample.

*Index cutoff* (`get_repo_index_cutoffs`) is the **latest** `authored_timestamp`
among a repo's test samples: the loosest date any of its samples could legally
use. The index is built once per repo at this bound, so a single BM25 fit is
shared by every sample of that repo.

*Sample cutoff* is the individual sample's own `authored_timestamp`, applied at
query time against the per-chunk dates carried on the index (see
`ChunkIndex.query`). This is what makes the filter per-sample without paying
for a per-sample index.

`get_repo_cutoffs` (the **earliest** sample date) remains the fallback for
callers with no sample timestamp to hand. It is what the whole corpus used
before per-sample filtering existed, and it is needlessly strict: repos here
span up to 4.8 years between their first and last sample, so a late sample was
being denied years of legitimate context.

Cutoffs always come from `dataset/full/test.jsonl`, the superset split, so they
do not shift when an experiment runs on `small`/`tiny`.

Policy — how a chunk is dated
-----------------------------
`start` (default, looser)
    Date the chunk by the blame time of its *first* line: when the block was
    introduced. A 2016 function that received a one-line fix in 2024 stays.
`last_touched` (stricter)
    Date the chunk by its *newest* line. Any post-cutoff line discards the
    whole chunk, including its older lines.

Neither policy is leak-free on its own: `start` can admit a post-cutoff line
that rode into an old block, while `last_touched` discards large amounts of
legitimately old context to prevent that. `start` is the default because the
retained context is far larger and the leaked fraction is small.

Caching
-------
Blame is expensive (a subprocess per file, tens of thousands of files across
the corpus), so a single pass records *both* dates for every chunk under
`dataset/blame_cache/`, keyed by repo, language, and chunk size. Cutoff and
policy are applied to the cached dates, so changing either is free.
"""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .file_context import REPOS_ROOT, _repo_dir

if TYPE_CHECKING:
    from .ast_chunks import Chunk

_log = logging.getLogger(__name__)

_DATASET_ROOT = REPOS_ROOT.parent
_TEST_SET = _DATASET_ROOT / "full" / "test.jsonl"
_CACHE_DIR = _DATASET_ROOT / "blame_cache"
_BLAME_WORKERS = 16
_BLAME_TIMEOUT = 300

POLICIES = ("start", "last_touched")

# (filepath, start_line, end_line) -> (first-line time, newest-line time)
ChunkDates = dict[tuple[str, int, int], tuple[int, int]]


def parse_timestamp(stamp: str | None) -> int | None:
    """Dataset `authored_timestamp` -> unix time. None for missing/malformed."""
    if not stamp:
        return None
    try:
        return int(
            datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except ValueError:
        _log.warning("unparseable authored_timestamp %r", stamp)
        return None


@lru_cache(maxsize=1)
def _repo_sample_times() -> dict[str, list[int]]:
    """repo -> unix times of all its test samples."""
    times: dict[str, list[int]] = {}
    if not _TEST_SET.is_file():
        _log.warning("no test set at %s; temporal filter inactive", _TEST_SET)
        return times
    with _TEST_SET.open() as fh:
        for line in fh:
            row = json.loads(line)
            repo, ts = row.get("repo"), parse_timestamp(row.get("authored_timestamp"))
            if repo and ts is not None:
                times.setdefault(repo, []).append(ts)
    return times


def get_repo_cutoffs() -> dict[str, int]:
    """repo -> unix time of its *earliest* test sample (strictest fallback cutoff)."""
    return {repo: min(ts) for repo, ts in _repo_sample_times().items()}


def get_repo_index_cutoffs() -> dict[str, int]:
    """repo -> unix time of its *latest* test sample (the index build bound)."""
    return {repo: max(ts) for repo, ts in _repo_sample_times().items()}


def _blame_line_times(repo_dir: Path, path: str) -> dict[int, int] | None:
    """line number (1-based) -> author unix time; None if the file can't be blamed."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "blame", "--line-porcelain", "-t", "--", path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_BLAME_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _log.debug("blame failed for %s: %s", path, e)
        return None
    if proc.returncode != 0:
        return None
    times: dict[int, int] = {}
    lineno, author_time = 0, 0
    for line in proc.stdout.splitlines():
        if line.startswith("author-time "):
            author_time = int(line.split()[1])
        elif line.startswith("\t"):  # the content line closes each blame record
            lineno += 1
            times[lineno] = author_time
    return times


def _cache_path(repo: str, language: str, max_chunk_size: int) -> Path:
    return _CACHE_DIR / f"{repo.replace('/', '__')}__{language}__{max_chunk_size}.json"


def _blame_chunks(repo: str, chunks: list["Chunk"]) -> ChunkDates:
    """Blame every chunk-bearing file; record each chunk's first and newest line dates."""
    repo_dir = _repo_dir(repo)
    by_file: dict[str, list["Chunk"]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.filepath, []).append(chunk)

    paths = list(by_file)
    with ThreadPoolExecutor(max_workers=_BLAME_WORKERS) as pool:
        blamed = pool.map(lambda p: _blame_line_times(repo_dir, p), paths)

    dates: ChunkDates = {}
    for path, times in zip(paths, blamed):
        if times is None:
            # Undateable (untracked or blame error): left out, so it is dropped.
            continue
        for chunk in by_file[path]:
            span = [
                times[i]
                for i in range(chunk.start_line, chunk.end_line + 1)
                if i in times
            ]
            if not span:
                continue
            first = times.get(chunk.start_line, min(span))
            dates[(chunk.filepath, chunk.start_line, chunk.end_line)] = (first, max(span))
    return dates


def get_chunk_dates(
    repo: str, language: str, chunks: list["Chunk"], max_chunk_size: int
) -> ChunkDates:
    """Cached per-chunk blame dates. Blames the repo on first call."""
    cache = _cache_path(repo, language, max_chunk_size)
    if cache.is_file():
        try:
            raw = json.loads(cache.read_text())
            return {(p, a, b): (first, last) for p, a, b, first, last in raw}
        except (ValueError, TypeError):
            _log.warning("discarding unreadable blame cache %s", cache)

    _log.info("blaming %s (%s) for temporal filter — first run is slow", repo, language)
    dates = _blame_chunks(repo, chunks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(sorted([p, a, b, first, last] for (p, a, b), (first, last) in dates.items()))
    )
    return dates


def filter_chunks(
    repo: str,
    language: str,
    chunks: list["Chunk"],
    max_chunk_size: int,
    policy: str = "start",
) -> tuple[list["Chunk"], list[int]]:
    """Drop chunks authored after the repo's *index* cutoff (its latest sample).

    Returns the surviving chunks alongside their policy-selected dates, so the
    caller can narrow further to an individual sample's own cutoff at query
    time without rebuilding the index. No cutoff -> chunks unchanged, dates
    empty (nothing to filter per-sample against).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown cutoff policy {policy!r}; expected one of {POLICIES}")
    cutoff = get_repo_index_cutoffs().get(repo)
    if cutoff is None or not chunks:
        return chunks, []

    dates = get_chunk_dates(repo, language, chunks, max_chunk_size)
    idx = 0 if policy == "start" else 1
    kept: list["Chunk"] = []
    kept_dates: list[int] = []
    for c in chunks:
        d = dates.get((c.filepath, c.start_line, c.end_line))
        if d is not None and d[idx] <= cutoff:
            kept.append(c)
            kept_dates.append(d[idx])
    _log.info(
        "temporal filter %s (%s, %s): %d/%d chunks kept (index cutoff %s)",
        repo,
        language,
        policy,
        len(kept),
        len(chunks),
        datetime.fromtimestamp(cutoff, timezone.utc).date(),
    )
    return kept, kept_dates
