"""AST-aware repository chunk index for retrieval-augmented summarization.

Chunks every source file of a repo with `astchunk` (ASTChunkBuilder), which
splits code on AST node boundaries instead of arbitrary lines, so each chunk
is a syntactically coherent block (whole functions/classes where they fit).
Chunks are indexed with BM25 for lexical pre-ranking against a query snippet.

The index is built lazily per (repo, language) and cached for the lifetime of
the process; building walks `dataset/repos/<owner__name>` at the current
checkout (no sha pinning). Because that checkout is HEAD, chunks are then
passed through the temporal filter in `blame_cutoff.py`, which drops code
authored after the repo's test-set cutoff so future code cannot leak into
retrieved context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from astchunk import ASTChunkBuilder
from rank_bm25 import BM25Okapi

from .blame_cutoff import filter_chunks
from .file_context import _repo_dir

_log = logging.getLogger(__name__)

_EXTENSIONS = {
    "python": ".py",
    "java": ".java",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
_MAX_FILE_BYTES = 512 * 1024  # skip generated/vendored monsters
_SKIP_DIR_PARTS = {".git", "node_modules", "build", "dist", ".tox", "venv", ".venv"}


@dataclass
class Chunk:
    filepath: str  # path relative to the repo root
    content: str
    start_line: int
    end_line: int


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class ChunkIndex:
    chunks: list[Chunk]
    _bm25: BM25Okapi | None
    # Per-chunk authored date (policy-selected), parallel to `chunks`. Empty
    # when the temporal filter is off — `cutoff` is then a no-op.
    dates: list[int] | None = None

    def query(
        self, code: str, k: int,
        cutoff: int | None = None, exclude_path: str | None = None,
    ) -> list[Chunk]:
        """Top-k chunks by BM25 score.

        `exclude_path` drops every chunk from the sample's own file, since
        that file is already supplied in full as local file context —
        retrieval is meant to surface other files.

        `cutoff` (unix time) narrows the shared per-repo index to one sample:
        chunks authored after the sample was written are skipped. The index is
        already bounded by the repo's latest sample, so this only ever
        tightens. Skipping happens during the ranked walk, so the loop simply
        continues until it has `k` eligible chunks — no over-fetch needed.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(code))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out: list[Chunk] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if cutoff is not None and self.dates and self.dates[i] > cutoff:
                continue
            chunk = self.chunks[i]
            if exclude_path is not None and chunk.filepath == exclude_path:
                continue
            out.append(chunk)
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=64)
def get_chunk_index(
    repo: str,
    language: str,
    max_chunk_size: int = 1200,
    temporal_filter: bool = True,
    cutoff_policy: str = "start",
) -> ChunkIndex:
    """Build (or return the cached) AST chunk index for a repo checkout.

    With `temporal_filter` (the default), chunks authored after the repo's
    *latest* test sample are dropped before BM25 is fitted, and each surviving
    chunk's date is kept on the index so `ChunkIndex.query(cutoff=...)` can
    narrow to an individual sample. `cutoff_policy` picks whether a chunk is
    dated by its first line ("start") or its newest line ("last_touched") —
    see `blame_cutoff`. The first build per repo blames every file and is
    slow; the dates are cached on disk.

    One index per repo means BM25's document-frequency statistics are fitted
    over every chunk any sample may see, so a sample's scores are mildly
    influenced by chunks its own cutoff then hides. No chunk content leaks —
    only corpus statistics, bounded by the repo's own sample window.
    """
    ext = _EXTENSIONS.get(language)
    repo_dir = _repo_dir(repo)
    if ext is None or not repo_dir.is_dir():
        return ChunkIndex([], None)

    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size, language=language, metadata_template="default"
    )

    chunks: list[Chunk] = []
    for file_path in sorted(repo_dir.rglob(f"*{ext}")):
        rel_parts = file_path.relative_to(repo_dir).parts
        if any(part in _SKIP_DIR_PARTS for part in rel_parts):
            continue
        try:
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
            src = file_path.read_text(encoding="utf-8", errors="replace")
            file_chunks = builder.chunkify(src)
        except Exception as e:  # astchunk/tree-sitter can choke on odd files
            _log.debug("chunkify failed for %s/%s: %s", repo, file_path, e)
            continue
        rel = "/".join(rel_parts)
        for c in file_chunks:
            content = c["content"].strip()
            if not content:
                continue
            meta = c.get("metadata", {})
            chunks.append(Chunk(
                filepath=rel,
                content=content,
                start_line=meta.get("start_line_no", 0) + 1,
                end_line=meta.get("end_line_no", 0) + 1,
            ))

    dates: list[int] | None = None
    if temporal_filter:
        chunks, dates = filter_chunks(repo, language, chunks, max_chunk_size, policy=cutoff_policy)

    if not chunks:
        return ChunkIndex([], None)
    _log.info("AST chunk index for %s (%s): %d chunks", repo, language, len(chunks))
    return ChunkIndex(chunks, BM25Okapi([_tokenize(c.content) for c in chunks]), dates)
