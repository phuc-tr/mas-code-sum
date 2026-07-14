"""AST-aware repository chunk index for retrieval-augmented summarization.

Chunks every source file of a repo with `astchunk` (ASTChunkBuilder), which
splits code on AST node boundaries instead of arbitrary lines, so each chunk
is a syntactically coherent block (whole functions/classes where they fit).
Chunks are indexed with BM25 for lexical pre-ranking against a query snippet.

The index is built lazily per (repo, language) and cached for the lifetime of
the process; building walks `dataset/repos/<owner__name>` at the current
checkout (no sha pinning).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from astchunk import ASTChunkBuilder
from rank_bm25 import BM25Okapi

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


_NORM_RE = re.compile(r"[^a-z0-9]+")

def _norm_text(text: str) -> str:
    """Normalize text for leakage containment checks (case/punct/space-insensitive)."""
    return _NORM_RE.sub(" ", text.lower()).strip()


# Below this length a normalized exclude_text is too generic — containment would
# discard legitimately useful chunks (e.g. reference "returns true").
_MIN_EXCLUDE_CHARS = 20


@dataclass
class ChunkIndex:
    chunks: list[Chunk]
    _bm25: BM25Okapi | None

    def query(
        self, code: str, k: int, exclude_text: str | None = None
    ) -> list[Chunk]:
        """Top-k chunks by BM25 score.

        Drops chunks containing the query code itself (normalized
        containment) — this is what keeps the target function's own chunk out
        of its context without excluding the rest of its file. `exclude_text`
        does the same for the ground-truth docstring, to keep it out of
        prompts when the target function has copies elsewhere in the repo, or
        moved since the dataset snapshot. Either exclusion is ignored when
        too short to be distinctive.
        """
        if self._bm25 is None:
            return []
        excludes = [
            norm for norm in (_norm_text(t) for t in (code, exclude_text) if t)
            if len(norm) >= _MIN_EXCLUDE_CHARS
        ]
        scores = self._bm25.get_scores(_tokenize(code))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out: list[Chunk] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            content_norm = _norm_text(chunk.content)
            if any(excl in content_norm for excl in excludes):
                continue
            out.append(chunk)
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=64)
def get_chunk_index(repo: str, language: str, max_chunk_size: int = 1200) -> ChunkIndex:
    """Build (or return the cached) AST chunk index for a repo checkout."""
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

    if not chunks:
        return ChunkIndex([], None)
    _log.info("AST chunk index for %s (%s): %d chunks", repo, language, len(chunks))
    return ChunkIndex(chunks, BM25Okapi([_tokenize(c.content) for c in chunks]))
