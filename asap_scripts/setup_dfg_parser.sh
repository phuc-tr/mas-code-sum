#!/usr/bin/env bash
# Sets up the CodeXGLUE DFG parser under asap_scripts/scripts/parser/.
#
# Usage (from repo root):
#   bash scripts/setup_dfg_parser.sh
#
# After this runs you should have:
#   asap_scripts/scripts/parser/__init__.py
#   asap_scripts/scripts/parser/my-languages.so
#
# Requires: git, python3, tree-sitter (pip install tree-sitter==0.20.*)

set -e

SCRIPTS_DIR="asap_scripts/scripts"
PARSER_DIR="$SCRIPTS_DIR/parser"
SITTER_DIR="$SCRIPTS_DIR/sitter-libs"

echo "=== Setting up DFG parser ==="

# ── 1. Clone tree-sitter grammar repos (shared with setup.sh) ────────────────
mkdir -p "$SITTER_DIR"
for lang_repo in \
    "tree-sitter-go:go" \
    "tree-sitter-javascript:js" \
    "tree-sitter-python:py" \
    "tree-sitter-java:java" \
    "tree-sitter-ruby:ruby" \
    "tree-sitter-php:php"; do
    repo="${lang_repo%%:*}"
    dir="${lang_repo##*:}"
    target="$SITTER_DIR/$dir"
    if [ ! -d "$target" ]; then
        echo "Cloning tree-sitter/$repo..."
        git clone --depth 1 "https://github.com/tree-sitter/$repo" "$target"
    else
        echo "Skipping $repo (already cloned)"
    fi
done

# ── 2. Download CodeXGLUE parser Python files ────────────────────────────────
mkdir -p "$PARSER_DIR"

BASE_URL="https://raw.githubusercontent.com/microsoft/CodeBERT/master/GraphCodeBERT/codesearch/parser"

for fname in "__init__.py" "DFG.py" "utils.py" "build.py"; do
    target="$PARSER_DIR/$fname"
    if [ ! -f "$target" ]; then
        echo "Downloading parser/$fname..."
        curl -sSfL "$BASE_URL/$fname" -o "$target"
    else
        echo "Skipping parser/$fname (already exists)"
    fi
done

# ── 3. Build my-languages.so ─────────────────────────────────────────────────
echo "Building parser/my-languages.so..."
(
    cd "$SCRIPTS_DIR"
    # Use the uv-managed Python (tree_sitter v0.25 doesn't support build_library,
    # so we use the old API via a pinned in-place install)
    uv run --project "$(git -C "$OLDPWD" rev-parse --show-toplevel)" python -c "
import os, sys
sys.path.insert(0, '.')
from tree_sitter import Language

libs = []
sitter_dir = 'sitter-libs'
for d in os.listdir(sitter_dir):
    libs.append(os.path.join(sitter_dir, d))

Language.build_library('parser/my-languages.so', libs)
print('Built parser/my-languages.so')
"
)

echo ""
echo "=== DFG parser setup complete ==="
echo "parser/ module: $PARSER_DIR"
echo "Now run: python scripts/precompute_dfg.py"
