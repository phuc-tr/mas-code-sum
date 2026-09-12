"""Score the story-notebook runs with SIDE (Mastropaolo et al., 2024).

SIDE is a reference-free code-summarization metric: a fine-tuned MPNet bi-encoder
embeds the code and the candidate summary separately, and the score is their cosine
similarity. Unlike BLEU it never looks at the human reference, so it measures
something orthogonal to what `metrics.py` already reports.

The checkpoint is not on the HF Hub -- download it from the Drive folder linked in
https://github.com/antonio-mastropaolo/code-summarization-metric (take the
hard-negatives variant, which is SIDE as reported in the paper) and pass the local
directory as --model-path.

Runs are read straight out of the story notebook's `RUNS` registry and filtered to the
asap and cara methods -- the comparison this is for -- across whatever models the
registry lists for them. Only the Java half of each run is scored; see SCORE_LANGUAGES
for why. The code side of each pair is
the dataset's own `code_tokens`, space-joined -- the same doc-comment-free snippet the
summarizers saw, with no re-parsing here. Output is one row per
(method, model, language, project, func_name, id) with its SIDE score, joinable onto
the notebook's per-row BLEU/RTC frame.

Usage:
    python scripts/side_metric.py                 # uses hard-negatives/ in the repo root
    python scripts/side_metric.py --limit 50      # smoke test
"""

import argparse
import ast
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from mas_code_sum.data import load_samples  # noqa: E402

DEFAULT_NOTEBOOK = REPO_ROOT / "scratch" / "cara_bleu_rtc_story.ipynb"
DEFAULT_OUT = REPO_ROOT / "scratch" / "side_scores.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "hard-negatives"
# The notebook's runs cover the Python + Java test splits, so the join index spans both
# -- that way a prediction row with no dataset match is still a hard error rather than
# being mistaken for one of the languages we deliberately drop.
LANGUAGES = ["python", "java"]
# ...but only Java is scored. SIDE is fine-tuned on CodeSearchNet Java with Javadoc
# first sentences as positives, and it does not transfer: on our test splits it tells a
# function's own summary from a random one 89% of the time in Java and 66% in Python
# (top-1 of 120: 43% vs 5%). Python scores would be noise dressed up as a metric.
SCORE_LANGUAGES = ["java"]
# Only these methods are scored; the registry's other runs are skipped.
METHODS = ["asap", "cara"]
# The paper truncates the concatenated method+summary at 512 tokens.
MAX_LENGTH = 512


def load_run_registry(notebook: Path) -> "OrderedDict[tuple[str, str | None], str]":
    """Pull `REPO_ROOT` and `RUNS` out of the notebook, keeping only `METHODS`.

    Parsing rather than importing keeps the notebook the single source of truth for
    where each run's predictions live: repoint a run there and it is scored from the
    new path here too, with no second list of paths to forget to update.
    """
    nb = json.loads(notebook.read_text())
    source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)

    wanted = {"REPO_ROOT", "RUNS"}
    namespace: dict = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<runs>", "exec"), namespace)

    missing = wanted - namespace.keys()
    if missing:
        raise SystemExit(f"{notebook}: could not find {', '.join(sorted(missing))} in any code cell")

    runs = OrderedDict(
        (name, path) for name, path in namespace["RUNS"].items() if name[0] in METHODS
    )
    if not runs:
        raise SystemExit(
            f"{notebook}: RUNS has no {' or '.join(METHODS)} entry -- only "
            f"{', '.join(sorted({n[0] for n in namespace['RUNS']}))}"
        )
    return runs


def build_code_index() -> dict[tuple[str, str, str], tuple[str, str]]:
    """(project, func_name, id) -> (code, language) for the notebook's test splits.

    The code is the dataset's own `code_tokens`, space-joined. Those tokens already
    exclude the doc comment -- the doc comment *is* the reference summary, and scoring
    a candidate against a snippet that contains it would push every method toward the
    ceiling and erase the between-method differences this is measuring. Taking the
    dataset's word for it keeps the snippet identical to what the summarizers were
    shown, with no second, script-local notion of "the code" to drift from it.
    """
    index = {}
    for language in LANGUAGES:
        for sample in load_samples(language, split="test", dataset="full"):
            key = (sample["repo"], sample["func_name"], str(sample["id"]))
            index[key] = (" ".join(sample["code_tokens"]), sample["language"])
    return index


def resolve_model_path(path: Path) -> Path:
    """Accept either the checkpoint dir itself or the folder downloaded from Drive.

    The Drive download nests the weights one level down in a step-numbered dir
    (`hard-negatives/141205/`), so point at either and this finds the config.json.
    """
    if (path / "config.json").exists():
        return path
    candidates = sorted(p for p in path.glob("*/config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise SystemExit(f"{path}: no config.json here or one level down -- is this the SIDE checkpoint?")
    raise SystemExit(
        f"{path}: several checkpoints ({', '.join(c.parent.name for c in candidates)}); "
        f"pass the one you want as --model-path"
    )


class SideScorer:
    def __init__(self, model_path: str, batch_size: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str], desc: str = "encoding") -> torch.Tensor:
        """Mean-pooled, L2-normalized embeddings, as SIDE's own inference snippet does."""
        out = []
        starts = range(0, len(texts), self.batch_size)
        for start in tqdm(starts, desc=desc, unit="batch", leave=False):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(out) if out else torch.empty(0)

    def embed_unique(self, texts: list[str], desc: str = "encoding") -> dict[str, torch.Tensor]:
        """Embed each distinct string once.

        The runs share one sample set, so every code snippet would otherwise be
        re-encoded once per run -- and identical predictions recur within a run too.
        """
        unique = list(dict.fromkeys(texts))
        return dict(zip(unique, self.embed(unique, desc=desc)))


def read_predictions(path: str, limit: int | None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"local SIDE checkpoint directory (default: {DEFAULT_MODEL_DIR.name}/ in the repo root)",
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="torch CPU threads; more is not better -- all 18 cores measured 3x slower than 8",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="only score the first N rows of each run (smoke test)")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    runs = load_run_registry(args.notebook)
    print(f"{len(runs)} runs from {args.notebook.name}: {', '.join(m for m, _ in runs)}")

    code_index = build_code_index()
    print(f"code index: {len(code_index)} samples across {', '.join(LANGUAGES)}")

    # Resolve every row's code first, so a bad join fails before the model is loaded.
    per_run_rows: dict[tuple[str, str | None], list[dict]] = {}
    missing = 0
    skipped = 0
    for name, path in runs.items():
        rows = []
        for r in read_predictions(path, args.limit):
            key = (r["project"], r["func_name"], r["id"])
            if key not in code_index:
                missing += 1
                continue
            code, language = code_index[key]
            if language not in SCORE_LANGUAGES:
                skipped += 1
                continue
            rows.append({**r, "_code": code, "_language": language})
        per_run_rows[name] = rows
    if missing:
        raise SystemExit(
            f"{missing} prediction rows had no matching (project, func_name, id) in the test "
            f"splits -- the runs and dataset/ are out of sync"
        )
    if skipped:
        print(f"skipped {skipped} rows outside {', '.join(SCORE_LANGUAGES)}")
    if not any(per_run_rows.values()):
        raise SystemExit(
            f"no {', '.join(SCORE_LANGUAGES)} rows to score -- with --limit "
            f"{args.limit}, every run's slice falls outside it"
        )

    total = sum(len(r) for r in per_run_rows.values())
    model_path = resolve_model_path(args.model_path)
    print(f"{total} rows to score; loading {model_path}")
    t0 = time.time()
    scorer = SideScorer(str(model_path), args.batch_size, args.device)
    print(f"model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    code_emb = scorer.embed_unique(
        [r["_code"] for rows in per_run_rows.values() for r in rows], desc="code"
    )
    print(f"{len(code_emb)} distinct snippets encoded in {time.time() - t0:.1f}s")

    results = []
    for (method, model), rows in tqdm(per_run_rows.items(), desc="runs", unit="run"):
        t0 = time.time()
        summary_emb = scorer.embed_unique(
            [r["prediction"] for r in rows], desc=f"{method} {model or '-'}"
        )
        scores = []
        for r in rows:
            score = float(torch.dot(code_emb[r["_code"]], summary_emb[r["prediction"]]))
            scores.append(score)
            results.append(
                {
                    "method": method,
                    "model": model or "",
                    "language": r["_language"],
                    "project": r["project"],
                    "func_name": r["func_name"],
                    "id": r["id"],
                    "side": score,
                }
            )
        mean = sum(scores) / len(scores) if scores else float("nan")
        elapsed = time.time() - t0
        print(
            f"  {method:9s} {model or '-':4s} n={len(scores):5d} SIDE={mean:.4f} "
            f"({elapsed:.1f}s, {elapsed / max(len(scores), 1) * 1000:.0f} ms/row)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "model", "language", "project", "func_name", "id", "side"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
