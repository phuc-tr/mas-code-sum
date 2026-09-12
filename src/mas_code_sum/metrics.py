"""Evaluation metrics for code summarization."""

import logging
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .evaluator import bleu as _bleu
from rouge_score import rouge_scorer

CHRF_MAX_N = 6
CHRF_BETA = 2.0

# METEOR 1.5 needs Java and the CMU jar, which is too large to vendor. Fetch it with:
#     curl -sL https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz | tar xz
# and keep it at the repo root (or point METEOR_JAR elsewhere).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METEOR_JAR = _REPO_ROOT / "meteor-1.5" / "meteor-1.5.jar"
_meteor_warned = False


def _ngram_pr(ref_units: list, hyp_units: list, max_n: int) -> list[tuple[float, float]]:
    """Per-order (precision, recall). Orders empty on both sides are skipped, as
    sacreBLEU does, so a short summary is not charged for missing orders."""
    pairs = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_units[i : i + n]) for i in range(len(ref_units) - n + 1))
        hyp_ng = Counter(tuple(hyp_units[i : i + n]) for i in range(len(hyp_units) - n + 1))
        if not ref_ng and not hyp_ng:
            continue
        overlap = sum((ref_ng & hyp_ng).values())
        pairs.append(
            (
                overlap / sum(hyp_ng.values()) if hyp_ng else 0.0,
                overlap / sum(ref_ng.values()) if ref_ng else 0.0,
            )
        )
    return pairs


def chrf(reference: str, prediction: str, beta: float = CHRF_BETA) -> float:
    """chrF (Popovic, 2015) for one pair, 0-100.

    Character n-grams up to order 6 with beta=2, the standard weighting (as in
    sacreBLEU's default). Whitespace is stripped from both sides, which also
    makes the score blind to the references' spaced-punctuation style
    ("sub - agents ." vs "sub-agents.") that some methods mimic and others do not.
    """
    pairs = _ngram_pr(
        list("".join(reference.split())), list("".join(prediction.split())), CHRF_MAX_N
    )
    if not pairs:
        return 0.0
    p = sum(x for x, _ in pairs) / len(pairs)
    r = sum(x for _, x in pairs) / len(pairs)
    if p == 0.0 and r == 0.0:
        return 0.0
    b2 = beta**2
    return 100 * (1 + b2) * p * r / (b2 * p + r)


def meteor_jar() -> Path | None:
    """The METEOR 1.5 jar, or None if it is not installed."""
    jar = Path(os.environ.get("METEOR_JAR") or DEFAULT_METEOR_JAR).expanduser()
    return jar if jar.exists() else None


def meteor(references: list[str], predictions: list[str]) -> list[float] | None:
    """Per-segment METEOR 1.5 scores, 0-100, or None if the jar is missing.

    Runs the official CMU jar -- not nltk's `meteor_score`, whose untuned
    METEOR 1.0 parameters (beta=3.0 vs 1.5's 0.2) score roughly twice as high on
    this data. `-norm` lets METEOR's own tokenizer handle punctuation, and `-q`
    puts per-segment scores on stderr; the aggregate on stdout is a corpus-level
    figure, so it is ignored in favour of a per-sample mean like every other
    metric here.

    A missing jar returns None rather than raising: an expensive run should not
    die at scoring time over an optional dependency.
    """
    global _meteor_warned
    jar = meteor_jar()
    if jar is None:
        if not _meteor_warned:
            logging.warning(
                "METEOR 1.5 jar not found at %s (or $METEOR_JAR) -- skipping the "
                "`meteor` metric. See src/mas_code_sum/metrics.py for the download.",
                DEFAULT_METEOR_JAR,
            )
            _meteor_warned = True
        return None

    one_line = lambda s: " ".join(str(s).split())  # the jar pairs files by line
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.txt"
        hyp_path = Path(tmp) / "hyp.txt"
        ref_path.write_text("".join(one_line(s) + "\n" for s in references))
        hyp_path.write_text("".join(one_line(s) + "\n" for s in predictions))
        proc = subprocess.run(
            ["java", "-Xmx2G", "-jar", str(jar), str(hyp_path), str(ref_path),
             "-l", "en", "-norm", "-q"],
            capture_output=True, text=True, check=True,
        )
    scores = [100 * float(tok) for tok in proc.stderr.split() if tok.strip()]
    assert len(scores) == len(references), (
        f"METEOR returned {len(scores)} segment scores for {len(references)} segments"
    )
    return scores


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute BLEU, ROUGE-L, chrF and METEOR scores.

    BLEU, chrF and METEOR are on a 0-100 scale; ROUGE-L stays 0-1, as it always
    has. `meteor` is omitted when the METEOR 1.5 jar is not installed.

    Args:
        predictions: generated summaries
        references: ground truth summaries

    Returns:
        dict with keys: bleu, rougeL, chrf, and meteor when available
    """
    # BLEU: per-sentence average using evaluator.py implementation (0-100 scale)
    bleu_scores = [_bleu([ref], pred)[0] for pred, ref in zip(predictions, references)]
    bleu = sum(bleu_scores) / len(bleu_scores) * 100

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rougeL += scores["rougeL"].fmeasure

    chrf_total = sum(chrf(ref, pred) for pred, ref in zip(predictions, references))

    n = len(predictions)
    metrics = {
        "bleu": bleu,
        "rougeL": rougeL / n,
        "chrf": chrf_total / n,
    }

    meteor_scores = meteor(references, predictions)
    if meteor_scores is not None:
        metrics["meteor"] = sum(meteor_scores) / n

    return metrics
