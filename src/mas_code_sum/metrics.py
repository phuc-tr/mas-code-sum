"""Evaluation metrics for code summarization."""

from .evaluator import bleu as _bleu
from rouge_score import rouge_scorer


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute BLEU and ROUGE-L scores.

    Args:
        predictions: generated summaries
        references: ground truth summaries

    Returns:
        dict with keys: bleu, rougeL
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

    n = len(predictions)
    return {
        "bleu": bleu,
        "rougeL": rougeL / n,
    }
