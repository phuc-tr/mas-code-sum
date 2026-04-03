"""Evaluation metrics for code summarization."""

from difflib import SequenceMatcher

from .evaluator import bleu as _bleu
from rouge_score import rouge_scorer


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute BLEU, ROUGE-L, exact match, and edit similarity scores.

    Args:
        predictions: generated summaries
        references: ground truth summaries

    Returns:
        dict with keys: bleu, rougeL, exact_match, edit_sim
    """
    # BLEU: per-sentence average using evaluator.py implementation (0-100 scale)
    bleu_scores = [_bleu([ref], pred)[0] for pred, ref in zip(predictions, references)]
    bleu = sum(bleu_scores) / len(bleu_scores) * 100

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = 0.0
    exact_match = 0.0
    edit_sim = 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rougeL += scores["rougeL"].fmeasure
        exact_match += float(pred.strip() == ref.strip())
        edit_sim += SequenceMatcher(None, pred.strip(), ref.strip()).ratio()

    n = len(predictions)
    return {
        "bleu": bleu,
        "rougeL": rougeL / n,
        "exact_match": exact_match / n,
        "edit_sim": edit_sim / n,
    }
