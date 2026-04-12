"""Evaluation metrics for code summarization."""

from .evaluator import bleu as _bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute BLEU, ROUGE-L, and BERTScore (roberta-large) F1 scores.

    Args:
        predictions: generated summaries
        references: ground truth summaries

    Returns:
        dict with keys: bleu, rougeL, bertscore_f1
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

    # BERTScore
    _, _, F1 = bert_score(predictions, references, model_type="roberta-large", verbose=False)
    bertscore_f1 = F1.mean().item()

    n = len(predictions)
    return {
        "bleu": bleu,
        "rougeL": rougeL / n,
        "bertscore_f1": bertscore_f1,
    }
