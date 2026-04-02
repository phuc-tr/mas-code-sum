"""Evaluation metrics for code summarization."""

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute BLEU and ROUGE scores.

    Args:
        predictions: generated summaries
        references: ground truth summaries

    Returns:
        dict with keys: bleu, rouge1, rouge2, rougeL
    """
    # BLEU
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_preds = [pred.split() for pred in predictions]
    smoother = SmoothingFunction().method1
    bleu = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoother)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    n = len(predictions)
    return {
        "bleu": bleu,
        "rouge1": rouge1 / n,
        "rouge2": rouge2 / n,
        "rougeL": rougeL / n,
    }
