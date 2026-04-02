from .exact_copy import ExactCopySummarizer
from .zero_shot_llm import ZeroShotLLMSummarizer

REGISTRY = {
    "exact_copy": ExactCopySummarizer,
    "zero_shot_llm": ZeroShotLLMSummarizer,
}
