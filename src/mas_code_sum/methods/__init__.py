from .exact_copy import ExactCopySummarizer
from .few_shot_llm import FewShotLLMSummarizer
from .zero_shot_llm import ZeroShotLLMSummarizer

REGISTRY = {
    "exact_copy": ExactCopySummarizer,
    "zero_shot_llm": ZeroShotLLMSummarizer,
    "few_shot_llm": FewShotLLMSummarizer,
}
