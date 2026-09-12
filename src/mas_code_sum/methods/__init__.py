from .agentic_rag import AgenticRagSummarizer
from .agentic_rag_all_context import AgenticRagAllContextSummarizer
from .codet5_summarizer import CodeT5Summarizer
from .few_shot_asap import FewShotAsapSummarizer
from .few_shot_llm import FewShotLLMSummarizer
from .zero_shot_llm import ZeroShotLLMSummarizer

REGISTRY = {
    "agentic_rag": AgenticRagSummarizer,
    "agentic_rag_all_context": AgenticRagAllContextSummarizer,
    "zero_shot_llm": ZeroShotLLMSummarizer,
    "few_shot_llm": FewShotLLMSummarizer,
    "few_shot_asap": FewShotAsapSummarizer,
    "codet5": CodeT5Summarizer,
}
