from .agentic_rag import AgenticRagSummarizer
from .agentic_rag_all_context import AgenticRagAllContextSummarizer
from .codet5_summarizer import CodeT5Summarizer
from .exact_copy import ExactCopySummarizer
from .few_shot_all_context import FewShotAllContextSummarizer
from .few_shot_all_context_instruct import FewShotAllContextInstructSummarizer
from .few_shot_asap import FewShotAsapSummarizer
from .few_shot_context_enriched import FewShotContextEnrichedSummarizer
from .few_shot_file_context import FewShotFileContextSummarizer
from .few_shot_llm import FewShotLLMSummarizer
from .zero_shot_context_enriched import ZeroShotContextEnrichedSummarizer
from .zero_shot_llm import ZeroShotLLMSummarizer

REGISTRY = {
    "agentic_rag": AgenticRagSummarizer,
    "agentic_rag_all_context": AgenticRagAllContextSummarizer,
    "exact_copy": ExactCopySummarizer,
    "zero_shot_llm": ZeroShotLLMSummarizer,
    "zero_shot_context_enriched": ZeroShotContextEnrichedSummarizer,
    "few_shot_llm": FewShotLLMSummarizer,
    "few_shot_context_enriched": FewShotContextEnrichedSummarizer,
    "few_shot_file_context": FewShotFileContextSummarizer,
    "few_shot_all_context": FewShotAllContextSummarizer,
    "few_shot_all_context_instruct": FewShotAllContextInstructSummarizer,
    "few_shot_asap": FewShotAsapSummarizer,
    "codet5": CodeT5Summarizer,
}
