"""Code summarization using Salesforce/codet5-base-multi-sum."""

from .base import BaseSummarizer

MODEL_NAME = "Salesforce/codet5-base-multi-sum"


class CodeT5Summarizer(BaseSummarizer):
    """Summarize code using the CodeT5 seq2seq model (local inference)."""

    name = "codet5"

    def __init__(self, max_length: int = 20):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        import torch

        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    def summarize(self, code: str, language: str, project: str | None = None) -> str:
        import torch

        input_ids = self._tokenizer(code, return_tensors="pt").input_ids.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(input_ids, max_length=self.max_length)

        return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "max_length": self.max_length,
        }
