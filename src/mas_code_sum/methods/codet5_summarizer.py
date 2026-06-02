"""Code summarization using Salesforce/codet5-base-multi-sum."""

from .base import BaseSummarizer

MODEL_NAME = "Salesforce/codet5-base-multi-sum"


class CodeT5Summarizer(BaseSummarizer):
    """Summarize code using the CodeT5 seq2seq model (local inference)."""

    name = "codet5"

    def __init__(self, max_length: int = 20, batch_size: int = 8):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    def summarize_batch(self, codes: list[str], languages: list[str], **kwargs) -> list[str]:
        import torch
        from tqdm import tqdm

        results = []
        project = (kwargs.get("projects") or [None])[0]
        desc = project or "batches"
        for i in tqdm(range(0, len(codes), self.batch_size), desc=desc, unit="batch"):
            batch = codes[i : i + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt",
                                     padding=True,
                                     truncation=True,
                                     max_length=512,
                                     )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_length=self.max_length)
            results.extend(
                self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            )
        return results

    def summarize(self, code: str, language: str, project: str | None = None, path: str | None = None, url: str | None = None) -> str:
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
