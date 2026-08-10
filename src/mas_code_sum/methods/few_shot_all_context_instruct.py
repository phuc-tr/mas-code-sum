"""Instruct-style variant of few_shot_all_context.

Same signals as `few_shot_all_context` — retrieved few-shot examples, repo
metadata, file context (module doc / enclosing class / imports), and the file
outline — but assembled into an instructed chat prompt instead of a base-model
text completion, so it works with instruct and thinking models.

The instruction preamble/closing and reply cleanup are shared with
`agentic_rag`, so the two methods differ only in where the extra context comes
from (file context + outline here vs. agent-gathered AST chunks there).

Prompt structure:
  You are ... (task instruction)
  Here are examples ...
  <example blocks: Repository/File/Code/Summary: {docstring}>
  Repository: {repo}
  Repository description: {about}
  File: {path}
  <file context: module doc / enclosing class / imports>
  Other functions in file:
  <outline of sibling functions/classes>
  Now summarize the following function:
  Code:
  {code}
  Output only the one-sentence summary ...
"""

from ..data import get_metadata_index
from .agentic_rag import (
    SUMMARIZER_FINAL_INSTRUCTION,
    SUMMARIZER_INSTRUCTION,
    clean_summary_reply,
)
from .few_shot_all_context import FewShotAllContextSummarizer


class FewShotAllContextInstructSummarizer(FewShotAllContextSummarizer):
    """few_shot_all_context with an instructed chat prompt (instruct/thinking models)."""

    name = "few_shot_all_context_instruct"

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_tokens: int = 2048,  # roomy: thinking models reason before answering
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens

    def _example_block(self, s: dict) -> str:
        code = " ".join(s["code_tokens"])
        docstring = " ".join(s["docstring_tokens"])
        repo = s.get("repo")
        about: str | None = None
        if repo and self.use_repo:
            about = get_metadata_index().get(repo, {}).get("about")
        path = s.get("path") if self.example_paths else None

        parts: list[str] = []
        if repo:
            parts.append(f"Repository: {repo}")
        if about:
            parts.append(f"Repository description: {about}")
        if path:
            parts.append(f"File: {path}")
        parts.append(f"Code:\n{code}")
        parts.append(f"Summary: {docstring}")
        return "\n".join(parts)

    def build_prompt(
        self,
        code: str,
        language: str,
        project: str | None,
        path: str | None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
        func_name: str | None = None,
    ) -> str:
        examples = self.retriever.retrieve(code, language, project=project, path=path)
        sections = [SUMMARIZER_INSTRUCTION]
        if examples:
            sections.append("Here are examples of well-summarized functions:")
            sections.extend(self._example_block(s) for s in examples)
        context_parts = self._context_parts(
            code, language, project, path, blame_timestamp=blame_timestamp, blame_sha=blame_sha, func_name=func_name
        )
        if context_parts:
            sections.append("\n".join(context_parts))
        sections.append(f"Now summarize the following function:\n\nCode:\n{code}")
        sections.append(SUMMARIZER_FINAL_INSTRUCTION)
        return "\n\n".join(sections)

    async def async_summarize(
        self,
        code: str,
        language: str,
        project: str | None = None,
        path: str | None = None,
        url: str | None = None,
        blame_timestamp: str | None = None,
        blame_sha: str | None = None,
        func_name: str | None = None,
    ) -> str:
        prompt = self.build_prompt(code, language, project, path, blame_timestamp=blame_timestamp, blame_sha=blame_sha, func_name=func_name)
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return clean_summary_reply(response.choices[0].message.content or "")

    def params(self) -> dict:
        return {**super().params(), "max_tokens": self.max_tokens}
