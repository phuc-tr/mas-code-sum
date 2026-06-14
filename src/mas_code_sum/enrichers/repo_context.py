"""Shared prompt-block builder for repo-context-enriched few-shot prompts."""


def build_block(
    code: str,
    repo: str | None,
    about: str | None,
    path: str | None,
    docstring: str | None,
) -> str:
    """Build one prompt block in the repo-context few-shot format.

    Example block (few-shot):
      Repository: {repo}
      Repository description: {about}
      File: {path}
      Code:
      {code}
      Summary: <s>{docstring}</s>

    Query block (docstring=None):
      ... same fields ...
      Summary: <s>
    """
    parts: list[str] = []
    if repo:
        parts.append(f"Repository: {repo}")
    if about:
        parts.append(f"Repository description: {about}")
    if path:
        parts.append(f"File: {path}")
    parts.append(f"Code:\n{code}")
    if docstring is not None:
        parts.append(f"Summary: <s>{docstring}</s>")
    else:
        parts.append("Summary: <s>")
    return "\n".join(parts)
