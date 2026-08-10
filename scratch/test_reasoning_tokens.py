"""Compare a plain call vs. a reasoning-enabled call for qwen/qwen3.5-flash-02-23 on OpenRouter.

Usage: OPENROUTER_API_KEY=... uv run python scratch/test_reasoning_tokens.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from mas_code_sum.methods.llm_client import make_clients

MODEL = "qwen/qwen3.5-flash-02-23"
PROMPT = "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning."


def run(label: str, *, reasoning: bool) -> None:
    client, _ = make_clients("openrouter")
    kwargs = {}
    if reasoning:
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        **kwargs,
    )

    message = response.choices[0].message
    reasoning_text = getattr(message, "reasoning", None)
    usage = response.usage

    print(f"\n=== {label} ===")
    print(f"content: {message.content}")
    print(f"reasoning: {reasoning_text!r}")
    if usage is not None:
        print(f"usage: {usage}")


if __name__ == "__main__":
    run("without reasoning", reasoning=False)
    run("with reasoning", reasoning=True)
