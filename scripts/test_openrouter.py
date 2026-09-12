"""Edit the variables below and run: uv run python scripts/test_openrouter.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from mas_code_sum.methods.llm_client import make_openai_clients

MODEL = "deepseek/deepseek-chat"
PROMPT = "which model are u"

client, _ = make_openai_clients()  # restricted to Novita/DeepInfra repo-wide, see llm_client.py
response = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": PROMPT
    }],
    max_tokens=50,
    temperature=0.0,
)

print(f"Served by: {getattr(response, 'provider', '?')}")
print(response.choices[0].message.content)
# print(response.usage)
