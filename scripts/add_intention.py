"""Add 'intention' field to train.jsonl and test.jsonl files using an LLM classifier."""

import asyncio
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.1-8b-instruct"
MAX_CONCURRENCY = 20
VALID_INTENTIONS = {"What", "Why", "How-to-use-it", "How-it-is-done", "Property"}
MAX_RETRIES = 3

PROMPT_TEMPLATE = """\
Classify the following code documentation comment into exactly one of these intention categories:
- What: describes what the function/method is or does (its purpose or identity)
- Why: explains the reason or motivation behind the function
- How-to-use-it: describes how to call or use the function (parameters, usage examples)
- How-it-is-done: explains the implementation approach or algorithm used internally
- Property: describes a property, attribute, or characteristic (return value, type, constraint)

Documentation comment:
{docstring}

Respond with exactly one category name from the list above. Do not include any other text."""


async def classify_intention(client: AsyncOpenAI, docstring: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(MAX_RETRIES):
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(docstring=docstring)}],
                max_tokens=32,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            if raw in VALID_INTENTIONS:
                return raw
            cleaned = raw.strip("\"'.,;:")
            if cleaned in VALID_INTENTIONS:
                return cleaned
            print(f"  [attempt {attempt+1}/{MAX_RETRIES}] Invalid response: {repr(raw)}", file=sys.stderr)
        raise ValueError(f"LLM returned invalid intention after {MAX_RETRIES} attempts: {repr(raw)}")


async def process_file(client: AsyncOpenAI, path: Path, sem: asyncio.Semaphore) -> None:
    bak_path = path.with_suffix(".jsonl.bak")
    print(f"Backing up {path} -> {bak_path}")
    shutil.copy2(path, bak_path)

    lines = path.read_text().splitlines()
    records = []
    for line in lines:
        if line.strip():
            records.append(json.loads(line))

    total = len(records)
    needs_classification = [r for r in records if "intention" not in r]
    already_done = total - len(needs_classification)
    if already_done:
        print(f"  {already_done}/{total} already classified, processing {len(needs_classification)} remaining.")

    completed = 0

    async def classify_record(record: dict) -> None:
        nonlocal completed
        docstring = record.get("docstring", "")
        if not docstring:
            raise ValueError(f"Record {record.get('id')} has no docstring")
        record["intention"] = await classify_intention(client, docstring, sem)
        completed += 1
        if completed % 50 == 0 or completed == len(needs_classification):
            print(f"  [{completed}/{len(needs_classification)}] classified")

    await asyncio.gather(*[classify_record(r) for r in needs_classification])

    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    counts = Counter(r["intention"] for r in records)
    print(f"\n--- {path.parent.name}/{path.name} ---")
    for intention in sorted(VALID_INTENTIONS):
        print(f"  {intention:<20} {counts.get(intention, 0):>5}  ({counts.get(intention, 0)/total*100:.1f}%)")
    print(f"  {'TOTAL':<20} {total:>5}")


async def main() -> None:
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    dataset_root = Path(__file__).parent.parent / "dataset"
    groups = {
        "python": [
            dataset_root / "python" / "train.jsonl",
            dataset_root / "python" / "test.jsonl",
        ],
        "java": [
            dataset_root / "java" / "train.jsonl",
            dataset_root / "java" / "test.jsonl",
        ],
    }

    for lang, files in groups.items():
        existing = [f for f in files if f.exists()]
        missing = [f for f in files if not f.exists()]
        for f in missing:
            print(f"Skipping missing file: {f}", file=sys.stderr)
        if not existing:
            continue
        print(f"\n=== Processing {lang} ===")
        await asyncio.gather(*[process_file(client, f, sem) for f in existing])


if __name__ == "__main__":
    asyncio.run(main())
