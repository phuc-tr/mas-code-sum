"""Build a prompt with FewShotFileContextSummarizer and print it, without hitting the LLM.

Usage:
    uv run python scripts/preview_file_context_prompt.py [--language python|java]
                                                         [--no-module-doc]
                                                         [--no-class-context]
                                                         [--max-imports N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_code_sum.data import load_samples  # noqa: E402
from mas_code_sum.methods.few_shot_file_context import FewShotFileContextSummarizer  # noqa: E402
from mas_code_sum.retrievers.bm25 import BM25Retriever  # noqa: E402

# Hand-picked samples whose files exercise all file-context signals (module doc,
# enclosing class, imports). Keeps the preview deterministic per language.
DEFAULT_SAMPLES = {
    "python": {
        "repo": "apache/airflow",
        "path": "airflow/contrib/hooks/bigquery_hook.py",
        "func_name": "BigQueryBaseCursor.get_dataset",
    },
    "java": {
        "repo": "spring-projects/spring-security",
        "path": "ldap/src/main/java/org/springframework/security/ldap/SpringSecurityLdapTemplate.java",
        "func_name": "SpringSecurityLdapTemplate.searchForSingleEntryInternal",
    },
}


def _pick_sample(language: str) -> dict:
    target = DEFAULT_SAMPLES[language]
    samples = load_samples(language, split="test")
    for s in samples:
        if (
            s["repo"] == target["repo"]
            and s["path"] == target["path"]
            and s["func_name"] == target["func_name"]
        ):
            return s
    raise RuntimeError(
        f"Default preview sample not found for {language}: {target}. "
        "Pick a different one in DEFAULT_SAMPLES."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("python", "java"), default="python")
    p.add_argument("--no-module-doc", action="store_true", help="Disable module docstring line")
    p.add_argument("--no-class-context", action="store_true", help="Disable enclosing class line")
    p.add_argument("--max-imports", type=int, default=25, help="Cap imports (0 disables)")
    p.add_argument("--n-shots", type=int, default=3, help="Number of few-shot examples")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample = _pick_sample(args.language)

    retriever = BM25Retriever(n=args.n_shots)
    method = FewShotFileContextSummarizer.__new__(FewShotFileContextSummarizer)
    # Bypass __init__ (which builds OpenAI clients needing an API key).
    method.model = "n/a"
    method.retriever = retriever
    method.example_paths = False
    method.max_concurrency = 1
    method.use_module_doc = not args.no_module_doc
    method.use_class_context = not args.no_class_context
    method.max_imports = args.max_imports

    code = " ".join(sample["code_tokens"])
    prompt = method.build_prompt(
        code=code,
        language=sample["language"],
        project=sample["repo"],
        path=sample["path"],
    )

    print(f"=== Query ({args.language}): {sample['repo']} :: {sample['func_name']} ===")
    print(f"Path: {sample['path']}")
    print(f"Gold summary: {' '.join(sample['docstring_tokens'])}")
    print(
        f"Knobs: use_module_doc={method.use_module_doc} "
        f"use_class_context={method.use_class_context} "
        f"max_imports={method.max_imports} n_shots={args.n_shots}"
    )
    print()
    print("=== PROMPT ===")
    print(prompt)
    print("=== END ===")
    print(f"\nPrompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens (rough)")


if __name__ == "__main__":
    main()
