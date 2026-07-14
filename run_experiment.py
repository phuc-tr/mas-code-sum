"""CLI entrypoint for running experiments.

Usage:
    python run_experiment.py experiments/example.yaml
    python run_experiment.py experiments/few_shot_llm_bm25.yaml --model meta-llama/Meta-Llama-3.1-8B
    python run_experiment.py experiments/few_shot_llm_bm25.yaml --model deepseek/deepseek-chat-v3.1 --backend openrouter
    python run_experiment.py experiments/few_shot_llm_bm25.yaml --model Qwen/Qwen3-8B --tag experiment=ablation --tag notes=no_imports
    python run_experiment.py experiments/few_shot_llm_bm25.yaml --dataset small
"""

import argparse
import shlex
import sys

import yaml

from src.mas_code_sum.runner import run_experiment
from src.mas_code_sum.methods import REGISTRY
from src.mas_code_sum.retrievers import RETRIEVER_REGISTRY


def main() -> None:
    cli_command = shlex.join(getattr(sys, "orig_argv", sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML")
    parser.add_argument("--model", default=None, help="Override method_params.model")
    parser.add_argument("--backend", default=None, help="Override method_params.backend")
    parser.add_argument("--dataset", default="full", choices=["full", "small"], help="Dataset variant to load (default: full)")
    parser.add_argument("--tag", metavar="KEY=VALUE", action="append", default=[], help="Set an MLflow tag (repeatable)")
    parser.add_argument(
        "--rtc", metavar="MODEL", default=None,
        help="Backward model to use for round-trip correctness (RTC). If omitted, the rtc_bleu metric is not computed.",
    )
    parser.add_argument("--rtc-backend", default="openrouter", help="Backend for the RTC backward model (default: openrouter)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method_key = cfg["method"]
    if method_key not in REGISTRY:
        raise ValueError(f"Unknown method '{method_key}'. Available: {list(REGISTRY)}")

    retriever = None
    if retriever_key := cfg.get("retriever"):
        if retriever_key not in RETRIEVER_REGISTRY:
            raise ValueError(f"Unknown retriever '{retriever_key}'. Available: {list(RETRIEVER_REGISTRY)}")
        retriever_params = {"dataset": args.dataset, **cfg.get("retriever_params", {})}
        retriever = RETRIEVER_REGISTRY[retriever_key](**retriever_params)

    method_params = cfg.get("method_params", {})
    if args.model is not None:
        method_params["model"] = args.model
    if args.backend is not None:
        method_params["backend"] = args.backend
    if retriever is not None:
        method_params = {**method_params, "retriever": retriever}

    method = REGISTRY[method_key](**method_params)

    tags: dict[str, str] = {}
    for kv in args.tag:
        if "=" not in kv:
            raise ValueError(f"--tag must be KEY=VALUE, got: {kv!r}")
        k, v = kv.split("=", 1)
        tags[k] = v

    run_experiment(
        method=method,
        languages=cfg.get("languages"),
        max_samples=cfg.get("max_samples"),
        num_runs=cfg.get("num_runs", 1),
        projects=cfg.get("projects"),
        dataset=args.dataset,
        tags=tags,
        cli_command=cli_command,
        rtc_model=args.rtc,
        rtc_backend=args.rtc_backend,
    )


if __name__ == "__main__":
    main()
