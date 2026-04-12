"""CLI entrypoint for running experiments.

Usage:
    python run_experiment.py experiments/example.yaml
"""

import sys
import yaml

from src.mas_code_sum.runner import run_experiment
from src.mas_code_sum.methods import REGISTRY
from src.mas_code_sum.retrievers import RETRIEVER_REGISTRY


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    method_key = cfg["method"]
    if method_key not in REGISTRY:
        raise ValueError(f"Unknown method '{method_key}'. Available: {list(REGISTRY)}")

    retriever = None
    if retriever_key := cfg.get("retriever"):
        if retriever_key not in RETRIEVER_REGISTRY:
            raise ValueError(f"Unknown retriever '{retriever_key}'. Available: {list(RETRIEVER_REGISTRY)}")
        retriever = RETRIEVER_REGISTRY[retriever_key](**cfg.get("retriever_params", {}))

    method_params = cfg.get("method_params", {})
    if retriever is not None:
        method_params = {**method_params, "retriever": retriever}

    method = REGISTRY[method_key](**method_params)

    run_experiment(
        method=method,
        languages=cfg["languages"],
        max_samples=cfg.get("max_samples"),
        num_runs=cfg.get("num_runs", 1),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
