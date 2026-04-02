"""CLI entrypoint for running experiments.

Usage:
    python run_experiment.py experiments/example.yaml
"""

import sys
import yaml

from src.mas_code_sum.runner import run_experiment
from src.mas_code_sum.methods import REGISTRY


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    method_key = cfg["method"]
    if method_key not in REGISTRY:
        raise ValueError(f"Unknown method '{method_key}'. Available: {list(REGISTRY)}")

    method = REGISTRY[method_key](**cfg.get("method_params", {}))

    run_experiment(
        method=method,
        languages=cfg["languages"],
        split=cfg.get("split", "test"),
        max_samples=cfg.get("max_samples"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
