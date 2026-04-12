#!/bin/bash

for config in experiments/*.yaml; do
    [ "$(basename "$config")" = "example.yaml" ] && continue
    echo "Running: $config"
    python run_experiment.py "$config"
done
