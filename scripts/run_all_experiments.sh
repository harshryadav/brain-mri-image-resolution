#!/usr/bin/env bash
# Train E1..E5 in order, then write runs/results.csv.
# Any args you pass are forwarded to brainsr.cli.train, e.g.:
#   bash scripts/run_all_experiments.sh --override epochs=20 batch_size=16
set -euo pipefail

CONFIGS=(
  configs/e1_bicubic.yaml
  configs/e2_srcnn.yaml
  configs/e3_agunet_mse.yaml
  configs/e4_agunet_attn.yaml
  configs/e5_agunet_attn_dcgan.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running ${cfg} ==="
  python -m brainsr.cli.train --config "${cfg}" "$@"
done

python -m brainsr.cli.eval --runs-dir runs --output runs/results.csv

echo "Results written to runs/results.csv"
