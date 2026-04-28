#!/bin/bash
#SBATCH --job-name=brainsr-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Zaratan single-experiment GPU training job.
#
# Usage (from the repo root):
#   sbatch scripts/zaratan/train.sbatch configs/e2_srcnn.yaml
#   sbatch scripts/zaratan/train.sbatch configs/e5_agunet_attn_dcgan.yaml \
#          --override epochs=100 batch_size=32
#
# A single A100 should chew through SRCNN/AGUNet at ~30-90s per epoch on this
# dataset. If your group has a partition (e.g. --partition=class or a lab
# allocation), set it via the SBATCH_PARTITION env var or edit above.

set -euo pipefail
mkdir -p logs runs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load python/3.10.10 || module load python
module load cuda/12.1 || module load cuda

# shellcheck disable=SC1091
source .venv/bin/activate

# Force CUDA so we never silently fall back to CPU on a bad node.
export BRAINSR_DEVICE=cuda
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/e2_srcnn.yaml}"
shift || true

echo "[$(date)] Running on $(hostname); GPU info:"
nvidia-smi || true

python -m brainsr.cli.train --config "$CONFIG" "$@"

echo "[$(date)] Done."
