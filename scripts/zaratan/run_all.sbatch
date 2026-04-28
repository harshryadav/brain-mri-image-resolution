#!/bin/bash
#SBATCH --job-name=brainsr-runall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run E1 -> E5 sequentially, then aggregate to runs/results.csv.
# Submit with:  sbatch scripts/zaratan/run_all.sbatch
#
# Edit EPOCHS / BATCH below if you want to override the YAML defaults
# without editing the configs themselves.

set -euo pipefail
mkdir -p logs runs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load python/3.10.10 || module load python
module load cuda/12.1 || module load cuda
# shellcheck disable=SC1091
source .venv/bin/activate

export BRAINSR_DEVICE=cuda
export PYTHONUNBUFFERED=1

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
COMMON_OVERRIDES=(--override "epochs=${EPOCHS}" "batch_size=${BATCH}" "num_workers=4" "mixed_precision=true")

echo "[$(date)] Node: $(hostname)"
nvidia-smi || true

python -m brainsr.cli.train --config configs/e1_bicubic.yaml
python -m brainsr.cli.train --config configs/e2_srcnn.yaml          "${COMMON_OVERRIDES[@]}"
python -m brainsr.cli.train --config configs/e3_agunet_mse.yaml     "${COMMON_OVERRIDES[@]}"
python -m brainsr.cli.train --config configs/e4_agunet_attn.yaml    "${COMMON_OVERRIDES[@]}"
python -m brainsr.cli.train --config configs/e5_agunet_attn_dcgan.yaml "${COMMON_OVERRIDES[@]}"

python -m brainsr.cli.eval --runs-dir runs --output runs/results.csv
echo "[$(date)] Done. See runs/results.csv"
