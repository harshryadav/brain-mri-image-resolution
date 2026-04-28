#!/bin/bash
#SBATCH --job-name=brainsr-preprocess
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Zaratan preprocessing job. Only needed if you re-download FastMRI on the
# cluster instead of rsync'ing data/processed/ from your laptop.
#
# Submit from the repo root with:
#   FASTMRI_RAW=/path/to/raw sbatch scripts/zaratan/preprocess.sbatch

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load python/3.10.10 || module load python

# shellcheck disable=SC1091
source .venv/bin/activate

# Allow the caller to set one or many input dirs via FASTMRI_DIRS or FASTMRI_RAW.
DIRS="${FASTMRI_DIRS:-${FASTMRI_RAW:-data/multicoil_test}}"
flags=""
for d in $DIRS; do flags="$flags --input-dir $d"; done

# shellcheck disable=SC2086
python -m brainsr.cli.preprocess $flags --output-dir data/processed "${@}"
