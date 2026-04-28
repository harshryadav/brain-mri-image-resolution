#!/bin/bash
# scripts/zaratan/setup_env.sh
# One-time environment setup on UMD Zaratan. Run from the repo root:
#   bash scripts/zaratan/setup_env.sh
#
# Creates a Python 3.10 venv at .venv/ with all dependencies (incl. CUDA torch).
# Run on a compute node via:
#   sinteractive --partition=standard --time=00:30:00 --mem=8G
# or just run on the login node (it's a small install).

set -euo pipefail

# Pick a Python module. Zaratan ships several; 3.10 matches our CI.
module purge
module load python/3.10.10  || module load python/3.10  || module load python

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# CUDA wheels for torch. Zaratan's GPU nodes run CUDA 12.x; cu121 wheels work.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.4,<2.6" "torchvision>=0.19,<0.21"

# The rest of the project. -e installs in editable mode so your code edits are
# picked up without reinstall.
python -m pip install -e ".[dev]"

echo
echo "Environment ready. Activate with:"
echo "  source $REPO_ROOT/.venv/bin/activate"
