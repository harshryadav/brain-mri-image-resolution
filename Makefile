.PHONY: help install preprocess preprocess-sample \
        train-e1 train-e2 train-e3 train-e4 train-e5 \
        eval run-all smoke test lint \
        docker-build docker-shell docker-preprocess docker-train tb clean

# Python executable (requires >= 3.10). Search PATH for python3.10 or python3.
# If not found, you can override: make PYTHON=python3.10 ... or set it in your shell
PYTHON_SEARCH := python3.10 python3
PYTHON := $(firstword $(shell $(foreach py,$(PYTHON_SEARCH),which $(py) 2>/dev/null ||) true))
CONFIG ?= configs/e2_srcnn.yaml

ifeq ($(PYTHON),)
$(error Python 3.10+ not found in PATH. Please install Python 3.10+ or set PYTHON=/path/to/python)
endif

help:
	@echo "Common targets:"
	@echo "  install          pip install -e '.[torch,dev]'"
	@echo "  preprocess       Convert FastMRI .h5 -> cached .npy (uses FASTMRI_DIR)"
	@echo "  preprocess-sample  Build a tiny synthetic sample under data/sample/"
	@echo "  train-e1..e5     Train each experiment via configs/eN_*.yaml"
	@echo "  eval             Compute PSNR/SSIM/NRMSE for all runs into runs/results.csv"
	@echo "  run-all          Train E1..E5 sequentially then eval"
	@echo "  smoke            1-epoch SRCNN run on data/sample/ (no FastMRI needed)"
	@echo "  test             pytest"
	@echo "  lint             ruff check"
	@echo "  docker-build     Build the Docker image"
	@echo "  docker-shell     Interactive shell in the dev container"
	@echo "  tb               Launch TensorBoard on runs/ (port 6006)"

venv:
	python3 -m venv .venv
	source .venv/bin/activate && pip install -e '.[torch,dev]'

install:
	$(PYTHON) -m pip install -e '.[torch,dev]'

## preprocess: converts every .h5 under the directories listed in FASTMRI_DIRS
## (whitespace-separated) or, if unset, the single FASTMRI_DIR. Forwards extra
## flags via ARGS, e.g.: make preprocess ARGS="--acquisition AXT2 --limit 50"
preprocess:
	@dirs="$${FASTMRI_DIRS:-$$FASTMRI_DIR}"; \
	if [ -z "$$dirs" ]; then \
		echo "Set FASTMRI_DIRS (space-separated) or FASTMRI_DIR in .env or your shell."; \
		exit 1; \
	fi; \
	flags=""; for d in $$dirs; do flags="$$flags --input-dir $$d"; done; \
	$(PYTHON) -m brainsr.cli.preprocess $$flags --output-dir data/processed $(ARGS)

preprocess-sample:
	$(PYTHON) -m brainsr.cli.preprocess --build-sample --output-dir data/sample

train-e1:
	$(PYTHON) -m brainsr.cli.train --config configs/e1_bicubic.yaml
train-e2:
	$(PYTHON) -m brainsr.cli.train --config configs/e2_srcnn.yaml
train-e3:
	$(PYTHON) -m brainsr.cli.train --config configs/e3_agunet_mse.yaml
train-e4:
	$(PYTHON) -m brainsr.cli.train --config configs/e4_agunet_attn.yaml
train-e5:
	$(PYTHON) -m brainsr.cli.train --config configs/e5_agunet_attn_dcgan.yaml

eval:
	$(PYTHON) -m brainsr.cli.eval --runs-dir runs --output runs/results.csv

run-all:
	bash scripts/run_all_experiments.sh

smoke:
	$(PYTHON) -m brainsr.cli.preprocess --build-sample --output-dir data/sample
	$(PYTHON) -m brainsr.cli.train --config configs/e2_srcnn.yaml \
		--override data.root=data/sample epochs=1 batch_size=2 output_dir=runs/_smoke

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests

docker-build:
	docker compose build

docker-shell:
	docker compose run --rm dev bash

docker-preprocess:
	docker compose run --rm preprocess

docker-train:
	docker compose run --rm train --config $(CONFIG)

tb:
	tensorboard --logdir runs --port 6006 --bind_all

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +
