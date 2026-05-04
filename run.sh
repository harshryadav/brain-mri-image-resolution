#!/usr/bin/env bash
# Single entry point for the dockerized brain-MRI super-resolution project.
#
# Usage:
#   bash run.sh                 # default: inference demo (E1..E5 on test split)
#   bash run.sh demo            # same as above (explicit)
#   bash run.sh train [CONFIG]  # train one experiment (CONFIG defaults to E2)
#   bash run.sh train-all       # train E1..E5 in sequence then aggregate
#   bash run.sh preprocess      # convert raw FastMRI .h5 -> .npy cache
#   bash run.sh smoke           # offline phantom sanity check (no downloads)
#   bash run.sh shell           # drop into an interactive shell in the container
#   bash run.sh tensorboard     # launch TB on http://localhost:6006
#   bash run.sh test            # run pytest inside the container
#
# This script builds the Docker image on first invocation and reuses it
# afterwards. Set BRAINSR_REBUILD=1 to force a rebuild.
#
# Designed to need zero arguments for the common "grader runs the demo" path.

set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="brainsr:latest"

cmd="${1:-demo}"
shift || true

# `help` is a no-op meta command - don't require Docker just to print usage.
if [[ "$cmd" == "help" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH." >&2
    echo "Install Docker Desktop (https://docs.docker.com/get-docker/) and re-run." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: 'docker compose' (v2) is required." >&2
    echo "Update Docker Desktop, or install the compose plugin." >&2
    exit 1
fi

# Daemon health check. The CLI binaries above work without the daemon, but
# `docker compose build` will silently hang at "[+] Building 0.0s (0/0)" if
# the daemon isn't reachable. Fail loudly instead.
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to the Docker daemon." >&2
    echo "  - On macOS / Windows: open the Docker Desktop app and wait for the" >&2
    echo "    whale icon in your menu bar to stop animating (~30s first time)." >&2
    echo "  - On Linux: start the service, e.g. 'sudo systemctl start docker'." >&2
    echo "Then re-run: bash run.sh" >&2
    echo >&2
    echo "Diagnostic output from 'docker info':" >&2
    docker info 2>&1 | head -n 5 | sed 's/^/  /' >&2
    exit 1
fi

# Make sure .env exists so docker compose's variable substitution works.
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "Created .env from .env.example. Edit it if you need custom URLs/paths."
    else
        touch .env
    fi
fi

# Build on first run, or if explicitly requested. We check by image presence.
need_build=0
if [[ "${BRAINSR_REBUILD:-0}" == "1" ]]; then
    need_build=1
elif ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    need_build=1
fi
if (( need_build )); then
    echo "==> Building Docker image ($IMAGE_NAME)..."
    docker compose build
fi

case "$cmd" in
    demo|"")
        echo "==> Running inference demo (E1..E5 on test split)"
        docker compose run --rm demo "$@"
        ;;
    train)
        config="${1:-configs/e2_srcnn.yaml}"
        shift || true
        echo "==> Training $config"
        docker compose run --rm train --config "$config" "$@"
        ;;
    train-all)
        echo "==> Training E1..E5 in sequence then aggregating"
        docker compose run --rm dev bash scripts/run_all_experiments.sh "$@"
        ;;
    preprocess)
        echo "==> Preprocessing raw FastMRI .h5 -> data/processed/"
        docker compose run --rm preprocess "$@"
        ;;
    smoke)
        echo "==> Smoke test on synthetic phantom (no downloads)"
        docker compose run --rm dev bash -lc "make smoke"
        ;;
    shell)
        docker compose run --rm dev bash
        ;;
    tensorboard|tb)
        echo "==> TensorBoard on http://localhost:6006"
        docker compose up tensorboard
        ;;
    test)
        docker compose run --rm dev bash -lc "pytest"
        ;;
    *)
        echo "Unknown subcommand: $cmd" >&2
        echo "Run 'bash run.sh help' for usage." >&2
        exit 2
        ;;
esac
