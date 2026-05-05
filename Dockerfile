# Pinned PyTorch + CUDA base for reproducibility. Runs CPU-only out of the
# box (just omit --gpus); for GPU, see the deploy block in docker-compose.yml.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1 + libglib are pulled in by matplotlib / PIL for headless rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install deps first so they get cached separately from source code edits.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY checkpoints ./checkpoints

# Mount points the compose services bind to host paths.
RUN mkdir -p /data/raw \
    /workspace/data/processed \
    /workspace/runs

# Default: run the inference demo. Other compose services override this.
ENTRYPOINT ["python", "-m", "brainsr.cli.demo"]
CMD []
