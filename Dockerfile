# Use the official PyTorch CUDA runtime image. It works on CPU as well: just
# omit `--gpus` when running. Pinned tag for reproducibility.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY checkpoints ./checkpoints
COPY data/sample ./data/sample
COPY Makefile ./Makefile

# Create the runtime mount points that docker-compose binds to host folders.
RUN mkdir -p /data/raw \
    /workspace/data/processed \
    /workspace/runs

# Default behavior: run the inference demo end-to-end. Compose services
# override this for the train/preprocess/dev paths.
ENTRYPOINT ["python", "-m", "brainsr.cli.demo"]
CMD []
