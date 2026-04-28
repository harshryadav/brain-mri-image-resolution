# Running on UMD Zaratan (HPC)

This guide walks you from "I have a Zaratan account" to "all 5 experiments
finished on a GPU and I have `runs/results.csv` to drop into the report."

The dataset on disk is small once preprocessed (~1.7 GB for the full 6,620
slices we get from the 3 multicoil-test batches). You do **not** need to move
the 26 GB of raw `.h5` files to Zaratan -- preprocess once on your laptop and
sync only the processed cache.

---

## 0. Prerequisites

- A Zaratan account (request via <https://hpcc.umd.edu/>; you need an allocation
  for GPU partitions, e.g. through your advisor or class).
- SSH access:
  ```bash
  ssh <directory_id>@login.zaratan.umd.edu
  ```
- This repo cloned anywhere on Zaratan; `~/brain-mri-image-resolution` is fine.

## 1. Get the code on Zaratan

```bash
ssh <directory_id>@login.zaratan.umd.edu
git clone <your-fork-url> ~/brain-mri-image-resolution
cd ~/brain-mri-image-resolution
```

## 2. Build the Python env (one time, ~10 min)

```bash
bash scripts/zaratan/setup_env.sh
```

This loads a Python 3.10 module, creates `.venv/`, installs CUDA 12.1 PyTorch
wheels, and installs `brainsr` in editable mode. Re-run only if you change
dependencies.

## 3. Move the processed data from your laptop to Zaratan

You have already preprocessed 558 volumes on your Mac into `data/processed/`
(~1.7 GB, 6,620 `.npy` slices + `splits.json`). Send only that:

```bash
# Run this from your Mac, in the repo root.
rsync -avh --progress data/processed/ \
    <directory_id>@login.zaratan.umd.edu:~/brain-mri-image-resolution/data/processed/
```

Expected throughput on a typical home connection: 5-15 minutes.

### Alternative: rclone + Google Drive

If campus networking is in the way (or you'd rather have the data backed up in
the cloud), rclone works well. On your Mac:

```bash
brew install rclone
rclone config              # one-time interactive setup; pick "drive" backend
rclone copy data/processed gdrive:brain-mri/processed --progress
```

Then on Zaratan:

```bash
# Install rclone in your home dir (no sudo needed):
curl https://rclone.org/install.sh | sudo bash    # if sudo
# OR a static binary:
mkdir -p ~/bin && cd ~/bin && \
    curl -L -o rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip && \
    unzip rclone.zip && cp rclone-*-linux-amd64/rclone . && chmod +x rclone
export PATH=$HOME/bin:$PATH
rclone config              # paste the same credentials you used on Mac
rclone copy gdrive:brain-mri/processed ~/brain-mri-image-resolution/data/processed --progress
```

### Alternative: Globus

UMD operates a Globus endpoint (`umd#zaratan-public` or similar -- check
hpcc.umd.edu for the current name). Globus is the right tool if your dataset
is much bigger; for 1.7 GB it's overkill.

### What about scratch?

For just 1.7 GB it's fine to keep data in your home directory. If you grow the
dataset (e.g. preprocess `multicoil_train` later), put it under
`/scratch/zt1/<your-allocation>/users/<your-id>/` and update `data.root` in the
configs to point there.

## 4. Verify the env on a GPU node (interactive, 5 min)

```bash
sinteractive --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=8G
cd ~/brain-mri-image-resolution
source .venv/bin/activate
nvidia-smi          # should show an A100 / H100 / similar
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
make smoke          # runs SRCNN for 1 epoch on synthetic data
exit
```

## 5. Submit the experiments (batched)

```bash
# Single experiment:
sbatch scripts/zaratan/train.sbatch configs/e2_srcnn.yaml --override epochs=100

# All 5 in sequence + aggregate:
sbatch scripts/zaratan/run_all.sbatch
# Or override defaults without editing the YAML:
EPOCHS=80 BATCH=32 sbatch scripts/zaratan/run_all.sbatch
```

Check progress:

```bash
squeue --me                # job state
tail -f logs/brainsr-train-*.out   # live training log
```

When the job is done you'll have `runs/results.csv` plus per-experiment
`runs/eN_*/summary.json` and TensorBoard logs.

## 6. Pull results back to your Mac

```bash
# From your Mac:
rsync -avh \
    <directory_id>@login.zaratan.umd.edu:~/brain-mri-image-resolution/runs/ \
    runs/
tensorboard --logdir runs --port 6006
```

## 7. Tuning knobs

The `train.sbatch` and `run_all.sbatch` scripts default to:

- `--partition=gpu --gres=gpu:1`: change to your class's partition if you have
  one (often `--partition=class` or a lab partition).
- `--time=04:00:00`: bump up if you train past 100 epochs.
- `--cpus-per-task=8 --mem=32G`: 8 CPU workers feed the GPU comfortably.
- `BRAINSR_DEVICE=cuda`: forces CUDA so a misconfigured node fails loud
  instead of silently dropping to CPU.
- `mixed_precision=true`: enabled in `run_all.sbatch`. Roughly 2x throughput
  on A100/H100 with no metric loss for our model sizes.

## 8. Common gotchas

- **`Permission denied` on the GPU partition.** You need an allocation. Ask
  your advisor / instructor; or run on `--partition=standard` first to verify
  the CPU pipeline.
- **`CUDA out of memory` during E5.** Drop `batch_size` from 32 to 16 and/or
  `num_filters` for the AGUNet from 16 to 12 in
  `configs/e5_agunet_attn_dcgan.yaml`.
- **Job timing out.** Increase `--time=` and resubmit; we always save
  `last.pt` so you can resume if you wire up a `--resume` flag (not yet
  needed for this dataset size).
- **Module names differ.** Zaratan occasionally renames modules. If
  `module load python/3.10.10` fails, `module avail python` shows what's
  there.
