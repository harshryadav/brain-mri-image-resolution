# Brain MRI Super-Resolution (FastMRI)

MSML640 final project. Authors: **Harsh Yadav**, **Deepika Ghotra**, **Utkrisht Nath**.

We build a deep-learning super-resolution (SR) system for brain MRI on the
[NYU FastMRI](https://fastmri.med.nyu.edu/) dataset. The pipeline starts
from raw multicoil k-space (`.h5`), reconstructs magnitude images via IFFT
+ root-sum-of-squares (RSS), and trains a sequence of progressively richer
SR models against bicubic-downsampled inputs:

1. **Bicubic** - classical baseline, no training (E1).
2. **SRCNN** - 3-layer conv baseline (E2).
3. **AGUNet** - Attention-Gated U-Net adapted from
   [Li et al. 2022](https://doi.org/10.3389/fncom.2022.887633), with three
   ablations: plain (E3), with attention gates (E4), and with attention +
   DCGAN critic (E5).

## Project context (proposal)

> **Goal.** Clinical brain MRIs often trade resolution for acquisition time;
> the resulting blurriness can hide subtle anatomical detail. Build a SR
> system that takes low-resolution axial brain slices (synthetically
> downsampled 4x, e.g. 64x64) and reconstructs sharper 256x256 outputs that
> preserve the anatomy a clinician needs to make a diagnosis.

The proposal lays out a five-experiment ablation (E1..E5), proposes a
custom FastMRI ingest pipeline (k-space -> IFFT + RSS -> magnitude), and
reuses the AGUNet from Li et al. 2022 as the strongest model. Success
criteria from the proposal:

- PSNR >= bicubic + 2 dB on the held-out test set
- SSIM > 0.90
- Reference benchmark: Li et al. 2022 (PSNR 35.39, SSIM 0.985)

See `docs/proposal.md` if you want the full proposal text checked in
alongside the code.

## Results so far

20-epoch runs on the three combined `multicoil_test` batches
(558 volumes, 6,620 slices, 70/20/10 split, MPS):

| Exp | Model                          | Test PSNR | Δ vs E1 | Test SSIM | Test NRMSE |
| --- | ------------------------------ | --------- | ------- | --------- | ---------- |
| E1  | Bicubic                        | 30.286    | --      | 0.886     | 0.0816     |
| E2  | SRCNN                          | 32.830    | +2.54   | 0.911     | 0.0609     |
| E3  | AGUNet (MSE only)              | **34.327**| **+4.04**| **0.920**| **0.0513** |
| E4  | AGUNet + attention             | 34.205    | +3.92   | 0.919     | 0.0520     |
| E5  | AGUNet + attention + critic    | 34.116    | +3.83   | 0.918     | 0.0525     |

All proposal targets met. E3-E5 numbers are expected to shift with
longer training (the proposal/paper used 100 epochs); we plan to re-run on
GPU on UMD Zaratan (see [`scripts/zaratan/`](scripts/zaratan/README.md)).

## Quick start (one command, Docker)

This is the path your grader / professor / anyone-on-a-fresh-machine takes:

1. **Unzip** the submission (or `git clone` the repo) and `cd` into it.
2. **Install Docker Desktop** if you don't already have it
   (<https://docs.docker.com/get-docker/>). Make sure `docker compose version`
   prints a v2 string.
3. **(Optional)** `cp .env.example .env` and paste in your
   `PROCESSED_DATA_URL` (Google Drive shareable link to the preprocessed
   tarball). The shipped `.env.example` has placeholder URLs you can edit.
   `bash run.sh` will create `.env` from the example automatically if you
   skip this step.
4. **Run the demo:**

   ```bash
   bash run.sh
   ```

That single command:

- Builds the Docker image (first time only, ~5 minutes).
- Downloads the preprocessed FastMRI test cache from Google Drive into
  `data/processed/` (first time only, ~1.7 GB; cached locally afterwards).
- Loads each bundled checkpoint under `checkpoints/eN_*/`, evaluates it on
  the held-out test split, and writes:
  - `runs/results.csv` - PSNR / SSIM / NRMSE for every experiment.
  - `runs/<exp>/samples/*.png` - side-by-side **LR | SR | HR** panels.
  - `runs/<exp>/summary.json` - per-experiment metrics.

CPU works out of the box. For GPU, uncomment the `deploy.resources` block
in [`docker-compose.yml`](docker-compose.yml) (requires NVIDIA Container
Toolkit on the host).

### Other one-shot commands

`run.sh` is a thin dispatcher; everything still runs inside the same
container.

```bash
bash run.sh demo              # default; same as `bash run.sh`
bash run.sh train             # train E2 (SRCNN) by default
bash run.sh train configs/e5_agunet_attn_dcgan.yaml
bash run.sh train-all         # train E1..E5 then aggregate to runs/results.csv
bash run.sh preprocess        # raw .h5 -> data/processed/ (needs FASTMRI_DIR set)
bash run.sh smoke             # synthetic phantom sanity check, no downloads
bash run.sh shell             # interactive shell in the container
bash run.sh tensorboard       # http://localhost:6006
bash run.sh test              # pytest
```

Force a rebuild with `BRAINSR_REBUILD=1 bash run.sh`.

### Native (no Docker) quick start

If you'd rather use a venv:

```bash
make install
make demo               # same flow as `bash run.sh`, but on the host Python
# or for the no-data-needed phantom sanity check:
make smoke
make tb                 # http://localhost:6006
```

## Project layout

```
brain-mri-image-resolution/
  run.sh           # one-command entry point (demo / train / preprocess / smoke)
  Dockerfile docker-compose.yml Makefile pyproject.toml requirements.txt
  configs/         # YAML, one per experiment + base.yaml
  checkpoints/     # bundled best.pt + config.resolved.yaml per experiment (committed)
  src/brainsr/     # installable package (brainsr-{preprocess,train,eval,predict,demo})
    data/          # fastmri_convert, degradation, dataset, splits
    models/        # bicubic, srcnn, agunet (+ attention gates), dcgan_critic
    cli/           # console-script entry points
    losses.py metrics.py trainer.py utils/
  scripts/         # download_{data,model}.py, run_all_experiments.sh, Zaratan SLURM jobs
  tests/           # pytest suite (uses synthetic phantom slices)
  data/sample/     # tiny synthetic dataset for `make smoke` (committed)
  data/{raw,processed}/  # gitignored - your FastMRI .h5 + the .npy cache
  runs/            # gitignored - per-experiment TB logs, checkpoints, samples
```

## Re-training from scratch (optional)

The bundled checkpoints in `checkpoints/` already give you publishable
metrics via `bash run.sh`. The rest of this section is only for users who
want to re-train the models themselves (e.g. to extend the experiment grid
or push past 100 epochs).

1. **Get the data.** Follow [`scripts/download_fastmri_help.md`](scripts/download_fastmri_help.md).
   FastMRI is not redistributable, so this repo cannot ship raw data.

   > **Caveat for the public `multicoil_test` batches.** These are 8x
   > **undersampled** (you'll see `acceleration: 8` and a `mask` attribute
   > inside the `.h5`) with **no fully-sampled ground truth**. The pipeline
   > still works: we IFFT + RSS the masked k-space and treat that
   > zero-filled reconstruction as our "HR" target, then synthetically
   > degrade it (Gaussian blur + bicubic down) to make LR. The SR
   > experiment is well defined and the metrics are meaningful, but absolute
   > PSNR/SSIM aren't directly comparable to papers that train on
   > fully-sampled `multicoil_train`.

2. **Point the project at it.** For a single directory:

   ```bash
   cp .env.example .env
   # edit FASTMRI_DIR=/abs/path/to/multicoil_train
   ```

   For multiple directories (e.g. several test batches untarred side by
   side), use `FASTMRI_DIRS` (space-separated):

   ```bash
   export FASTMRI_DIRS="data/multicoil_test data/multicoil_test\ 2 data/multicoil_test\ 3"
   ```

3. **Preprocess once.** Converts every `.h5` volume to per-slice `.npy`
   magnitude images and writes a deterministic 70/20/10 split:

   ```bash
   make preprocess                                       # native
   make docker-preprocess                                # inside Docker
   make preprocess ARGS="--acquisition AXT2 --limit 100" # forward extra flags
   ```

   Useful flags:

   - `--acquisition AXT2,AXFLAIR` - subset to one or more contrasts (FastMRI
     brain has `AXT1`, `AXT1PRE`, `AXT1POST`, `AXT2`, `AXFLAIR`).
   - `--limit N` - cap to the first N volumes (handy for quick iteration).
   - `--target-size 256` - default; readout dim is center-cropped to a
     square FOV first, then bicubic-resized.

4. **Train an experiment.** Each YAML maps to one row of the experiment plan:

   | Config                              | Experiment                              |
   | ----------------------------------- | --------------------------------------- |
   | `e1_bicubic.yaml`                   | E1 - bicubic baseline (no training)     |
   | `e2_srcnn.yaml`                     | E2 - SRCNN, MSE loss                    |
   | `e3_agunet_mse.yaml`                | E3 - AGUNet w/o attention, MSE          |
   | `e4_agunet_attn.yaml`               | E4 - AGUNet + attention gates, MSE      |
   | `e5_agunet_attn_dcgan.yaml`         | E5 - AGUNet + attention + DCGAN critic  |

   ```bash
   make train-e2                                         # native
   make docker-train CONFIG=configs/e4_agunet_attn.yaml  # Docker
   ```

   Override anything from the CLI without editing YAML:

   ```bash
   python -m brainsr.cli.train --config configs/e3_agunet_mse.yaml \
       --override epochs=20 batch_size=8 data.scale=2
   ```

5. **Run all experiments and aggregate metrics.**

   ```bash
   make run-all     # trains E1..E5 then writes runs/results.csv
   make eval        # re-aggregate any time
   ```

## Speeding things up

### Apple Silicon (Mac)

The trainer auto-detects MPS, so on an M-series Mac you should see roughly
**5-15x** speedup vs CPU with no config changes -- check that the first log
line says `Device: mps`. If it falls back to CPU, force it:

```bash
export BRAINSR_DEVICE=mps
python -m brainsr.cli.train --config configs/e2_srcnn.yaml --override epochs=20
```

A few more knobs:

- `--override num_workers=4` - more dataloader workers
- `--override batch_size=32` - higher batch size if memory allows
- Mixed precision (`autocast` + `GradScaler`) is intentionally disabled on
  MPS; PyTorch 2.4-2.5 fp16 support there is still incomplete.

### UMD Zaratan (HPC, recommended for E3-E5)

The full pipeline runs on Zaratan with one `sbatch`. SLURM job files,
setup script, and a step-by-step playbook live in
[`scripts/zaratan/README.md`](scripts/zaratan/README.md).

Headline workflow:

```bash
# On Zaratan (one-time):
git clone <this repo> ~/brain-mri-image-resolution
cd ~/brain-mri-image-resolution
bash scripts/zaratan/setup_env.sh

# On your Mac (one-time, ~1.7 GB; raw .h5 stay local):
rsync -avh data/processed/ <id>@login.zaratan.umd.edu:~/brain-mri-image-resolution/data/processed/

# On Zaratan: run all 5 experiments on a GPU node
sbatch scripts/zaratan/run_all.sbatch
```

Per-epoch timing from this codebase:

| Experiment             | Mac CPU       | Mac MPS    | Zaratan A100 |
| ---------------------- | ------------- | ---------- | ------------ |
| E2 SRCNN               | ~13 min       | ~45 s      | ~30 s        |
| E3-E4 AGUNet           | ~30+ min      | ~55 s      | ~60 s        |
| E5 AGUNet + critic     | ~50+ min      | ~75 s      | ~90 s        |

100-epoch E5 goes from "literally days on CPU" to ~2.5 hours on a single A100.

## Docker

The image is based on `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`. It
runs CPU-only out of the box; uncomment the `deploy.resources.reservations.devices`
block in [`docker-compose.yml`](docker-compose.yml) for GPU (requires the
NVIDIA Container Toolkit).

The friendly entry point is [`run.sh`](run.sh) (see *Quick start* above).
Underneath, it just calls these standard `docker compose` services:

| Service       | What it runs                                        |
| ------------- | --------------------------------------------------- |
| `demo`        | `python -m brainsr.cli.demo` (the default)          |
| `preprocess`  | `python -m brainsr.cli.preprocess`                  |
| `train`       | `python -m brainsr.cli.train --config ...`          |
| `dev`         | interactive shell with the source bind-mounted      |
| `tensorboard` | TB on port 6006                                     |

If you'd rather call them directly:

```bash
make docker-build
make docker-demo                         # the one-command path
make docker-shell                        # interactive shell
make docker-preprocess                   # one-shot preprocessing
make docker-train CONFIG=configs/e2_srcnn.yaml
docker compose up tensorboard            # http://localhost:6006
```

`docker-compose.yml` bind-mounts `${PROCESSED_DIR}` (the .npy cache),
`${RUNS_DIR}`, and `./checkpoints`, so all artifacts persist on the host
across container runs. The `dev` and `preprocess` services additionally
bind-mount `${FASTMRI_DIR}` read-only.

### Where the bundled checkpoints come from

`checkpoints/eN_*/` ships in the repo (and zip submission). Each folder
holds a `best.pt` plus the `config.resolved.yaml` it was trained with, so
the demo CLI can rebuild the exact model architecture and load weights
without any guesswork. See [`checkpoints/README.md`](checkpoints/README.md)
for the layout and how to refresh them after re-training.

If `checkpoints/` is missing entirely (e.g. you cloned a stripped-down
branch), set `CHECKPOINTS_URL` in `.env` and the demo will pull a tarball
on first run.

### Hosting the preprocessed dataset (one-time, by you)

The 1.7 GB preprocessed cache is too large to bundle in the zip and (per
FastMRI's data-use agreement) can't be redistributed alongside the code.
The demo expects to fetch it on first run from a URL you set in
`PROCESSED_DATA_URL`. Pick whichever host you prefer:

**Hugging Face Hub Datasets (recommended).** Free, no quota, fast,
exactly what your professor recommended for ML artifacts. ~10 minutes
one-time:

```bash
# 1. Create the tarball (run on whichever machine has data/processed/ ready)
tar czf processed.tar.gz -C data processed/

# 2. Install the CLI and log in (uses a free HF account + access token)
pip install huggingface_hub
huggingface-cli login   # paste a write-token from https://huggingface.co/settings/tokens

# 3. Create a public dataset repo and upload the tarball
huggingface-cli repo create brain-mri-processed --type dataset
huggingface-cli upload <your-username>/brain-mri-processed processed.tar.gz \
    processed.tar.gz --repo-type dataset

# 4. Set the URL in .env.example so it ships with the submission
#    (note: /resolve/main/ is the raw-bytes endpoint; /blob/main/ also works
#    and is auto-rewritten by the downloader)
PROCESSED_DATA_URL=https://huggingface.co/datasets/<your-username>/brain-mri-processed/resolve/main/processed.tar.gz
```

**GitHub Releases (alternative).** Easiest if you already use GitHub. The
2 GB per-file limit fits your tarball with margin.

1. `tar czf processed.tar.gz -C data processed/`
2. On your repo's GitHub page: *Releases* -> *Draft a new release* ->
   tag it `v1.0-data` -> drag-and-drop `processed.tar.gz` into the
   "Assets" panel -> *Publish release*.
3. Right-click the asset link in the published release page -> *Copy link*.
4. `PROCESSED_DATA_URL=https://github.com/<user>/<repo>/releases/download/v1.0-data/processed.tar.gz`.

**Why not Google Drive for the file?** For files >100 MB, Drive shows a
"can't scan for viruses" confirmation page that breaks `gdown` and is
rate-limited per file. The downloader will refuse a Drive file URL and
print this advice. Drive *folder* URLs still work as a slow fallback.

## Architecture overview

```
.h5 (k-space) -> IFFT per coil + RSS -> 256x256 center-cropped magnitude
                                           |
                                           +-> data/processed/*.npy + splits.json
                                                          |
                                MRISliceDataset reads HR, applies on-the-fly
                                Gaussian blur + bicubic downsample to make LR
                                                          |
                                            +-------------+-------------+
                                            v                           v
                                       Generator                   (optional) DCGAN
                               (bicubic / SRCNN / AGUNet)               critic
                                            |                           |
                                            +-------- trainer ----------+
                                                          |
                                                runs/<exp>/{tb, *.pt,
                                                            summary.json,
                                                            samples/}
                                                          |
                                              brainsr-eval -> results.csv
```

For a step-by-step technical walkthrough (per-stage data shapes, design
rationale for each choice, comparison to the Li et al. 2022 reference
implementation), see the longer chat history that produced this repo or
the in-source docstrings - everything important is documented in place.

## Evaluation metrics

PSNR / SSIM / NRMSE are computed both during training (torchmetrics,
batched on whatever device we're on) and offline against the held-out test
split. Targets per the proposal:

- PSNR >= bicubic + 2 dB
- SSIM > 0.90
- Reference benchmark: Li et al. 2022 (PSNR 35.39, SSIM 0.985)

## Development

```bash
make test       # pytest (uses synthetic phantom data, no FastMRI needed)
make lint       # ruff
```

## Acknowledgements / references

- Li, B. M. et al. (2022). *Deep attention super-resolution of brain MRI
  acquired under clinical protocols.* Frontiers in Computational Neuroscience.
  <https://doi.org/10.3389/fncom.2022.887633> (architecture reference,
  AGUNet + DCGAN critic)
- Zbontar, J. et al. (2018). *fastMRI: An open dataset and benchmarks for
  accelerated MRI.* NYU FastMRI initiative.
- Dong, C. et al. (2014). *Image super-resolution using deep convolutional
  networks.* (SRCNN baseline)
- Oktay, O. et al. (2018). *Attention U-Net: Learning where to look for the
  pancreas.* (attention gate module)

## License

MIT. See [`LICENSE`](LICENSE).
