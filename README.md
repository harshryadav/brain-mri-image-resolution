# Brain MRI Super-Resolution (FastMRI)

MSML640 final project. Authors: **Harsh Yadav**, **Deepika Ghotra**, **Utkrisht Nath**.

A deep-learning super-resolution (SR) system for brain MRI on the
[NYU FastMRI](https://fastmri.med.nyu.edu/) dataset. The pipeline starts from
raw multicoil k-space (`.h5`), reconstructs magnitude images via per-coil IFFT
+ root-sum-of-squares (RSS), and trains a sequence of progressively richer SR
models against bicubic-downsampled inputs:

1. **Bicubic** — classical baseline, no training (E1).
2. **SRCNN** — 3-layer conv baseline, Dong et al. 2014 (E2).
3. **AGUNet** — Attention-Gated U-Net adapted from
   [Li et al. 2022](https://doi.org/10.3389/fncom.2022.887633), with three
   ablations: plain (E3), with attention gates (E4), and with attention +
   DCGAN critic (E5).

## Project context

> **Goal.** Clinical brain MRIs trade resolution for acquisition time; the
> resulting blurriness can hide subtle anatomical detail. We build an SR
> system that takes a low-resolution axial brain slice (synthetically
> 4×-downsampled, e.g. 64×64) and reconstructs a sharper 256×256 output
> that preserves the anatomy a clinician needs to make a diagnosis.

The proposal lays out the five-experiment ablation (E1..E5), a custom FastMRI
ingest pipeline, and reuses the AGUNet from Li et al. 2022 as the strongest
model. Success criteria:

- **PSNR ≥ bicubic + 2 dB** on the held-out test set
- **SSIM > 0.90**
- Reference benchmark: Li et al. 2022 (PSNR 35.39, SSIM 0.985)

The full proposal is checked in at [`docs/proposal.md`](docs/proposal.md).

## Results

100-epoch runs on `brain_multicoil_train_batch_0` (5,397 slices, 70/20/10
volume-level split, 548 test slices, 4× super-resolution, A100 GPU on UMD
Zaratan):

| Exp | Model                        | Test PSNR | Δ vs E1 | Test SSIM | Test NRMSE |
| --- | ---------------------------- | --------- | ------- | --------- | ---------- |
| E1  | Bicubic                      | 25.209    | —       | 0.842     | 0.1648     |
| E2  | SRCNN                        | 27.694    | +2.49   | 0.878     | 0.1238     |
| E3  | AGUNet (MSE only)            | 29.673    | +4.46   | 0.897     | 0.0986     |
| E4  | **AGUNet + attention**       | **29.755**| **+4.55**| **0.899**| **0.0977** |
| E5  | AGUNet + attention + critic  | 28.368    | +3.16   | 0.871     | 0.1146     |

E4 is the headline model. The PSNR target is met by all four trained models;
SSIM lands within 0.001 of the 0.90 threshold (rounds to 0.90 at the
precision typically reported). E5's drop is the classic
perception–distortion trade-off — adversarial losses optimize visual
sharpness at the expense of pixel-level metrics.

## Quick start (one command, Docker)

This is the path graders take on a fresh machine:

1. **Unzip** the submission (or `git clone` the repo) and `cd` into it.
2. **Install Docker Desktop** if you don't already have it
   (<https://docs.docker.com/get-docker/>). Confirm `docker compose version`
   prints a v2 string.
3. **(Optional)** `cp .env.example .env` if you want to override the data
   URL or paths. `bash run.sh` does this automatically if you skip the step.
4. **Run the demo:**

   ```bash
   bash run.sh
   ```

That single command:

- Builds the Docker image (~5 minutes the first time, cached afterwards).
- Downloads the preprocessed FastMRI cache from Hugging Face into
  `data/processed/` (~1.7 GB, first time only).
- Loads each bundled checkpoint under `checkpoints/eN_*/`, evaluates on the
  held-out test split, and writes:
  - `runs/results.csv` — PSNR / SSIM / NRMSE per experiment
  - `runs/<exp>/samples/*.png` — side-by-side **LR | SR | HR** panels
  - `runs/<exp>/summary.json` — per-experiment metrics

CPU works out of the box. For GPU, uncomment the `deploy.resources` block in
[`docker-compose.yml`](docker-compose.yml) (requires NVIDIA Container Toolkit
on the host).

### Other one-shot commands

```bash
bash run.sh demo              # default; same as `bash run.sh`
bash run.sh train             # train E2 (SRCNN)
bash run.sh train configs/e5_agunet_attn_dcgan.yaml
bash run.sh train-all         # train E1..E5, then aggregate results.csv
bash run.sh preprocess        # raw .h5 -> data/processed/ (needs FASTMRI_DIR)
bash run.sh smoke             # synthetic phantom sanity check, no downloads
bash run.sh shell             # interactive shell in the container
bash run.sh tensorboard       # TensorBoard on http://localhost:6006
bash run.sh test              # pytest
```

Force a rebuild with `BRAINSR_REBUILD=1 bash run.sh`.

## Project layout

```
brain-mri-image-resolution/
  run.sh                      # one-command entry point (demo / train / ...)
  Dockerfile docker-compose.yml pyproject.toml requirements.txt
  configs/                    # one YAML per experiment + base.yaml
  checkpoints/                # bundled best.pt + config.resolved.yaml per exp
  src/brainsr/                # installable package (brainsr-{preprocess,train,eval,demo})
    data/                       # fastmri_convert, degradation, dataset, splits
    models/                     # bicubic, srcnn, agunet, attention_gates, dcgan_critic
    cli/                        # console-script entry points
    losses.py metrics.py trainer.py utils/
  scripts/                    # download_data.py, run_all_experiments.sh, Zaratan SLURM
  tests/                      # pytest suite (synthetic phantom data)
  data/sample/                # tiny synthetic dataset for smoke tests
  data/{raw,processed}/       # gitignored: your FastMRI .h5 + the .npy cache
  runs/                       # gitignored: per-experiment TB logs, checkpoints, samples
```

## Re-training from scratch (optional)

The bundled checkpoints already give you the metrics in the table above via
`bash run.sh`. The rest of this section is only for reproducing or extending
those runs.

1. **Get the data.** Follow [`scripts/download_fastmri_help.md`](scripts/download_fastmri_help.md).
   FastMRI is not redistributable, so this repo can't ship raw data.

   > **Caveat for the public `multicoil_test` batches.** They're 8×
   > undersampled with no fully-sampled ground truth. The pipeline still
   > works (we IFFT + RSS the masked k-space and treat the zero-filled
   > reconstruction as our HR target), but absolute PSNR/SSIM aren't
   > directly comparable to papers that train on `multicoil_train`. The
   > shipped checkpoints are trained on the fully-sampled
   > `multicoil_train` partition.

2. **Point the project at the raw data:**

   ```bash
   cp .env.example .env
   # edit FASTMRI_DIR=/abs/path/to/multicoil_train
   ```

   For multiple directories use `FASTMRI_DIRS` (space-separated), e.g.
   several test batches untarred side by side.

3. **Preprocess once.** Converts every `.h5` volume to per-slice `.npy`
   magnitude images and writes a deterministic 70/20/10 split:

   ```bash
   bash run.sh preprocess
   # forward extra flags via the trailing args, e.g.:
   bash run.sh preprocess --acquisition AXT2 --limit 100
   ```

   Useful flags: `--acquisition AXT2,AXFLAIR` (subset by FastMRI brain
   contrast), `--limit N`, `--target-size 256`.

4. **Train an experiment.** Each YAML maps to one row of the ablation:

   | Config                              | Experiment                              |
   | ----------------------------------- | --------------------------------------- |
   | `e1_bicubic.yaml`                   | E1 — bicubic baseline (no training)     |
   | `e2_srcnn.yaml`                     | E2 — SRCNN, MSE                         |
   | `e3_agunet_mse.yaml`                | E3 — AGUNet w/o attention, MSE          |
   | `e4_agunet_attn.yaml`               | E4 — AGUNet + attention gates, MSE      |
   | `e5_agunet_attn_dcgan.yaml`         | E5 — AGUNet + attention + DCGAN critic  |

   ```bash
   bash run.sh train configs/e4_agunet_attn.yaml
   ```

   Override anything from the CLI without editing YAML:

   ```bash
   docker compose run --rm train --config configs/e3_agunet_mse.yaml \
       --override epochs=20 batch_size=8 data.scale=2
   ```

5. **Run all experiments and aggregate metrics:**

   ```bash
   bash run.sh train-all   # trains E1..E5, then writes runs/results.csv
   ```

## Speeding things up

### Apple Silicon (Mac)

The trainer auto-detects MPS — on an M-series Mac you should see roughly
**5–15× speedup** vs CPU with no config changes. Check the first log line
says `Device: mps`. If it falls back to CPU, force it:

```bash
export BRAINSR_DEVICE=mps
```

Mixed precision (`autocast` + `GradScaler`) is intentionally disabled on
MPS; PyTorch 2.4–2.5 fp16 support there is still incomplete.

### UMD Zaratan (HPC, recommended for E3–E5)

The full pipeline runs on Zaratan with one `sbatch`. SLURM job files,
setup script, and a step-by-step playbook live in
[`scripts/zaratan/README.md`](scripts/zaratan/README.md).

Headline workflow:

```bash
# On Zaratan (one-time):
git clone <this-repo> ~/brain-mri-image-resolution
cd ~/brain-mri-image-resolution
bash scripts/zaratan/setup_env.sh

# On your Mac (one-time, ~1.7 GB):
rsync -avh data/processed/ <id>@login.zaratan.umd.edu:~/brain-mri-image-resolution/data/processed/

# On Zaratan: all 5 experiments on a GPU node
sbatch scripts/zaratan/run_all.sbatch
```

Per-epoch timing from this codebase:

| Experiment             | Mac CPU       | Mac MPS    | Zaratan A100 |
| ---------------------- | ------------- | ---------- | ------------ |
| E2 SRCNN               | ~13 min       | ~45 s      | ~30 s        |
| E3–E4 AGUNet           | ~30+ min      | ~55 s      | ~60 s        |
| E5 AGUNet + critic     | ~50+ min      | ~75 s      | ~90 s        |

100-epoch E5 goes from "literally days on CPU" to ~2.5 hours on a single A100.

## Docker setup

The image is based on `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (CPU
out of the box; uncomment the `deploy` block in `docker-compose.yml` for GPU).
[`run.sh`](run.sh) is the friendly wrapper around these compose services:

| Service       | What it runs                                        |
| ------------- | --------------------------------------------------- |
| `demo`        | `python -m brainsr.cli.demo` (the default)          |
| `preprocess`  | `python -m brainsr.cli.preprocess`                  |
| `train`       | `python -m brainsr.cli.train --config ...`          |
| `dev`         | interactive shell with the source bind-mounted      |
| `tensorboard` | TB on port 6006                                     |

`docker-compose.yml` bind-mounts `${PROCESSED_DIR}` (the .npy cache),
`${RUNS_DIR}`, and `./checkpoints`, so all artifacts persist on the host
across container runs.

### Where the bundled checkpoints come from

`checkpoints/eN_*/` ships in the repo (and zip submission). Each folder
holds a `best.pt` plus the `config.resolved.yaml` it was trained with, so
the demo can rebuild the exact model architecture and load weights without
any guesswork. See [`checkpoints/README.md`](checkpoints/README.md) for the
layout and how to refresh after re-training.

### Hosting the preprocessed dataset (one-time)

The 1.7 GB preprocessed cache is too large to bundle in the zip and (per
FastMRI's data-use agreement) can't be redistributed alongside the code.
The demo fetches it on first run from a URL set in `PROCESSED_DATA_URL`.
Recommended: **Hugging Face Hub Datasets** — free, no quota, ~10 minutes
one-time:

```bash
# 1. Create the tarball (on whichever machine has data/processed/ ready)
tar czf processed.tar.gz -C data processed/

# 2. Install the CLI and log in
pip install huggingface_hub
huggingface-cli login   # paste a write-token from https://huggingface.co/settings/tokens

# 3. Create a public dataset repo and upload the tarball
huggingface-cli repo create brain-mri-processed --type dataset
huggingface-cli upload <username>/brain-mri-processed processed.tar.gz \
    processed.tar.gz --repo-type dataset

# 4. Point .env.example at the new URL (both /resolve/ and /blob/ work)
PROCESSED_DATA_URL=https://huggingface.co/datasets/<username>/brain-mri-processed/resolve/main/processed.tar.gz
```

GitHub Releases asset URLs (≤2 GB per file) and any other public HTTPS
.tar.gz / .tar / .zip also work. Drive *folder* URLs work as a slow
fallback. Drive *file* URLs are not supported (Google's virus-scan UI
breaks scripted downloads for files >100 MB).

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

## Evaluation metrics

PSNR / SSIM / NRMSE are computed both during training (torchmetrics, batched
on whatever device we're on) and offline against the held-out test split.
Targets per the proposal:

- PSNR ≥ bicubic + 2 dB (achieved: +4.55 dB on E4)
- SSIM > 0.90 (best E4: 0.8995)
- Reference: Li et al. 2022 — 35.39 PSNR, 0.985 SSIM

## Development

```bash
bash run.sh test    # pytest (synthetic phantom data, no FastMRI needed)
docker compose run --rm dev ruff check src tests
```

## Acknowledgements / references

- Li, B. M. et al. (2022). *Deep attention super-resolution of brain MRI
  acquired under clinical protocols.* Frontiers in Computational Neuroscience.
  <https://doi.org/10.3389/fncom.2022.887633> (AGUNet + DCGAN critic)
- Zbontar, J. et al. (2018). *fastMRI: An open dataset and benchmarks for
  accelerated MRI.* NYU FastMRI initiative.
- Dong, C. et al. (2014). *Image super-resolution using deep convolutional
  networks.* (SRCNN baseline)
- Oktay, O. et al. (2018). *Attention U-Net: Learning where to look for the
  pancreas.* (additive attention gate)
- Radford, A. et al. (2016). *Unsupervised representation learning with
  DCGAN.* (critic architecture)
- Miyato, T. et al. (2018). *Spectral normalization for GANs.* (training
  stability for E5)

## License

MIT. See [`LICENSE`](LICENSE).
