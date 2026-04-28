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

## Project layout

```
brain-mri-image-resolution/
  configs/         # YAML, one per experiment + base.yaml
  src/brainsr/     # installable package (brainsr-{preprocess,train,eval,predict})
    data/          # fastmri_convert, degradation, dataset, splits
    models/        # bicubic, srcnn, agunet (+ attention gates), dcgan_critic
    cli/           # console-script entry points
    losses.py metrics.py trainer.py utils/
  scripts/         # run_all_experiments.sh, FastMRI download notes, Zaratan SLURM jobs
  tests/           # pytest suite (uses synthetic phantom slices)
  data/sample/     # tiny synthetic dataset for `make smoke` (committed)
  data/{raw,processed}/  # gitignored - your FastMRI .h5 + the .npy cache
  runs/            # gitignored - per-experiment TB logs, checkpoints, samples
  Dockerfile docker-compose.yml Makefile pyproject.toml requirements.txt
```

## Quick start (no FastMRI required)

Generate a tiny synthetic phantom dataset and run a 1-epoch SRCNN training
to verify the whole pipeline boots:

```bash
make install
make smoke
```

Outputs land in `runs/_smoke/`. Open them in TensorBoard:

```bash
make tb     # http://localhost:6006
```

## Full pipeline with FastMRI

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

```bash
make docker-build
make docker-shell                        # interactive shell
make docker-preprocess                   # one-shot preprocessing
make docker-train CONFIG=configs/e2_srcnn.yaml
docker compose up tensorboard            # http://localhost:6006
```

`docker-compose.yml` bind-mounts `${FASTMRI_DIR}` (read-only),
`./data/processed`, and `./runs`, so checkpoints and TB logs persist on the
host.

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
