# Re-training, dataset prep, and performance

The bundled checkpoints in `checkpoints/eN_*/` already produce the headline
metrics in the main [`README.md`](../README.md). This doc is for reproducing
or extending those runs from scratch.

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

## Re-training from scratch

### 1. Get the raw FastMRI data

FastMRI is not redistributable, so the repo can't ship raw data. Follow
[`scripts/download_fastmri_help.md`](../scripts/download_fastmri_help.md)
to grab the `.h5` files yourself.

> **Caveat for the public `multicoil_test` batches.** They're 8×
> undersampled with no fully-sampled ground truth. The pipeline still
> works (we IFFT + RSS the masked k-space and treat the zero-filled
> reconstruction as our HR target), but absolute PSNR/SSIM aren't
> directly comparable to papers that train on `multicoil_train`. The
> shipped checkpoints are trained on the fully-sampled `multicoil_train`
> partition.

### 2. Point the project at the raw data

```bash
cp .env.example .env
# edit FASTMRI_DIR=/abs/path/to/multicoil_train
```

For multiple directories use `FASTMRI_DIRS` (space-separated), e.g.
several test batches untarred side by side.

### 3. Preprocess once

Converts every `.h5` volume to per-slice `.npy` magnitude images and writes
a deterministic 70/20/10 volume-level split:

```bash
bash run.sh preprocess
# forward extra flags via the trailing args, e.g.:
bash run.sh preprocess --acquisition AXT2 --limit 100
```

Useful flags: `--acquisition AXT2,AXFLAIR` (subset by FastMRI brain
contrast), `--limit N`, `--target-size 256`.

### 4. Train

Each YAML maps to one row of the ablation:

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

### 5. Run all experiments and aggregate metrics

```bash
bash run.sh train-all   # trains E1..E5, then writes runs/results.csv
```

### 6. Refresh the bundled checkpoints

After a re-training run, copy the artifacts so the demo picks them up:

```bash
for exp in e1_bicubic e2_srcnn e3_agunet_mse e4_agunet_attn e5_agunet_attn_dcgan; do
    mkdir -p checkpoints/$exp
    [ -f runs/$exp/best.pt ] && cp runs/$exp/best.pt checkpoints/$exp/
    cp runs/$exp/config.resolved.yaml checkpoints/$exp/
done
```

## Performance tips

### Apple Silicon (Mac)

The trainer auto-detects MPS — on an M-series Mac you should see roughly
**5–15× speedup** vs CPU with no config changes. Check the first log line
says `Device: mps`. If it falls back to CPU, force it:

```bash
export BRAINSR_DEVICE=mps
```

Mixed precision (`autocast` + `GradScaler`) is intentionally disabled on
MPS; PyTorch 2.4–2.5 fp16 support there is still incomplete.

### Per-epoch timing

| Experiment             | Mac CPU       | Mac MPS    | Zaratan A100 |
| ---------------------- | ------------- | ---------- | ------------ |
| E2 SRCNN               | ~13 min       | ~45 s      | ~30 s        |
| E3–E4 AGUNet           | ~30+ min      | ~55 s      | ~60 s        |
| E5 AGUNet + critic     | ~50+ min      | ~75 s      | ~90 s        |

100-epoch E5 goes from "literally days on CPU" to ~2.5 hours on a single A100.

### UMD Zaratan (HPC, recommended for E3–E5)

Full SLURM playbook with setup script and sbatch files lives at
[`scripts/zaratan/README.md`](../scripts/zaratan/README.md).

## Hosting the preprocessed dataset

The 1.7 GB preprocessed cache is too large to bundle in the zip and (per
FastMRI's data-use agreement) can't be redistributed alongside the code.
The demo fetches it on first run from a URL set in `PROCESSED_DATA_URL`.
Recommended host: **Hugging Face Hub Datasets** (free, no quota, ~10 min
one-time):

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

## Underlying compose services

`run.sh` is a wrapper around these `docker compose` services:

| Service       | What it runs                                        |
| ------------- | --------------------------------------------------- |
| `demo`        | `python -m brainsr.cli.demo` (the default)          |
| `preprocess`  | `python -m brainsr.cli.preprocess`                  |
| `train`       | `python -m brainsr.cli.train --config ...`          |
| `dev`         | interactive shell with the source bind-mounted      |
| `tensorboard` | TB on port 6006                                     |

`docker-compose.yml` bind-mounts `${PROCESSED_DIR}` (the `.npy` cache),
`${RUNS_DIR}`, and `./checkpoints`, so all artifacts persist on the host
across container runs.

## References

- Li, B. M. et al. (2022). *Deep attention super-resolution of brain MRI
  acquired under clinical protocols.* Frontiers in Computational Neuroscience.
  <https://doi.org/10.3389/fncom.2022.887633> (AGUNet + DCGAN critic)
- Zbontar, J. et al. (2018). *fastMRI: An open dataset and benchmarks for
  accelerated MRI.* NYU FastMRI initiative.
- Dong, C. et al. (2014). *Image super-resolution using deep convolutional
  networks.* (SRCNN baseline)
- Oktay, O. et al. (2018). *Attention U-Net: Learning where to look for the
  pancreas.* (additive attention gate)
- Radford, A. et al. (2016). *Unsupervised representation learning with DCGAN.*
- Miyato, T. et al. (2018). *Spectral normalization for GANs.* (E5 stability)
- Wang, X. et al. (2021). *Real-ESRGAN: Training real-world blind super-
  resolution with pure synthetic data.* (LR degradation pipeline)
