# Brain MRI Super-Resolution (FastMRI)

Authors: **Harsh Yadav**, **Deepika Ghotra**, **Utkrisht Nath**.

A deep-learning super-resolution system for brain MRI on the
[NYU FastMRI](https://fastmri.med.nyu.edu/) dataset. Five-experiment ablation
(bicubic → SRCNN → AGUNet variants). Pre-trained checkpoints ship with the
repo; the demo runs end-to-end inference on the held-out test split.

## Prerequisites

- **Docker Desktop** with Compose v2 (verify with `docker compose version`).
  Get it at <https://docs.docker.com/get-docker/>.
- ~2 GB free disk for the preprocessed dataset (auto-downloaded on first run).
- Internet access on the first run.

## One-command demo

```bash
bash run.sh
```

What it does on the first run:

1. Builds the Docker image (~5 min the first time due to the large pytorch image, cached afterwards).
2. Downloads the preprocessed FastMRI cache from Hugging Face into
   `data/processed/` (~1.7 GB, cached afterwards).
3. Evaluates every bundled checkpoint under `checkpoints/eN_*/` on the
   held-out test split.

Outputs are stored in `runs/`:

- `runs/results.csv` — contains PSNR / SSIM / NRMSE per experiment
- `runs/<exp>/samples/*.png` — side-by-side **LR | SR | HR** images
- `runs/<exp>/summary.json` — per-experiment metrics

Force a rebuild with `BRAINSR_REBUILD=1 bash run.sh`.

## Other entry points

All commands are dispatched through `run.sh` and run inside the same image.

| Command                                         | What it does                                          |
| ----------------------------------------------- | ----------------------------------------------------- |
| `bash run.sh`                                   | inference demo (default)                              |
| `bash run.sh train [CONFIG]`                    | train one experiment (defaults to `configs/e2_srcnn.yaml`) |
| `bash run.sh train-all`                         | train E1..E5 in sequence then aggregate `results.csv` |
| `bash run.sh preprocess`                        | convert raw FastMRI `.h5` → `data/processed/`         |
| `bash run.sh smoke`                             | synthetic-phantom sanity check, no downloads          |
| `bash run.sh shell`                             | interactive bash shell in the container               |
| `bash run.sh tensorboard`                       | TB on <http://localhost:6006>                         |
| `bash run.sh test`                              | pytest                                                |
| `bash run.sh help`                              | print this list                                       |

## More documentation

- **Re-training, dataset prep, performance tips** → [`docs/training.md`](docs/training.md)
- **Running on UMD Zaratan (HPC)** → [`scripts/zaratan/README.md`](scripts/zaratan/README.md)
- **Bundled checkpoint layout** → [`checkpoints/README.md`](checkpoints/README.md)
