"""``brainsr-demo``: end-to-end inference demo on the held-out test split.

This is the "single command" entry point for graders. It:

1. Ensures the preprocessed dataset is present (downloads it from Google
   Drive via ``scripts/download_data.py`` on first run).
2. Ensures the ``checkpoints/`` folder is populated (optionally downloads
   it via ``scripts/download_model.py`` if missing).
3. For every experiment found under ``checkpoints/eN_*/``, rebuilds the
   model from its ``config.resolved.yaml``, loads ``best.pt``, evaluates
   on the test split, and saves a few sample LR/SR/HR triplet PNGs.
4. Aggregates per-experiment metrics into ``runs/results.csv`` and
   prints a summary table.

Run directly::

    python -m brainsr.cli.demo
    python -m brainsr.cli.demo --checkpoints-dir checkpoints --output-dir runs --num-samples 6
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import MRISliceDataset
from ..models.registry import build_model
from ..trainer import _device, _save_sample_panels, evaluate

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_helper_script(script: Path, *args: str) -> int:
    """Invoke a helper Python script in a subprocess; surface its exit code."""
    cmd = [sys.executable, str(script), *args]
    log.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def _ensure_data(processed_dir: Path) -> None:
    if (processed_dir / "splits.json").exists():
        log.info("Processed data already present at %s", processed_dir)
        return
    helper = REPO_ROOT / "scripts" / "download_data.py"
    if not helper.exists():
        raise SystemExit(
            f"No processed data at {processed_dir} and no download helper found at {helper}.\n"
            "Run `make preprocess` first, or place the preprocessed .npy cache at that path."
        )
    rc = _run_helper_script(helper, "--output-dir", str(processed_dir))
    if rc != 0 or not (processed_dir / "splits.json").exists():
        raise SystemExit(
            f"Failed to obtain processed data at {processed_dir}. "
            "Set PROCESSED_DATA_URL in .env or run `make preprocess` manually."
        )


def _ensure_checkpoints(ckpt_dir: Path) -> None:
    has_any = ckpt_dir.exists() and any(ckpt_dir.glob("*/config.resolved.yaml"))
    if has_any:
        log.info("Checkpoints already present at %s", ckpt_dir)
        return
    helper = REPO_ROOT / "scripts" / "download_model.py"
    if not helper.exists():
        raise SystemExit(
            f"No checkpoints at {ckpt_dir} and no download helper found. "
            "Train models first, or set CHECKPOINTS_URL in .env."
        )
    rc = _run_helper_script(helper, "--output-dir", str(ckpt_dir))
    if rc != 0 or not any(ckpt_dir.glob("*/config.resolved.yaml")):
        raise SystemExit(
            f"Failed to obtain checkpoints at {ckpt_dir}. "
            "Set CHECKPOINTS_URL in .env or train E1-E5 first."
        )


def _build_model_from_cfg(cfg: dict[str, Any]) -> tuple[torch.nn.Module, int]:
    scale = int(cfg.get("data", {}).get("scale", cfg.get("scale", 4)))
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    if name == "agunet" and "scale" not in model_cfg:
        model_cfg["scale"] = scale
    model = build_model(name, **model_cfg)
    return model, scale


def _load_test_loader(cfg: dict[str, Any], data_root: Path, batch_size: int) -> DataLoader:
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["root"] = str(data_root)
    sigma_range = tuple(data_cfg.get("sigma_range", [0.5, 2.0]))
    scale = int(data_cfg.get("scale", cfg.get("scale", 4)))
    test_ds = MRISliceDataset(
        root=data_cfg["root"],
        split="test",
        scale=scale,
        sigma_range=sigma_range,
        deterministic_lr=True,
    )
    pin = torch.cuda.is_available()
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )


def _evaluate_one(
    exp_name: str,
    ckpt_subdir: Path,
    data_root: Path,
    out_root: Path,
    num_samples: int,
    batch_size: int,
) -> dict[str, float] | None:
    cfg_path = ckpt_subdir / "config.resolved.yaml"
    if not cfg_path.exists():
        log.warning("Skipping %s: no config.resolved.yaml", exp_name)
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    model, scale = _build_model_from_cfg(cfg)

    weights_path = ckpt_subdir / "best.pt"
    if weights_path.exists():
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        log.info("[%s] Loaded weights from %s", exp_name, weights_path)
    else:
        # E1 (bicubic) has no learnable weights; this is fine for it only.
        if cfg.get("model", {}).get("name", "").lower() != "bicubic":
            log.warning("[%s] No best.pt found; falling back to randomly-initialized weights.", exp_name)

    device = _device()
    model = model.to(device).eval()

    test_loader = _load_test_loader(cfg, data_root, batch_size=batch_size)
    log.info("[%s] Evaluating on %d test slices (device=%s)", exp_name, len(test_loader.dataset), device)
    metrics = evaluate(model, test_loader, device, scale)

    samples_dir = out_root / exp_name / "samples"
    _save_sample_panels(model, test_loader, device, scale, samples_dir, "test", n=num_samples)

    log.info(
        "[%s] PSNR=%.3f  SSIM=%.4f  NRMSE=%.4f  (samples -> %s)",
        exp_name, metrics["psnr"], metrics["ssim"], metrics["nrmse"], samples_dir,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end inference demo over bundled checkpoints")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--num-samples", type=int, default=4, help="LR/SR/HR triplet PNGs per experiment")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-data-download",
        action="store_true",
        help="Don't try to fetch data from Google Drive even if data/processed is empty",
    )
    parser.add_argument(
        "--skip-checkpoint-download",
        action="store_true",
        help="Don't try to fetch checkpoints from a remote URL even if checkpoints/ is empty",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    ckpt_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_checkpoint_download:
        _ensure_checkpoints(ckpt_dir)
    if not args.skip_data_download:
        _ensure_data(data_dir)

    if not (data_dir / "splits.json").exists():
        raise SystemExit(
            f"Cannot run demo: no splits.json at {data_dir}. "
            "Run preprocessing or provide PROCESSED_DATA_URL in .env."
        )

    exp_dirs = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and (p / "config.resolved.yaml").exists())
    if not exp_dirs:
        raise SystemExit(
            f"No experiments under {ckpt_dir}. Each subfolder needs a config.resolved.yaml "
            "(and a best.pt unless it's the bicubic baseline)."
        )

    rows: list[dict[str, str | float]] = []
    for exp_subdir in exp_dirs:
        metrics = _evaluate_one(
            exp_name=exp_subdir.name,
            ckpt_subdir=exp_subdir,
            data_root=data_dir,
            out_root=out_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        if metrics is None:
            continue

        # Drop a per-experiment summary into runs/<exp>/ to match the trainer's layout.
        per_run_dir = out_dir / exp_subdir.name
        per_run_dir.mkdir(parents=True, exist_ok=True)
        (per_run_dir / "summary.json").write_text(
            yaml.safe_dump({"test": metrics, "source": "demo"}, sort_keys=False)
        )
        # Ship the resolved config alongside so brainsr-eval and brainsr-predict can find it.
        cfg_src = exp_subdir / "config.resolved.yaml"
        cfg_dst = per_run_dir / "config.resolved.yaml"
        if cfg_src.exists() and not cfg_dst.exists():
            shutil.copy2(cfg_src, cfg_dst)

        rows.append({
            "run": exp_subdir.name,
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "nrmse": float(metrics["nrmse"]),
            "checkpoint": str(exp_subdir / "best.pt"),
        })

    if not rows:
        raise SystemExit("No experiments produced metrics; check the logs above.")

    results_csv = out_dir / "results.csv"
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "psnr", "ssim", "nrmse", "checkpoint"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print()
    print("=" * 72)
    print(f"{'experiment':<32} {'PSNR':>8} {'SSIM':>8} {'NRMSE':>8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['run']:<32} {r['psnr']:>8.3f} {r['ssim']:>8.4f} {r['nrmse']:>8.4f}")
    print("=" * 72)
    print(f"Per-experiment samples: {out_dir}/<exp>/samples/")
    print(f"Aggregate metrics:      {results_csv}")


if __name__ == "__main__":
    main()
