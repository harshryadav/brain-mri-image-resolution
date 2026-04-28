"""``brainsr-preprocess``: FastMRI .h5 -> per-slice .npy + train/val/test splits.

Two modes:

- ``--input-dir <dir>`` (repeatable): convert every ``.h5`` under those dirs.
- ``--build-sample``: synthesize a tiny phantom dataset for ``make smoke`` /
  CI (no FastMRI required).

Optional filters: ``--acquisition AXT2,AXFLAIR`` to subset by FastMRI brain
contrast; ``--limit N`` to cap volume count for quick iteration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.fastmri_convert import (
    ConvertConfig,
    convert_volume,
    list_h5_files,
    read_acquisition,
)
from ..data.splits import build_splits

log = logging.getLogger(__name__)


def _make_phantom_slice(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Brain-ish phantom: ellipse + a few inner blobs + noise. For tests only."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2, size / 2
    a, b = size * 0.40, size * 0.32
    skull = ((xx - cx) ** 2 / a**2 + (yy - cy) ** 2 / b**2) <= 1.0
    img = skull.astype(np.float32) * 0.6
    for _ in range(rng.integers(2, 6)):
        rx = rng.uniform(20, 50)
        ry = rng.uniform(20, 50)
        cyi = rng.uniform(cy - b * 0.4, cy + b * 0.4)
        cxi = rng.uniform(cx - a * 0.4, cx + a * 0.4)
        blob = ((xx - cxi) ** 2 / rx**2 + (yy - cyi) ** 2 / ry**2) <= 1.0
        img += blob.astype(np.float32) * float(rng.uniform(0.1, 0.3))
    img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _build_sample(output_dir: Path, n_volumes: int = 6, slices_per_volume: int = 4, size: int = 256) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for v in range(n_volumes):
        vol_id = f"sample_vol{v:02d}"
        for s in range(slices_per_volume):
            arr = _make_phantom_slice(rng, size=size).astype(np.float32)
            np.save(output_dir / f"{vol_id}_slice{s:03d}.npy", arr)
    splits = build_splits(output_dir, train=0.5, val=0.25, test=0.25, seed=42)
    log.info("Sample built: %s | sizes: %s", output_dir, {k: len(v) for k, v in splits.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="FastMRI .h5 -> .npy preprocessing")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Directory containing FastMRI .h5 files. Repeat for multiple dirs.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--skip-edge-slices", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument(
        "--acquisition",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. 'AXT2' or 'AXT2,AXFLAIR'. Default: all.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N volumes")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--build-sample",
        action="store_true",
        help="Generate synthetic phantom slices instead of converting FastMRI",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    output_dir = Path(args.output_dir)

    if args.build_sample:
        _build_sample(output_dir)
        return

    if not args.input_dir:
        raise SystemExit("--input-dir is required unless --build-sample is set")

    input_dirs = [Path(p) for p in args.input_dir]
    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dir not found: {d}")

    acquisitions: tuple[str, ...] | None = None
    if args.acquisition:
        acquisitions = tuple(a.strip() for a in args.acquisition.split(",") if a.strip())

    cfg = ConvertConfig(
        target_size=args.target_size,
        skip_edge_slices=args.skip_edge_slices,
        percentile=args.percentile,
        acquisitions=acquisitions,
    )

    h5_files = list_h5_files(input_dirs)
    log.info("Discovered %d unique .h5 files across %d directories", len(h5_files), len(input_dirs))

    if acquisitions:
        keep: list[Path] = []
        for p in tqdm(h5_files, desc="filtering by acquisition"):
            acq = read_acquisition(p)
            if acq in acquisitions:
                keep.append(p)
        h5_files = keep
        log.info("After acquisition filter %s: %d files", acquisitions, len(h5_files))

    if args.limit:
        h5_files = h5_files[: args.limit]
        log.info("Limited to first %d files", len(h5_files))

    if not h5_files:
        raise SystemExit("No .h5 files match the requested filters")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Converting %d volumes -> %s", len(h5_files), output_dir)
    total_slices = 0
    failures = 0
    for h5 in tqdm(h5_files, desc="volumes"):
        try:
            written = convert_volume(h5, output_dir, cfg)
            total_slices += len(written)
        except Exception as e:  # noqa: BLE001
            failures += 1
            log.error("Failed to convert %s: %s", h5, e)

    log.info("Wrote %d slices (failures=%d)", total_slices, failures)
    splits = build_splits(
        output_dir, train=args.train, val=args.val, test=args.test, seed=args.seed,
    )
    log.info("Splits: %s", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
