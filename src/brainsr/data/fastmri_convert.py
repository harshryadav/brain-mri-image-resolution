"""FastMRI ``.h5`` k-space -> per-slice ``.npy`` magnitude images.

For each volume we IFFT each coil, root-sum-of-squares across coils,
center-crop to a square FOV, bicubic-resize to a fixed resolution, and
normalize by the 99th percentile so values land in roughly ``[0, 1]``.
Slices are saved as ``{volume_id}_slice{idx:03d}.npy`` and treated as
independent samples downstream.

Caveat: the public ``multicoil_test`` batches are 8x undersampled with no
fully-sampled ground truth, so the IFFT+RSS image we treat as "HR" here is a
zero-filled aliased reconstruction. See the project README for what that
means for the metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class ConvertConfig:
    target_size: int = 256
    skip_edge_slices: int = 2
    percentile: float = 99.0
    dtype: np.dtype = np.float32
    acquisitions: tuple[str, ...] | None = None  # filter, e.g. ("AXT2", "AXFLAIR")


def kspace_to_magnitude(kspace: np.ndarray) -> np.ndarray:
    """Convert (slices, coils, H, W) complex k-space to (slices, H, W) magnitude via IFFT + RSS."""
    if kspace.ndim != 4:
        raise ValueError(f"Expected 4D k-space, got shape {kspace.shape}")
    centered = np.fft.ifftshift(kspace, axes=(-2, -1))
    img = np.fft.ifft2(centered, axes=(-2, -1), norm="ortho")
    img = np.fft.fftshift(img, axes=(-2, -1))
    rss = np.sqrt(np.sum(np.abs(img) ** 2, axis=1))
    return rss.astype(np.float32)


def center_crop_to_square(img: np.ndarray) -> np.ndarray:
    """Center-crop the longer axis so the result is square (``min(H, W)`` per side)."""
    h, w = img.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top : top + side, left : left + side]


def resize_bicubic(img: np.ndarray, target: int) -> np.ndarray:
    if img.shape == (target, target):
        return img.astype(np.float32, copy=False)
    pil = Image.fromarray(img.astype(np.float32))
    pil = pil.resize((target, target), resample=Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32)


def to_target_image(slice_img: np.ndarray, target: int) -> np.ndarray:
    """Square center-crop then bicubic resize to ``target x target``."""
    sq = center_crop_to_square(slice_img)
    return resize_bicubic(sq, target)


def normalize_volume(volume: np.ndarray, percentile: float) -> np.ndarray:
    """Divide a (slices, H, W) volume by its global p-th percentile (>0)."""
    p = float(np.percentile(volume, percentile))
    if p <= 0:
        p = float(volume.max()) or 1.0
    return volume / p


def list_h5_files(input_dirs: Iterable[Path]) -> list[Path]:
    """Sorted, deduplicated list of ``*.h5`` files across one or more input dirs."""
    files: dict[str, Path] = {}
    for d in input_dirs:
        d = Path(d)
        if not d.exists():
            log.warning("Input dir does not exist: %s", d)
            continue
        for p in sorted(d.rglob("*.h5")):
            files.setdefault(p.name, p)
    return sorted(files.values())


def read_acquisition(h5_path: Path) -> str | None:
    try:
        with h5py.File(h5_path, "r") as f:
            val = f.attrs.get("acquisition")
        if val is None:
            return None
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)
    except Exception:  # noqa: BLE001
        return None


def convert_volume(
    h5_path: Path,
    output_dir: Path,
    cfg: ConvertConfig,
) -> list[Path]:
    """Convert one .h5 volume to per-slice .npy files. Returns the written paths."""
    written: list[Path] = []
    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            log.warning("No 'kspace' key in %s; skipping", h5_path.name)
            return written
        if cfg.acquisitions:
            acq = f.attrs.get("acquisition")
            if isinstance(acq, bytes):
                acq = acq.decode("utf-8", errors="replace")
            if str(acq) not in cfg.acquisitions:
                return written
        kspace = f["kspace"][()]

    mag = kspace_to_magnitude(kspace)
    mag = normalize_volume(mag, cfg.percentile).astype(cfg.dtype)

    s = cfg.skip_edge_slices
    sl_start = min(s, max(0, mag.shape[0] - 1))
    sl_end = max(sl_start + 1, mag.shape[0] - s)

    output_dir.mkdir(parents=True, exist_ok=True)
    vol_id = h5_path.stem
    for idx in range(sl_start, sl_end):
        sl = to_target_image(mag[idx], cfg.target_size)
        out_path = output_dir / f"{vol_id}_slice{idx:03d}.npy"
        np.save(out_path, sl.astype(cfg.dtype))
        written.append(out_path)
    return written
