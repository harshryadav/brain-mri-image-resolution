"""Dataset over cached ``.npy`` magnitude slices.

Returns ``(lr, hr)`` tensors in ``[0, 1]``. HR is loaded from disk; LR is
generated on the fly by :class:`Degradation` so we can sweep scale/blur
without re-preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import Degradation
from .splits import load_splits


class MRISliceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        scale: int = 4,
        sigma_range: tuple[float, float] = (0.5, 2.0),
        deterministic_lr: bool = False,
        return_filename: bool = False,
    ) -> None:
        self.root = Path(root)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train|val|test, got {split!r}")
        self.split = split
        self.return_filename = return_filename

        splits = load_splits(self.root)
        self.files: list[Path] = [self.root / name for name in splits[split]]
        if not self.files:
            raise RuntimeError(f"Empty split '{split}' under {self.root}")

        self.degrade = Degradation(
            scale=scale,
            sigma_range=sigma_range,
            deterministic=deterministic_lr,
        )

    def __len__(self) -> int:
        return len(self.files)

    def _load_hr(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        hr = self._load_hr(path)
        lr = self.degrade(hr)
        if self.return_filename:
            return lr, hr, path.name
        return lr, hr
