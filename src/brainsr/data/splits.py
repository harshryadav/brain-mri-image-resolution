"""Train / val / test split by volume id (default 70/20/10).

We split per-volume (one ``.h5`` -> one volume) so slices from the same
patient never end up in two splits. The result is written to
``splits.json`` next to the cached slices.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SplitName = str  # "train" | "val" | "test"


def _volume_id_from_slice_path(p: Path) -> str:
    name = p.stem
    if "_slice" in name:
        return name.split("_slice")[0]
    return name


def build_splits(
    processed_dir: Path,
    train: float = 0.70,
    val: float = 0.20,
    test: float = 0.10,
    seed: int = 42,
) -> dict[SplitName, list[str]]:
    """Group ``*.npy`` slices by volume id and emit a deterministic split.

    Returns a mapping ``{split_name: [slice_filename, ...]}`` (relative to
    ``processed_dir``) and writes ``splits.json`` next to the slices.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    processed_dir = Path(processed_dir)
    slice_paths = sorted(processed_dir.glob("*.npy"))
    if not slice_paths:
        raise FileNotFoundError(f"No .npy slices found under {processed_dir}")

    by_volume: dict[str, list[str]] = defaultdict(list)
    for p in slice_paths:
        by_volume[_volume_id_from_slice_path(p)].append(p.name)

    volumes = sorted(by_volume.keys())
    rng = random.Random(seed)
    rng.shuffle(volumes)

    n = len(volumes)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_vols = volumes[:n_train]
    val_vols = volumes[n_train : n_train + n_val]
    test_vols = volumes[n_train + n_val :]

    splits: dict[SplitName, list[str]] = {"train": [], "val": [], "test": []}
    for v in train_vols:
        splits["train"].extend(sorted(by_volume[v]))
    for v in val_vols:
        splits["val"].extend(sorted(by_volume[v]))
    for v in test_vols:
        splits["test"].extend(sorted(by_volume[v]))

    out_path = processed_dir / "splits.json"
    out_path.write_text(json.dumps(splits, indent=2))
    return splits


def load_splits(processed_dir: Path) -> dict[SplitName, list[str]]:
    path = Path(processed_dir) / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No splits.json under {processed_dir}. Run preprocess first or call build_splits()."
        )
    return json.loads(path.read_text())
