"""Side-by-side ``LR | SR | HR`` PNGs for debugging and report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3:
        img = img[0]
    return img.detach().cpu().clamp(0.0, 1.0).numpy()


def save_triplet_grid(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    """Save a side-by-side ``LR | SR | HR`` PNG. Inputs are tensors in [0,1]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if lr.shape[-2:] != hr.shape[-2:]:
        scale = hr.shape[-1] // lr.shape[-1]
        lr_disp = F.interpolate(
            lr if lr.ndim == 4 else lr.unsqueeze(0),
            scale_factor=scale,
            mode="nearest",
        )
    else:
        lr_disp = lr

    panels = [_to_numpy(lr_disp), _to_numpy(sr), _to_numpy(hr)]
    labels = ["LR (nearest-up)", "SR", "HR"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, img, label in zip(axes, panels, labels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
