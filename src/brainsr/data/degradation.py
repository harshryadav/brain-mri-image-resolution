"""Synthetic LR generation: Gaussian blur + bicubic downsampling.

Matches the proposal's degradation: HR (256x256) -> Gaussian blur with sigma
sampled from ``sigma_range`` -> bicubic downsample by ``scale`` (default 4x),
giving an LR of size ``HR // scale``. Done on the fly per ``__getitem__``,
which doubles as data augmentation in train mode (random sigma) and stays
deterministic in eval mode for reproducible PSNR/SSIM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


@dataclass
class Degradation:
    scale: int = 4
    sigma_range: tuple[float, float] = (0.5, 2.0)
    kernel_size: int = 7  # odd
    deterministic: bool = False  # if True, always uses mean(sigma_range)

    def __call__(self, hr: torch.Tensor) -> torch.Tensor:
        """``hr``: (C, H, W) or (B, C, H, W) float tensor."""
        squeeze_batch = False
        if hr.ndim == 3:
            hr = hr.unsqueeze(0)
            squeeze_batch = True
        elif hr.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {hr.shape}")

        if self.deterministic:
            sigma = sum(self.sigma_range) / 2.0
        else:
            lo, hi = self.sigma_range
            sigma = float(torch.empty(1).uniform_(lo, hi).item())

        blurred = gaussian_blur(hr, kernel_size=[self.kernel_size, self.kernel_size], sigma=[sigma, sigma])

        _, _, h, w = blurred.shape
        out_h, out_w = h // self.scale, w // self.scale
        lr = F.interpolate(blurred, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)

        if squeeze_batch:
            lr = lr.squeeze(0)
        return lr.clamp(0.0, 1.0)


def upsample_bicubic(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic upsample (used for the E1 baseline and to align LR with HR for SRCNN)."""
    squeeze_batch = False
    if lr.ndim == 3:
        lr = lr.unsqueeze(0)
        squeeze_batch = True
    _, _, h, w = lr.shape
    up = F.interpolate(lr, size=(h * scale, w * scale), mode="bicubic", align_corners=False)
    if squeeze_batch:
        up = up.squeeze(0)
    return up.clamp(0.0, 1.0)
