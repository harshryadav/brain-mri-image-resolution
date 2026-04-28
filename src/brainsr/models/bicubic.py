"""Trivial bicubic upsampler used as the E1 baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicUpsampler(nn.Module):
    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        h, w = lr.shape[-2:]
        return F.interpolate(
            lr,
            size=(h * self.scale, w * self.scale),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
