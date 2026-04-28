"""Additive attention gate (Oktay et al., 2018) used on U-Net skip connections.

The gate takes a low-resolution gating signal ``g`` (from the decoder side)
and a higher-resolution skip feature ``x`` (from the encoder), produces a
soft attention map alpha in ``[0, 1]``, and returns ``x * alpha``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, x_channels: int, g_channels: int, inter_channels: int | None = None) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(x_channels // 2, 1)
        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta = self.theta_x(x)
        phi = self.phi_g(g)
        if phi.shape[-2:] != theta.shape[-2:]:
            phi = F.interpolate(phi, size=theta.shape[-2:], mode="bilinear", align_corners=False)
        f = F.relu(theta + phi, inplace=True)
        alpha = torch.sigmoid(self.psi(f))
        return x * alpha
