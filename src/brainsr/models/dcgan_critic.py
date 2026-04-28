"""DCGAN-style critic for the AGUNet generator.

Stack of stride-2 convolutions (each halving spatial size), then a final
conv that collapses to a single logit per image. Spectral norm is on by
default for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DCGANCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.0,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        def _maybe_sn(module: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(module) if use_spectral_norm else module

        layers: list[nn.Module] = []
        in_ch = in_channels
        out_ch = num_filters
        for i in range(num_blocks):
            layers.append(_maybe_sn(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)))
            if i > 0:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
            out_ch = min(out_ch * 2, num_filters * 8)

        layers.append(_maybe_sn(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.flatten(1).mean(dim=1)
