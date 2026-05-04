"""SRCNN: the classic 3-layer SR baseline from Dong et al. (2014).

Three named conv layers - patch extraction, non-linear mapping, then
reconstruction - with the canonical (9, 5, 5) kernel sizes and (64, 32)
channel widths from Section 3 of the paper. The model expects its input
to already be bicubic-upsampled to HR size; the trainer does that for us
when it sees ``needs_bicubic_input = True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    needs_bicubic_input: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        f1: int = 9,
        f2: int = 5,
        f3: int = 5,
        n1: int = 64,
        n2: int = 32,
    ) -> None:
        super().__init__()
        self.patch_extraction = nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2)
        self.non_linear_mapping = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.reconstruction = nn.Conv2d(n2, out_channels, kernel_size=f3, padding=f3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.patch_extraction(x))
        x = self.relu(self.non_linear_mapping(x))
        return self.reconstruction(x)
