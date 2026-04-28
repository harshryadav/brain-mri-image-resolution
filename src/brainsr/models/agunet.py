"""Attention-Gated U-Net for super-resolution.

Clean PyTorch reimplementation of the architecture from Li et al. (2022),
"Deep attention super-resolution of brain MRI acquired under clinical
protocols". The network takes an LR input of shape ``(N, C, H/scale, W/scale)``,
upsamples to ``(N, C, H, W)`` with a PixelShuffle head, then runs a 4-level
encoder/decoder U-Net with optional additive attention gates on each skip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_gates import AttentionGate


def _norm(name: str, num_features: int) -> nn.Module:
    name = name.lower()
    if name == "instancenorm":
        return nn.InstanceNorm2d(num_features, affine=True)
    if name == "batchnorm":
        return nn.BatchNorm2d(num_features)
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown normalization: {name}")


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        normalization: str = "instancenorm",
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(normalization, out_ch),
            _act(activation),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(normalization, out_ch),
            _act(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Transposed-conv upsample then ConvBlock on the concatenated skip."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        normalization: str,
        activation: str,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(x_channels=skip_ch, g_channels=out_ch)
        self.conv = ConvBlock(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention:
            skip = self.attn(skip, g=x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AGUNet(nn.Module):
    """Attention-gated U-Net super-resolver."""

    needs_bicubic_input: bool = False

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale: int = 4,
        num_filters: int = 32,
        depth: int = 4,
        normalization: str = "instancenorm",
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_attention: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if scale not in {2, 4}:
            raise ValueError("scale must be 2 or 4")
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.scale = scale
        self.residual = residual

        head_layers: list[nn.Module] = [nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)]
        s = scale
        while s > 1:
            head_layers += [
                nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                _act(activation),
            ]
            s //= 2
        self.head = nn.Sequential(*head_layers)

        chs = [num_filters * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.encoders.append(
            ConvBlock(num_filters, chs[0], normalization, activation, dropout)
        )
        for i in range(1, depth):
            self.encoders.append(
                ConvBlock(chs[i - 1], chs[i], normalization, activation, dropout)
            )
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2, normalization, activation, dropout)

        self.decoders = nn.ModuleList()
        prev_ch = chs[-1] * 2
        for i in reversed(range(depth)):
            self.decoders.append(
                UpBlock(
                    in_ch=prev_ch,
                    skip_ch=chs[i],
                    out_ch=chs[i],
                    normalization=normalization,
                    activation=activation,
                    dropout=dropout,
                    use_attention=use_attention,
                )
            )
            prev_ch = chs[i]

        self.tail = nn.Conv2d(chs[0], out_channels, kernel_size=3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(
            lr,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        ) if self.residual else None

        x = self.head(lr)

        skips: list[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < len(self.encoders) - 1:
                x = self.pool(x)

        x = self.pool(x)
        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        out = self.tail(x)
        if self.residual and up is not None:
            if out.shape[-2:] != up.shape[-2:]:
                out = F.interpolate(out, size=up.shape[-2:], mode="bilinear", align_corners=False)
            out = out + up
        return out
