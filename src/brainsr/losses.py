"""Reconstruction (MSE/L1) and adversarial (BCE/hinge) losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown reconstruction loss: {name}")


class AdversarialLoss(nn.Module):
    """Wraps BCE-with-logits or hinge loss for both critic and generator updates."""

    def __init__(self, kind: str = "bce", label_smoothing: float = 0.0) -> None:
        super().__init__()
        kind = kind.lower()
        if kind not in {"bce", "hinge"}:
            raise ValueError(f"Unknown adversarial loss: {kind}")
        self.kind = kind
        self.label_smoothing = float(label_smoothing)

    def critic_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.kind == "bce":
            real_target = torch.ones_like(real_logits)
            fake_target = torch.zeros_like(fake_logits)
            if self.label_smoothing > 0:
                real_target = real_target - self.label_smoothing
            return F.binary_cross_entropy_with_logits(real_logits, real_target) + F.binary_cross_entropy_with_logits(fake_logits, fake_target)
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.kind == "bce":
            target = torch.ones_like(fake_logits)
            return F.binary_cross_entropy_with_logits(fake_logits, target)
        return -fake_logits.mean()
