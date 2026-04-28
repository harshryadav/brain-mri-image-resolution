"""PSNR / SSIM / NRMSE.

torchmetrics for the batched train+val path (runs on GPU/MPS), skimage for
per-image offline use and as a sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import normalized_root_mse as sk_nrmse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


@dataclass
class MetricValues:
    psnr: float
    ssim: float
    nrmse: float

    def as_dict(self) -> dict[str, float]:
        return {"psnr": self.psnr, "ssim": self.ssim, "nrmse": self.nrmse}


class MetricBank:
    """Accumulator wrapping torchmetrics for batched updates during eval."""

    def __init__(self, data_range: float = 1.0, device: torch.device | str = "cpu") -> None:
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        self._sq_err_sum = torch.tensor(0.0, device=device)
        self._sq_target_sum = torch.tensor(0.0, device=device)
        self._device = device

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.to(self._device).clamp(0.0, 1.0)
        target = target.to(self._device).clamp(0.0, 1.0)
        self.psnr.update(pred, target)
        self.ssim.update(pred, target)
        self._sq_err_sum += ((pred - target) ** 2).sum()
        self._sq_target_sum += (target**2).sum()

    def compute(self) -> MetricValues:
        psnr_val = float(self.psnr.compute().item())
        ssim_val = float(self.ssim.compute().item())
        denom = float(self._sq_target_sum.item()) or 1.0
        nrmse_val = float(np.sqrt(float(self._sq_err_sum.item()) / denom))
        return MetricValues(psnr=psnr_val, ssim=ssim_val, nrmse=nrmse_val)

    def reset(self) -> None:
        self.psnr.reset()
        self.ssim.reset()
        self._sq_err_sum.zero_()
        self._sq_target_sum.zero_()


def metrics_per_image(pred: np.ndarray, target: np.ndarray) -> MetricValues:
    """skimage-based per-image metrics for offline evaluation / sanity checks."""
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    target = np.clip(target, 0.0, 1.0).astype(np.float64)
    psnr = float(sk_psnr(target, pred, data_range=1.0))
    ssim = float(sk_ssim(target, pred, data_range=1.0))
    nrmse = float(sk_nrmse(target, pred))
    return MetricValues(psnr=psnr, ssim=ssim, nrmse=nrmse)
