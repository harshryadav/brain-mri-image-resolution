import numpy as np
import torch

from brainsr.metrics import MetricBank, metrics_per_image


def test_perfect_prediction_high_psnr():
    target = np.clip(np.random.rand(64, 64).astype(np.float32), 0, 1)
    out = metrics_per_image(target.copy(), target)
    assert out.psnr > 60  # essentially +inf
    assert out.ssim > 0.99
    assert out.nrmse < 1e-4


def test_metric_bank_matches_skimage_within_tol():
    rng = np.random.default_rng(0)
    target = rng.random((4, 1, 64, 64)).astype(np.float32)
    pred = np.clip(target + 0.05 * rng.standard_normal(target.shape).astype(np.float32), 0, 1)

    bank = MetricBank(data_range=1.0, device="cpu")
    bank.update(torch.from_numpy(pred), torch.from_numpy(target))
    bank_vals = bank.compute()

    sk_vals = metrics_per_image(pred[0, 0], target[0, 0])
    assert abs(bank_vals.psnr - sk_vals.psnr) < 5.0
    assert abs(bank_vals.ssim - sk_vals.ssim) < 0.1
