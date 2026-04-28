import torch

from brainsr.data.degradation import Degradation, upsample_bicubic


def test_degradation_shape_and_range_x4():
    hr = torch.rand(2, 1, 64, 64)
    lr = Degradation(scale=4, deterministic=True)(hr)
    assert lr.shape == (2, 1, 16, 16)
    assert 0.0 <= lr.min().item()
    assert lr.max().item() <= 1.0


def test_degradation_shape_x2():
    hr = torch.rand(1, 1, 32, 32)
    lr = Degradation(scale=2, deterministic=True)(hr)
    assert lr.shape == (1, 1, 16, 16)


def test_upsample_bicubic_roundtrip():
    lr = torch.rand(1, 1, 8, 8)
    up = upsample_bicubic(lr, scale=4)
    assert up.shape == (1, 1, 32, 32)
