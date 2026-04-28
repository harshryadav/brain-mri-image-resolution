import torch

from brainsr.models.registry import build_critic, build_model


def test_bicubic_forward():
    model = build_model("bicubic", scale=4)
    lr = torch.rand(2, 1, 16, 16)
    out = model(lr)
    assert out.shape == (2, 1, 64, 64)


def test_srcnn_needs_bicubic_input():
    model = build_model("srcnn")
    assert model.needs_bicubic_input is True
    x = torch.rand(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_agunet_x4_shape():
    model = build_model("agunet", scale=4, num_filters=8, depth=3, use_attention=True)
    lr = torch.rand(1, 1, 16, 16)
    out = model(lr)
    assert out.shape == (1, 1, 64, 64)


def test_agunet_x2_no_attention():
    model = build_model("agunet", scale=2, num_filters=8, depth=3, use_attention=False)
    lr = torch.rand(1, 1, 32, 32)
    out = model(lr)
    assert out.shape == (1, 1, 64, 64)


def test_dcgan_critic_returns_scalar_per_sample():
    critic = build_critic("dcgan", in_channels=1, num_filters=8, num_blocks=3)
    x = torch.rand(2, 1, 64, 64)
    out = critic(x)
    assert out.shape == (2,)


def test_build_critic_none():
    assert build_critic(None) is None
    assert build_critic("none") is None
