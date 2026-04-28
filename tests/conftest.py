"""Pytest fixtures. ``sample_dir`` builds a tiny phantom dataset once per session."""

from __future__ import annotations

from pathlib import Path

import pytest

from brainsr.cli.preprocess import _build_sample


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sample")
    _build_sample(out, n_volumes=4, slices_per_volume=3, size=64)
    return out
