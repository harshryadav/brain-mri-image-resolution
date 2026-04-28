"""Maps short config names (``bicubic``, ``srcnn``, ``agunet``, ``dcgan``) to constructors."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .agunet import AGUNet
from .bicubic import BicubicUpsampler
from .dcgan_critic import DCGANCritic
from .srcnn import SRCNN

_GENERATORS: dict[str, type[nn.Module]] = {
    "bicubic": BicubicUpsampler,
    "srcnn": SRCNN,
    "agunet": AGUNet,
}

_CRITICS: dict[str, type[nn.Module]] = {
    "dcgan": DCGANCritic,
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower()
    if name in _GENERATORS:
        return _GENERATORS[name](**kwargs)
    raise ValueError(f"Unknown model {name!r}. Available: {sorted(_GENERATORS)}")


def build_critic(name: str | None, **kwargs: Any) -> nn.Module | None:
    if name is None or str(name).lower() in {"none", "off", ""}:
        return None
    n = name.lower()
    if n in _CRITICS:
        return _CRITICS[n](**kwargs)
    raise ValueError(f"Unknown critic {name!r}. Available: {sorted(_CRITICS)} or 'none'")
