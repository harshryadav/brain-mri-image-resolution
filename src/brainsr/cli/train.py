"""``brainsr-train``: one entrypoint, five experiments.

Picks up a YAML config (which can ``defaults: base.yaml``), applies any
``--override key=value`` pairs (dotted paths like ``data.scale=2`` work,
values are parsed as YAML so ``true``/``1e-4`` do the right thing), then
hands off to ``trainer.fit``.

Examples::

    python -m brainsr.cli.train --config configs/e2_srcnn.yaml
    python -m brainsr.cli.train --config configs/e5_agunet_attn_dcgan.yaml \\
        --override epochs=10 batch_size=8 output_dir=runs/_debug
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import MRISliceDataset
from ..models.registry import build_critic, build_model
from ..trainer import TrainerConfig, fit
from ..utils.seed import set_seed


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_override(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return out


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_full_config(path: Path) -> dict[str, Any]:
    """Load a config and resolve a single optional ``defaults: <path>`` entry."""
    cfg = _load_config(path)
    if "defaults" in cfg:
        base_path = (path.parent / cfg.pop("defaults")).resolve()
        base = _load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def _build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    common = dict(
        root=data_cfg["root"],
        scale=data_cfg.get("scale", cfg.get("scale", 4)),
        sigma_range=tuple(data_cfg.get("sigma_range", [0.5, 2.0])),
    )
    train_ds = MRISliceDataset(**common, split="train", deterministic_lr=False)
    val_ds = MRISliceDataset(**common, split="val", deterministic_lr=True)
    test_ds = MRISliceDataset(**common, split="test", deterministic_lr=True)

    bs = int(cfg.get("batch_size", 32))
    nw = int(cfg.get("num_workers", 2))
    pin = torch.cuda.is_available()
    persistent = nw > 0
    return (
        DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=nw,
            pin_memory=pin, drop_last=True, persistent_workers=persistent,
        ),
        DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=pin, persistent_workers=persistent,
        ),
        DataLoader(
            test_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=pin, persistent_workers=persistent,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a super-resolution model")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--override", nargs="*", default=[], help="Dotted overrides: e.g. epochs=5 data.scale=2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    cfg_path = Path(args.config)
    cfg = load_full_config(cfg_path)
    cfg = _deep_merge(cfg, _parse_override(args.override))

    set_seed(int(cfg.get("seed", 42)))

    output_dir = Path(cfg.get("output_dir", "runs/_default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_loader, val_loader, test_loader = _build_loaders(cfg)

    scale = int(cfg.get("data", {}).get("scale", cfg.get("scale", 4)))
    model_cfg = dict(cfg.get("model", {}))
    model_name = model_cfg.pop("name")
    if model_name in {"agunet"} and "scale" not in model_cfg:
        model_cfg["scale"] = scale
    model = build_model(model_name, **model_cfg)

    critic_cfg = dict(cfg.get("critic", {}))
    critic_name = critic_cfg.pop("name", None)
    critic = build_critic(critic_name, **critic_cfg) if critic_name else None

    tcfg = TrainerConfig(
        output_dir=output_dir,
        epochs=int(cfg.get("epochs", 100)),
        lr=float(cfg.get("lr", 1e-3)),
        lr_step_size=int(cfg.get("lr_step_size", 20)),
        lr_gamma=float(cfg.get("lr_gamma", 0.5)),
        loss=str(cfg.get("loss", "mse")),
        scale=scale,
        mixed_precision=bool(cfg.get("mixed_precision", False)),
        log_every=int(cfg.get("log_every", 50)),
        save_plots_every=int(cfg.get("save_plots_every", 5)),
        grad_clip=float(cfg.get("grad_clip", 0.0)),
        critic_lr=float(cfg.get("critic_lr", 1e-4)),
        critic_steps=int(cfg.get("critic_steps", 1)),
        critic_intensity=float(cfg.get("critic_intensity", 0.1)),
        critic_kind=str(cfg.get("critic_kind", "bce")),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )

    summary = fit(model, train_loader, val_loader, test_loader, tcfg, critic=critic)
    print(json.dumps({"output_dir": str(output_dir), "test": summary.get("test", {})}, indent=2))


if __name__ == "__main__":
    main()
