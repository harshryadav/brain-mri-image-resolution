"""Single training loop shared across E1-E5.

Picks the best available device (CUDA -> MPS -> CPU), builds Adam optimizers
for the generator (and optional critic), runs train/val every epoch, and
saves ``best.pt`` / ``last.pt`` plus ``summary.json``. The bicubic baseline
(E1) skips training entirely and just runs eval.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.degradation import upsample_bicubic
from .losses import AdversarialLoss, reconstruction_loss
from .metrics import MetricBank
from .models.bicubic import BicubicUpsampler
from .utils.viz import save_triplet_grid

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    output_dir: Path
    epochs: int = 100
    lr: float = 1e-3
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    loss: str = "mse"
    scale: int = 4
    mixed_precision: bool = False
    log_every: int = 50
    save_plots_every: int = 5
    grad_clip: float = 0.0
    critic_lr: float = 1e-4
    critic_steps: int = 1
    critic_intensity: float = 0.1
    critic_kind: str = "bce"
    label_smoothing: float = 0.0
    save_best: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _device() -> torch.device:
    """Pick the best available device, with optional ``BRAINSR_DEVICE`` override.

    Priority: explicit env var > CUDA > MPS (Apple Silicon) > CPU.
    """
    import os

    forced = os.environ.get("BRAINSR_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        if forced == "cuda" and not torch.cuda.is_available():
            log.warning("BRAINSR_DEVICE=cuda set but CUDA unavailable; falling back")
        elif forced == "mps" and not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            log.warning("BRAINSR_DEVICE=mps set but MPS unavailable; falling back")
        else:
            return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_input(model: nn.Module, lr: torch.Tensor, scale: int) -> torch.Tensor:
    if getattr(model, "needs_bicubic_input", False):
        return upsample_bicubic(lr, scale=scale)
    return lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
) -> dict[str, float]:
    model.eval()
    bank = MetricBank(data_range=1.0, device=device)
    for batch in loader:
        lr, hr = batch[0], batch[1]
        lr, hr = lr.to(device), hr.to(device)
        x = _prepare_input(model, lr, scale)
        sr = model(x).clamp(0.0, 1.0)
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = torch.nn.functional.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        bank.update(sr, hr)
    return bank.compute().as_dict()


@torch.no_grad()
def _save_sample_panels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
    out_dir: Path,
    tag: str,
    n: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        x = _prepare_input(model, lr, scale)
        sr = model(x).clamp(0.0, 1.0)
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = torch.nn.functional.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        for b in range(min(lr.size(0), n - saved)):
            save_triplet_grid(
                lr[b : b + 1], sr[b : b + 1], hr[b : b + 1],
                out_dir / f"{tag}_{saved:02d}.png",
                title=tag,
            )
            saved += 1
            if saved >= n:
                return
        if saved >= n:
            return


def _evaluate_only_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainerConfig,
) -> dict[str, Any]:
    """For E1 (bicubic): no training; just evaluate val/test once."""
    device = _device()
    model = model.to(device).eval()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    val_metrics = evaluate(model, val_loader, device, cfg.scale)
    test_metrics = evaluate(model, test_loader, device, cfg.scale)
    for k, v in val_metrics.items():
        writer.add_scalar(f"val/{k}", v, 0)
    for k, v in test_metrics.items():
        writer.add_scalar(f"test/{k}", v, 0)
    writer.close()

    _save_sample_panels(model, val_loader, device, cfg.scale, output_dir / "samples", "val")

    summary = {"val": val_metrics, "test": test_metrics, "epoch": 0}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainerConfig,
    critic: nn.Module | None = None,
) -> dict[str, Any]:
    """Train ``model`` (and optionally ``critic``) and return summary metrics."""
    if isinstance(model, BicubicUpsampler):
        return _evaluate_only_pipeline(model, train_loader, val_loader, test_loader, cfg)

    device = _device()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    log.info("Trainer config: %s", asdict(cfg))
    log.info("Device: %s", device)

    model = model.to(device)
    rec_loss = reconstruction_loss(cfg.loss).to(device)
    g_optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    g_sched = torch.optim.lr_scheduler.StepLR(g_optim, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    g_scaler = GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    use_critic = critic is not None
    if use_critic:
        critic = critic.to(device)
        adv = AdversarialLoss(kind=cfg.critic_kind, label_smoothing=cfg.label_smoothing).to(device)
        c_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, betas=(0.5, 0.999))
        c_scaler = GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    best_psnr = -float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, Any]] = []

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if use_critic:
            critic.train()

        epoch_t0 = time.time()
        epoch_g_loss = 0.0
        epoch_c_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            lr, hr = batch[0].to(device), batch[1].to(device)

            if use_critic:
                for _ in range(cfg.critic_steps):
                    with autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == "cuda"):
                        with torch.no_grad():
                            x = _prepare_input(model, lr, cfg.scale)
                            sr = model(x).clamp(0.0, 1.0)
                            if sr.shape[-2:] != hr.shape[-2:]:
                                sr = torch.nn.functional.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
                        real_logits = critic(hr)
                        fake_logits = critic(sr)
                        c_loss = adv.critic_loss(real_logits, fake_logits)
                    c_optim.zero_grad(set_to_none=True)
                    c_scaler.scale(c_loss).backward()
                    if cfg.grad_clip > 0:
                        c_scaler.unscale_(c_optim)
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
                    c_scaler.step(c_optim)
                    c_scaler.update()
                epoch_c_loss += float(c_loss.detach().item())

            with autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == "cuda"):
                x = _prepare_input(model, lr, cfg.scale)
                sr = model(x)
                if sr.shape[-2:] != hr.shape[-2:]:
                    sr = torch.nn.functional.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
                rec = rec_loss(sr, hr)
                g_loss = rec
                if use_critic:
                    fake_logits = critic(sr.clamp(0.0, 1.0))
                    g_adv = adv.generator_loss(fake_logits)
                    g_loss = rec + cfg.critic_intensity * g_adv

            g_optim.zero_grad(set_to_none=True)
            g_scaler.scale(g_loss).backward()
            if cfg.grad_clip > 0:
                g_scaler.unscale_(g_optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            g_scaler.step(g_optim)
            g_scaler.update()

            epoch_g_loss += float(g_loss.detach().item())
            n_batches += 1
            global_step += 1
            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/g_loss", float(g_loss.item()), global_step)
                if use_critic:
                    writer.add_scalar("train/c_loss", float(c_loss.item()), global_step)

        g_sched.step()
        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_c = epoch_c_loss / max(n_batches, 1) if use_critic else 0.0

        val_metrics = evaluate(model, val_loader, device, cfg.scale)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/g_loss_epoch", avg_g, epoch)
        if use_critic:
            writer.add_scalar("train/c_loss_epoch", avg_c, epoch)

        elapsed = time.time() - epoch_t0
        log.info(
            "Epoch %3d/%d | g_loss %.5f | val PSNR %.3f | val SSIM %.4f | val NRMSE %.4f | %.1fs",
            epoch, cfg.epochs, avg_g, val_metrics["psnr"], val_metrics["ssim"], val_metrics["nrmse"], elapsed,
        )

        history.append({"epoch": epoch, "g_loss": avg_g, "c_loss": avg_c, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if cfg.save_best and val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, best_path)
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if cfg.save_plots_every and epoch % cfg.save_plots_every == 0:
            _save_sample_panels(model, val_loader, device, cfg.scale, output_dir / "samples", f"val_epoch{epoch:03d}")

    if cfg.save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, device, cfg.scale)
    for k, v in test_metrics.items():
        writer.add_scalar(f"test/{k}", v, cfg.epochs)
    writer.close()

    summary = {"val_best": {"psnr": best_psnr}, "test": test_metrics, "history": history}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
