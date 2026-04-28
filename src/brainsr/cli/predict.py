"""``brainsr-predict``: run a trained model over cached ``.npy`` slices.

Loads ``best.pt`` (or ``last.pt``) from a run directory, generates LR using
the same degradation seen in training, and writes both the SR ``.npy`` and a
side-by-side ``LR | SR | HR`` PNG for each input.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.degradation import Degradation
from ..models.registry import build_model
from ..utils.viz import save_triplet_grid

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Super-resolve cached .npy slices")
    parser.add_argument("--run-dir", required=True, type=str, help="Directory with config.resolved.yaml and best.pt")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory of .npy HR slices to degrade and SR")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    scale = int(cfg.get("data", {}).get("scale", cfg.get("scale", 4)))

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    if model_name == "agunet" and "scale" not in model_cfg:
        model_cfg["scale"] = scale
    model = build_model(model_name, **model_cfg)

    ckpt = torch.load(run_dir / args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    degrade = Degradation(scale=scale, sigma_range=tuple(cfg.get("data", {}).get("sigma_range", [0.5, 2.0])), deterministic=True)

    output_dir = Path(args.output_dir)
    (output_dir / "panels").mkdir(parents=True, exist_ok=True)
    (output_dir / "sr_npy").mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.input_dir).glob("*.npy"))
    if args.limit:
        files = files[: args.limit]

    with torch.no_grad():
        for path in files:
            hr_arr = np.clip(np.load(path).astype(np.float32), 0.0, 1.0)
            hr = torch.from_numpy(hr_arr).unsqueeze(0).unsqueeze(0)
            lr = degrade(hr)
            x = lr
            if getattr(model, "needs_bicubic_input", False):
                x = torch.nn.functional.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
            sr = model(x).clamp(0.0, 1.0)
            np.save(output_dir / "sr_npy" / path.name, sr.squeeze().cpu().numpy().astype(np.float32))
            save_triplet_grid(lr, sr, hr, output_dir / "panels" / (path.stem + ".png"), title=path.stem)
    log.info("Wrote %d outputs to %s", len(files), output_dir)


if __name__ == "__main__":
    main()
