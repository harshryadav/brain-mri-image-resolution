"""``brainsr-eval``: collect each run's ``summary.json`` into a single CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def collect(runs_dir: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        cfg_path = run_dir / "config.resolved.yaml"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping %s: %s", run_dir, e)
            continue
        test = summary.get("test", {})
        row = {
            "run": run_dir.name,
            "psnr": float(test.get("psnr", float("nan"))),
            "ssim": float(test.get("ssim", float("nan"))),
            "nrmse": float(test.get("nrmse", float("nan"))),
            "config": str(cfg_path) if cfg_path.exists() else "",
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run summaries into a CSV")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--output", type=str, default="runs/results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = collect(Path(args.runs_dir))
    if not rows:
        raise SystemExit(f"No summaries found under {args.runs_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "psnr", "ssim", "nrmse", "config"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote %d rows to %s", len(rows), output)
    print(f"{'run':<40} {'PSNR':>7} {'SSIM':>7} {'NRMSE':>7}")
    for r in rows:
        print(f"{r['run']:<40} {r['psnr']:>7.3f} {r['ssim']:>7.4f} {r['nrmse']:>7.4f}")


if __name__ == "__main__":
    main()
