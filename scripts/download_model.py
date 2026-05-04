"""Optional fallback: pull bundled checkpoints from a remote URL.

Primary distribution is the ``checkpoints/`` folder shipped in the repo /
zip. This script only matters when that folder is missing entirely - for
example, a user who cloned from a fresh git checkout without the LFS-like
binary blobs.

Configuration mirrors ``download_data.py``:

1. ``--url`` CLI flag.
2. ``CHECKPOINTS_URL`` env var.
3. Hard-coded default at the top of this file.

The remote artifact must be a ``.tar.gz`` (or ``.tar`` / ``.zip``)
containing one folder per experiment, each with ``best.pt`` and
``config.resolved.yaml``. See ``checkpoints/README.md`` for the layout.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# Reuse the helpers from download_data so we keep one source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import _download_from_gdrive, _download_http, _parse_gdrive_id  # type: ignore  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS_URL: str = ""


def _has_any_experiment(ckpt_dir: Path) -> bool:
    if not ckpt_dir.exists():
        return False
    return any(p.is_dir() and (p / "config.resolved.yaml").exists() for p in ckpt_dir.iterdir())


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", archive, target)
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        raise SystemExit(
            f"Unsupported archive type for {archive}. Use .tar.gz, .tgz, .tar, or .zip."
        )


def _flatten_into(ckpt_dir: Path, scratch: Path) -> None:
    """Move extracted contents so ``ckpt_dir`` directly contains the eN_* folders.

    Handles both layouts:
      - ``scratch/checkpoints/<exp>/...``  (preferred, mirrors repo layout)
      - ``scratch/<exp>/...``              (already flat)
    """
    nested = scratch / "checkpoints"
    src = nested if nested.exists() else scratch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in src.iterdir():
        target = ckpt_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
        moved += 1
    log.info("Moved %d entries into %s", moved, ckpt_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bundled model checkpoints (fallback)")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    out_dir = Path(args.output_dir)
    if _has_any_experiment(out_dir) and not args.force:
        log.info("Checkpoints already present at %s. Use --force to redownload.", out_dir)
        return

    url = args.url or os.environ.get("CHECKPOINTS_URL") or DEFAULT_CHECKPOINTS_URL
    if not url:
        sys.exit(
            "No checkpoints URL configured and no checkpoints found locally. "
            "Either ship the `checkpoints/` folder with the repo, set CHECKPOINTS_URL "
            "in .env, or pass --url."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir.parent / "_ckpt_download_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    suffix = Path(urlparse(url).path).suffix or ".tar.gz"
    archive = scratch / f"checkpoints{suffix}"

    gdrive_id = _parse_gdrive_id(url)
    if gdrive_id:
        _download_from_gdrive(gdrive_id, archive)
    else:
        _download_http(url, archive)

    extract_target = scratch / "extracted"
    _extract(archive, extract_target)
    _flatten_into(out_dir, extract_target)
    shutil.rmtree(scratch, ignore_errors=True)

    if not _has_any_experiment(out_dir):
        sys.exit(
            f"Download finished but no eN_*/config.resolved.yaml found under {out_dir}. "
            "Check the tarball layout."
        )
    log.info("Done. Checkpoints ready at %s", out_dir)


if __name__ == "__main__":
    main()
