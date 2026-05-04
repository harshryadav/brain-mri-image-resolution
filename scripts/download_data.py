"""Fetch the preprocessed FastMRI dataset from a configurable URL.

Designed for the demo flow: idempotent (no-op when ``data/processed/splits.json``
already exists). Supports three remote layouts, in order of preference:

1. **Single ``.tar.gz`` / ``.tar`` / ``.zip``** on Google Drive or HTTPS
   (recommended; one fast request, no per-file rate limits). When
   extracted, the archive should contain a ``processed/`` directory with
   the ``.npy`` slices and ``splits.json`` at its root - matching the
   layout ``brainsr-preprocess`` produces locally. A flat archive
   (no ``processed/`` wrapper) also works.
2. **Google Drive folder URL** (e.g. ``https://drive.google.com/drive/folders/<id>``).
   Falls back to ``gdown.download_folder`` which downloads each file
   individually. Slower and more rate-limit-prone, but works without
   re-uploading anything.
3. **Plain HTTPS URL to a single archive** - same as (1) over HTTP.

If ``splits.json`` is missing after the download, we regenerate it
deterministically (seed=42, 70/20/10) from the .npy filenames so the
demo can still run.

Configuration (priority order): ``--url`` flag > ``PROCESSED_DATA_URL``
env var > ``DEFAULT_PROCESSED_DATA_URL`` constant in this file.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

# Replace this with your actual Google Drive share URL or file ID once the
# preprocessed tarball is uploaded. Until then, set PROCESSED_DATA_URL in .env.
DEFAULT_PROCESSED_DATA_URL: str = ""


def _is_gdrive_folder(url: str) -> bool:
    """True if the URL points at a Google Drive *folder* (vs a single file)."""
    if not url:
        return False
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return False
    return "/folders/" in parsed.path


def _parse_gdrive_id(url: str) -> str | None:
    """Extract a Google Drive *file* ID from any of the common URL shapes.

    Returns None for folder URLs (use :func:`_is_gdrive_folder` for those).
    """
    if not url:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url):
        return url
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None
    if "/folders/" in parsed.path:
        return None
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    if m:
        return m.group(1)
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    return None


def _download_gdrive_folder(url: str, dest_dir: Path) -> None:
    """Download every file from a Google Drive folder URL into ``dest_dir``."""
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required for Google Drive downloads. Install with `pip install gdown`."
        ) from e
    log.warning(
        "Downloading Google Drive *folder* via gdown - this is slow and prone to "
        "rate-limit errors with many files. Prefer uploading a single .tar.gz."
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("gdown.download_folder: %s -> %s", url, dest_dir)
    gdown.download_folder(url=url, output=str(dest_dir), quiet=False, use_cookies=False)
    if not any(dest_dir.iterdir()):
        raise SystemExit(
            f"gdown produced no files at {dest_dir}. "
            "Confirm the folder is shared with 'Anyone with the link' and try again."
        )


def _download_from_gdrive(file_id: str, dest: Path) -> Path:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required for Google Drive downloads. Install with `pip install gdown`."
        ) from e
    log.info("Downloading via gdown: id=%s -> %s", file_id, dest)
    gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit(
            f"gdown produced no file at {dest}. "
            "Confirm the share link is set to 'Anyone with the link' and try again."
        )
    return dest


def _download_http(url: str, dest: Path) -> Path:
    import urllib.request

    log.info("Downloading via HTTP: %s -> %s", url, dest)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


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


def _flatten_into(processed_dir: Path, scratch: Path) -> None:
    """Move extracted contents so ``processed_dir`` directly contains the slices.

    Handles both layouts:
      - ``scratch/processed/<files>``  (preferred)
      - ``scratch/<files>``            (already flat)
    """
    nested = scratch / "processed"
    src = nested if nested.exists() else scratch
    processed_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in src.iterdir():
        target = processed_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
        moved += 1
    log.info("Moved %d entries into %s", moved, processed_dir)


def _ensure_splits(processed_dir: Path) -> None:
    """If splits.json wasn't shipped with the data, regenerate it deterministically."""
    splits_path = processed_dir / "splits.json"
    if splits_path.exists():
        return
    n_npy = sum(1 for _ in processed_dir.glob("*.npy"))
    if n_npy == 0:
        sys.exit(
            f"Download finished but no .npy slices found in {processed_dir}. "
            "Check the upload contents."
        )
    log.warning(
        "splits.json not present; regenerating deterministic 70/20/10 split (seed=42) "
        "over %d .npy slices.", n_npy,
    )
    # Lazy import so the script stays runnable without the brainsr package installed.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from brainsr.data.splits import build_splits  # type: ignore  # noqa: E402

    build_splits(processed_dir, train=0.70, val=0.20, test=0.10, seed=42)
    log.info("Wrote %s", splits_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download preprocessed FastMRI cache")
    parser.add_argument("--url", type=str, default=None, help="Override the data URL or Google Drive file ID")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--force", action="store_true", help="Re-download even if data is already present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    out_dir = Path(args.output_dir)
    if (out_dir / "splits.json").exists() and not args.force:
        log.info("Data already present at %s (splits.json exists). Use --force to redownload.", out_dir)
        return

    url = args.url or os.environ.get("PROCESSED_DATA_URL") or DEFAULT_PROCESSED_DATA_URL
    if not url:
        sys.exit(
            "No data URL configured. Set PROCESSED_DATA_URL in .env, pass --url, "
            "or edit DEFAULT_PROCESSED_DATA_URL in scripts/download_data.py."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if _is_gdrive_folder(url):
        # Folder mode: gdown writes files directly into the target.
        if args.force and out_dir.exists():
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _download_gdrive_folder(url, out_dir)
        _ensure_splits(out_dir)
        log.info("Done. Processed data ready at %s", out_dir)
        return

    # Single-archive mode (Drive file or plain HTTPS).
    scratch = out_dir.parent / "_download_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    gdrive_id = _parse_gdrive_id(url)
    if gdrive_id:
        archive = scratch / "processed.tar.gz"
        _download_from_gdrive(gdrive_id, archive)
    else:
        suffix = Path(urlparse(url).path).suffix or ".tar.gz"
        archive = scratch / f"processed{suffix}"
        _download_http(url, archive)

    extract_target = scratch / "extracted"
    _extract(archive, extract_target)
    _flatten_into(out_dir, extract_target)
    shutil.rmtree(scratch, ignore_errors=True)

    _ensure_splits(out_dir)
    log.info("Done. Processed data ready at %s", out_dir)


if __name__ == "__main__":
    main()
