"""Fetch the preprocessed FastMRI dataset from a configurable URL.

Idempotent: no-op when ``data/processed/splits.json`` already exists.

Supported sources, in order of recommendation:

1. **Hugging Face Hub Datasets (recommended).** Free, no quota, designed
   for exactly this. Public dataset URLs look like::

       https://huggingface.co/datasets/<user>/<repo>/resolve/main/processed.tar.gz

   The ``/blob/`` form (the page you see in your browser) is auto-rewritten
   to ``/resolve/`` (the raw bytes endpoint). Both work.

2. **GitHub Releases / Zenodo / any other public HTTPS URL** to a single
   ``.tar.gz`` / ``.tar`` / ``.zip`` archive.

3. **Google Drive folder URL** (e.g. ``https://drive.google.com/drive/folders/<id>``).
   Slow (one HTTP request per file) and rate-limit-prone; only used if you
   haven't migrated off Drive yet.

After download, if ``splits.json`` is missing it's regenerated
deterministically (seed=42, 70/20/10 by volume) from the .npy filenames so
the demo can still run.

**Why not Google Drive *files*?** For files >100MB Google forces a virus-scan
confirmation page and aggressively rate-limits scripted access. Both
``gdown`` and direct ``requests`` calls fail intermittently in practice.
Pointing this script at a Drive *file* URL now produces an explanatory
error rather than a confusing stack trace - host the tarball on HF or
GitHub Releases instead.

Configuration (priority order): ``--url`` flag > ``PROCESSED_DATA_URL``
env var > ``DEFAULT_PROCESSED_DATA_URL`` constant in this file.
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

log = logging.getLogger(__name__)

# Hard-coded last-resort fallback. Used only when neither --url nor
# PROCESSED_DATA_URL (env / .env) is set. Keeping the project's HF dataset
# here means a fresh clone with a missing .env still works out of the box.
DEFAULT_PROCESSED_DATA_URL: str = (
    "https://huggingface.co/datasets/UMaryland/brain-mri-superresolution-group11"
    "/resolve/main/processed.tar.gz"
)

# Hosts whose URLs we treat as "single-archive HTTPS download". Anything not
# matching the Drive-folder pattern lands here.
_HF_HOSTS = ("huggingface.co", "hf.co")


def _is_gdrive_folder(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return "drive.google.com" in parsed.netloc and "/folders/" in parsed.path


def _is_gdrive_file(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return False
    return "/file/" in parsed.path or "id=" in parsed.query


def _normalize_url(url: str) -> str:
    """Rewrite known web-page URLs into their raw-bytes equivalents.

    Currently handles Hugging Face's ``/blob/`` (HTML page) -> ``/resolve/``
    (raw download) substitution so users can paste either form.
    """
    parsed = urlparse(url)
    if parsed.netloc in _HF_HOSTS and "/blob/" in parsed.path:
        new_url = url.replace("/blob/", "/resolve/", 1)
        log.info("Rewrote Hugging Face URL %s -> %s", url, new_url)
        return new_url
    return url


def _download_https_archive(url: str, dest: Path) -> Path:
    """Stream a public HTTPS URL to disk in 1 MB chunks.

    Works with Hugging Face Hub, GitHub Releases, Zenodo, and any plain
    static file host. Fails loudly if we get HTML back (which usually means
    the URL points at a web page rather than the raw bytes).
    """
    import requests

    log.info("Downloading: %s -> %s", url, dest)
    headers = {"User-Agent": "brainsr-demo/1.0 (+https://github.com/)"}
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if ctype.startswith("text/html"):
            preview = r.text[:400].replace("\n", " ")
            raise SystemExit(
                "Got HTML instead of file bytes from the download URL.\n"
                f"  URL: {url}\n"
                f"  Content-Type: {ctype}\n"
                f"  Body preview: {preview!r}\n"
                "If this is a Hugging Face URL, make sure it's the /resolve/ "
                "form (not /blob/), and that the dataset repo is public. "
                "If it's a GitHub Releases URL, paste the asset's *download* URL."
            )
        total_bytes = int(r.headers.get("Content-Length") or 0)
        if total_bytes:
            log.info("Expected size: %.1f MB", total_bytes / (1024 * 1024))
        downloaded = 0
        next_log = 50 * 1024 * 1024  # log every ~50 MB
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    if total_bytes:
                        log.info(
                            "Downloaded %.1f / %.1f MB (%.0f%%)",
                            downloaded / (1024 * 1024),
                            total_bytes / (1024 * 1024),
                            100 * downloaded / total_bytes,
                        )
                    else:
                        log.info("Downloaded %.1f MB", downloaded / (1024 * 1024))
                    next_log += 50 * 1024 * 1024
    log.info("Saved %.1f MB to %s", downloaded / (1024 * 1024), dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit(f"Empty file at {dest} after download.")
    return dest


def _download_gdrive_folder(url: str, dest_dir: Path) -> None:
    """Slow fallback: download every file from a Google Drive folder.

    Only used when ``PROCESSED_DATA_URL`` points at ``drive.google.com/.../folders/``.
    For any new project, host the tarball on Hugging Face / GitHub Releases
    instead - this path is here only so existing Drive folders keep working
    without a re-upload.
    """
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required for Google Drive folder downloads. "
            "Install with `pip install gdown`."
        ) from e
    log.warning(
        "Downloading a Google Drive *folder* via gdown - this is slow and "
        "prone to rate limits with many files. For a robust demo, host the "
        "tarball on Hugging Face Hub instead and update PROCESSED_DATA_URL."
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("gdown.download_folder: %s -> %s", url, dest_dir)
    gdown.download_folder(url=url, output=str(dest_dir), quiet=False)
    if not any(dest_dir.iterdir()):
        raise SystemExit(
            f"gdown produced no files at {dest_dir}. "
            "Confirm the folder is shared with 'Anyone with the link'."
        )


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
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from brainsr.data.splits import build_splits  # type: ignore  # noqa: E402

    build_splits(processed_dir, train=0.70, val=0.20, test=0.10, seed=42)
    log.info("Wrote %s", splits_path)


def _archive_suffix_for(url: str) -> str:
    """Pick a sensible filename suffix from the URL so the extractor works."""
    path = urlparse(url).path
    for ext in (".tar.gz", ".tgz", ".tar", ".zip"):
        if path.lower().endswith(ext):
            return ext
    return ".tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download preprocessed FastMRI cache")
    parser.add_argument("--url", type=str, default=None, help="Override PROCESSED_DATA_URL")
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
    url = _normalize_url(url)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Drive folders: slow, but functional. Direct fallback for users who
    # haven't migrated to HF / GH Releases yet.
    if _is_gdrive_folder(url):
        if args.force:
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _download_gdrive_folder(url, out_dir)
        _ensure_splits(out_dir)
        log.info("Done. Processed data ready at %s", out_dir)
        return

    # Drive *files* are explicitly unsupported - the virus-scan / quota wall
    # makes them unreliable for the >100MB tarballs we ship. Fail with an
    # actionable hint instead of pretending to work.
    if _is_gdrive_file(url):
        sys.exit(
            "Google Drive *file* URLs aren't supported for the demo because Google's "
            "virus-scan confirmation page and per-file quota make large downloads "
            "fail intermittently for graders.\n\n"
            "Recommended fix: re-host the tarball on Hugging Face Hub (free, no quota):\n"
            "  1. pip install huggingface_hub\n"
            "  2. huggingface-cli login\n"
            "  3. huggingface-cli upload <user>/<repo> processed.tar.gz . --repo-type=dataset\n"
            "  4. Set PROCESSED_DATA_URL=https://huggingface.co/datasets/<user>/<repo>/resolve/main/processed.tar.gz\n\n"
            "Or use a Drive *folder* URL (slower but works): "
            "https://drive.google.com/drive/folders/<folder-id>"
        )

    # Generic HTTPS path: HF, GitHub Releases, Zenodo, etc.
    scratch = out_dir.parent / "_download_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    archive = scratch / f"processed{_archive_suffix_for(url)}"
    _download_https_archive(url, archive)

    extract_target = scratch / "extracted"
    _extract(archive, extract_target)
    _flatten_into(out_dir, extract_target)
    shutil.rmtree(scratch, ignore_errors=True)

    _ensure_splits(out_dir)
    log.info("Done. Processed data ready at %s", out_dir)


if __name__ == "__main__":
    main()
