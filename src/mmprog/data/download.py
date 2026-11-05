"""Downloading and unpacking the raw GEO supplementary files.

Only the three files the pipeline needs are fetched per sample: the cell by
gene matrix, the cell table with centroids, and the morphology image.
``transcripts.parquet`` and the boundary files are several gigabytes each and
are never used, so they are skipped.

Every transfer and every decompression writes to a ``.part`` file and is
renamed into place only when complete. A Colab session that drops mid-transfer
therefore never leaves a complete-looking but truncated file behind; the next
run sees the ``.part`` and resumes it with ``wget -c``.
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
import tarfile
import time
from typing import Any, Dict, List, Optional

from ..config import Config
from ..logging_utils import say

_WANTED = re.compile(r"cell_feature_matrix|cells\.csv|cells\.parquet|morphology|\.tar|\.tif", re.I)
_UNWANTED = re.compile(r"transcripts|boundar", re.I)
_IMAGE = re.compile(r"morphology|\.tif", re.I)

# Preference order for the morphology file. Xenium exports may ship several:
#   morphology_mip  : already max-projected DAPI, the cheapest to use
#   morphology      : the DAPI z-stack (projected by load_image)
#   morphology_focus: XOA 2.x multi-channel (DAPI + 3 stains); channel 0 is DAPI
_IMAGE_PREFERENCE = [
    ("mip", r"morphology_mip[^/]*\.(ome\.)?tif{1,2}(\.gz)?$"),
    ("zstack", r"morphology\.(ome\.)?tif{1,2}(\.gz)?$"),
    ("focus", r"morphology_focus[^/]*\.(ome\.)?tif{1,2}(\.gz)?$"),
    ("other", r"morphology[^/]*\.tif{1,2}(\.gz)?$"),
]


def _pick(files: List[str], pattern: str) -> Optional[str]:
    matches = sorted(f for f in files if re.search(pattern, os.path.basename(f), re.I))
    return matches[0] if matches else None


def pick_image(files: List[str]) -> Dict[str, Optional[str]]:
    """Choose the morphology file by explicit preference and report which kind it is."""
    for kind, pattern in _IMAGE_PREFERENCE:
        path = _pick(files, pattern)
        if path:
            return dict(image=path, image_kind=kind)
    return dict(image=None, image_kind=None)


def download_file(url: str, dst: str, wget: Optional[List[str]] = None) -> str:
    """Fetch ``url`` to ``dst`` through ``dst.part`` with resume; rename on completion."""
    if os.path.exists(dst):
        return dst
    part = dst + ".part"
    t0 = time.time()
    resumed = os.path.exists(part)
    say(f"{'resuming' if resumed else 'downloading'} {os.path.basename(dst)}"
        + (f" from {os.path.getsize(part) / 1e9:.2f} GB" if resumed else ""), indent=1)
    cmd = (wget or ["wget", "-q", "-c", "-O"]) + [part, url]
    subprocess.run(cmd, check=True)
    os.replace(part, dst)  # atomic: a complete file never exists under the final name until it is complete
    say(f"got {os.path.basename(dst)} ({os.path.getsize(dst) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", indent=1)
    return dst


def _is_image_name(name: str) -> bool:
    return bool(_IMAGE.search(name))


def fetch_sample_files(row: Dict[str, Any], cfg: Config, need_image: bool = True) -> Dict[str, Optional[str]]:
    """Download what the sample needs into ``cfg.local_data/<sample>`` and locate the files.

    Matrix and cell table come first, the image last, so a caller that decides to
    exclude the sample after reading the matrix can skip the multi-GB image by
    calling with ``need_image=False`` first and again with ``need_image=True``.
    """
    sample_dir = os.path.join(cfg.local_data, row["sample"])
    os.makedirs(sample_dir, exist_ok=True)
    wanted = [u for u in row.get("supp", []) if _WANTED.search(os.path.basename(u))
              and not _UNWANTED.search(os.path.basename(u))]
    ordered = sorted(wanted, key=lambda u: _is_image_name(os.path.basename(u)))  # non-image files first
    for url in ordered:
        name = os.path.basename(url)
        if not need_image and _is_image_name(name):
            say(f"skipping image download for {row['sample']} (not needed yet)", indent=1)
            continue
        dst = download_file(url, os.path.join(sample_dir, name))
        if re.search(r"\.tar(\.gz)?$", name) and not os.path.exists(dst + ".extracted"):
            say(f"extracting {name}", indent=1)
            with tarfile.open(dst) as tar:
                tar.extractall(sample_dir)
            open(dst + ".extracted", "w").close()  # marker so re-runs skip the extraction

    files = [os.path.join(r, f) for r, _, fs in os.walk(sample_dir) for f in fs
             if not f.endswith(".part")]
    found: Dict[str, Optional[str]] = dict(
        h5=_pick(files, r"cell_feature_matrix\.h5$"),
        cells=_pick(files, r"cells\.(csv\.gz|csv|parquet)$"),
    )
    found.update(pick_image(files))
    say(f"{row['sample']}: h5={bool(found['h5'])} cells={bool(found['cells'])} "
        f"image={os.path.basename(found['image']) if found['image'] else None} (kind {found['image_kind']})")
    return found


def ensure_raw(row: Dict[str, Any], cfg: Config, need_image: bool = True) -> Dict[str, Any]:
    """Re-download whatever is missing on disk (the Colab disk is wiped on restart)."""
    keys = ("h5", "cells", "image") if need_image else ("h5", "cells")
    if all(row.get(k) and os.path.exists(row[k]) for k in keys):
        return row
    say(f"raw files missing for {row['sample']}, re-downloading"
        + ("" if need_image else " (matrix and cells only)"))
    got = fetch_sample_files(row, cfg, need_image=need_image)
    row.update({k: v for k, v in got.items() if v})
    return row


def ensure_decompressed(path: str) -> str:
    """Gunzip ``path`` once to disk if it ends in ``.gz``; return the readable path.

    GEO ships ``morphology.ome.tif.gz``. tifffile needs random access, so the
    file is streamed through gzip in 64 MB blocks (constant RAM) to a sibling
    file, via ``.part`` so an interrupted gunzip is never mistaken for a whole
    one. The caller may delete that copy after cropping; the ``.gz`` stays.
    """
    if not path.endswith(".gz"):
        return path
    out = path[:-3]
    if not os.path.exists(out):
        part = out + ".part"
        t0 = time.time()
        with gzip.open(path, "rb") as fi, open(part, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
        os.replace(part, out)
        say(f"gunzipped {os.path.basename(out)} ({os.path.getsize(out) / 1e9:.1f} GB) in {time.time() - t0:.0f}s")
    return out
