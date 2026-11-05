"""Save five original-slide versus pipeline-crop comparisons.

The pipeline deletes each morphology file after the crop is cached. This script
fetches a slide back only when the TIFF is already gone, reads a coarse pyramid
level for the full field, and places the cached crop beside it.
"""
from __future__ import annotations

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle

ROOT = "/home/aj0486@students.ad.unt.edu/CB"
RAW = "/dev/shm/mmprog_raw"
CROP_DIR = os.path.join(ROOT, "workspace", "crops")
OUT = os.path.join(ROOT, "workspace", "comparisons")
THUMB_LEVEL = 5  # level 1 is the crop's coordinate frame; each step halves it

SLIDES = [
    ("GSM9035023", "Ctrl-1", "control"),
    ("GSM9035024", "Ctrl-2", "control"),
    ("GSM9035025", "Ctrl-3", "control"),
    ("GSM9035026", "Ctrl-4", "control"),
    ("GSM9035027", "MGUS-1", "MGUS"),
]


def slide_paths(gsm: str, tag: str):
    folder = os.path.join(RAW, gsm)
    os.makedirs(folder, exist_ok=True)
    name = f"{gsm}_{tag}_morphology.ome.tif"
    return folder, os.path.join(folder, name + ".gz"), os.path.join(folder, name)


def ensure_tiff(gsm: str, tag: str) -> str:
    folder, gz, tif = slide_paths(gsm, tag)
    if not os.path.exists(tif):
        if not os.path.exists(gz):
            url = (f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9035nnn/{gsm}/suppl/"
                   f"{gsm}_{tag}_morphology.ome.tif.gz")
            print(f"downloading {os.path.basename(gz)}", flush=True)
            subprocess.run(["wget", "-q", "-c", "-O", gz + ".part", url], check=True)
            os.replace(gz + ".part", gz)
        print(f"gunzipping {os.path.basename(gz)}", flush=True)
        subprocess.run(["gzip", "-dc", gz], check=True, stdout=open(tif + ".part", "wb"))
        os.replace(tif + ".part", tif)
    return tif


def full_field(tif_path: str):
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        level1 = series.levels[1]
        thumb = series.levels[THUMB_LEVEL]
        field = np.asarray(thumb.asarray()).max(axis=0)
        full_hw = (int(level1.shape[-2]), int(level1.shape[-1]))  # y, x at the crop's level
    lo, hi = np.percentile(field, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    show = np.clip((field.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return show, full_hw


def load_crop(gsm: str):
    path = os.path.join(CROP_DIR, f"{gsm}_L1_zmax_m64_native.npz")
    z = np.load(path)
    return z["img"], tuple(int(v) for v in z["bbox"])


def shrink(img: np.ndarray, long_side: int = 900) -> np.ndarray:
    step = max(1, int(np.ceil(max(img.shape) / long_side)))
    return img[::step, ::step]


def one(gsm: str, tag: str, stage: str) -> str:
    tif_path = ensure_tiff(gsm, tag)
    field, (full_h, full_w) = full_field(tif_path)
    crop, (x0, y0, x1, y1) = load_crop(gsm)
    sy, sx = field.shape[0] / full_h, field.shape[1] / full_w
    kept = (x1 - x0) * (y1 - y0) / float(full_w * full_h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=120)
    axes[0].imshow(field, cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy,
                                fill=False, edgecolor="#e4572e", linewidth=1.5))
    axes[0].set_title(f"original field, level 1\n{full_w} x {full_h} px\nbox is the crop")
    axes[1].imshow(shrink(crop), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"pipeline crop\n{x1 - x0} x {y1 - y0} px, origin ({x0}, {y0})\n{kept:.0%} of the field")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{gsm}  {tag}  {stage}   DAPI, max projection", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, f"{gsm}_{tag}_original_vs_crop.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  kept {kept:.1%}", flush=True)
    # The pipeline gunzips again from the .gz. Drop the multi-GB TIFF now.
    if os.path.exists(tif_path):
        os.remove(tif_path)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    paths = [one(*slide) for slide in SLIDES]
    print("saved", len(paths), "comparisons")


if __name__ == "__main__":
    main()
