"""A short research note at each transformation.

Every note is printed, then appended under ``<run_dir>/observations/`` so a later
reading can see what went in, what came out, and a few concrete examples.
Image crops also save up to five original-versus-crop pictures.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import say, warn

MAX_IMAGE_PAIRS = 5


def observe(run_dir: str, transformation: str, before: str, after: str,
            note: str = "", examples: Optional[Sequence[Any]] = None) -> None:
    """Print one transformation and append it to ``observations/<name>.jsonl``."""
    shown = list(examples or [])[:5]
    say(f"observation [{transformation}]")
    say(f"before: {before}", indent=1)
    say(f"after:  {after}", indent=1)
    if note:
        say(f"note:   {note}", indent=1)
    for ex in shown:
        say(f"example: {ex}", indent=1)
    folder = os.path.join(run_dir, "observations")
    os.makedirs(folder, exist_ok=True)
    record = dict(time=datetime.now().isoformat(timespec="seconds"), transformation=transformation,
                  before=before, after=after, note=note, examples=[str(e) for e in shown])
    with open(os.path.join(folder, f"{transformation}.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")
    say(f"record: {os.path.join(folder, transformation + '.jsonl')}", indent=1)


def record_image_crop(cfg, sample: str, crop: np.ndarray, bbox: Tuple[int, int, int, int],
                      full_hw: Tuple[int, int], tif_path: Optional[str] = None) -> None:
    """Record the field-to-crop cut. Save a picture while fewer than five exist and the TIFF is still on disk."""
    x0, y0, x1, y1 = bbox
    full_h, full_w = full_hw
    kept = (x1 - x0) * (y1 - y0) / float(max(full_w * full_h, 1))
    picture = ""
    folder = os.path.join(cfg.run_dir, "observations", "image_crop")
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.endswith(".png")]
    if tif_path and os.path.exists(tif_path) and len(existing) < MAX_IMAGE_PAIRS:
        try:
            picture = _save_pair(folder, sample, crop, bbox, full_hw, tif_path)
        except Exception as exc:
            warn(f"image comparison for {sample} was not saved: {exc}")
    observe(cfg.run_dir, "image_crop",
            before=f"{sample} level-1 field {full_w} x {full_h} px",
            after=f"crop {x1 - x0} x {y1 - y0} px at ({x0}, {y0}), {crop.shape[0]} x {crop.shape[1]} stored, {kept:.0%} of the field",
            note="DAPI max projection. The crop is the only image later components see. "
                 + (f"picture {picture}" if picture else "no picture this slide (five already saved, or the raw TIFF was already removed)"),
            examples=[f"{sample} origin ({x0}, {y0}) size {x1 - x0}x{y1 - y0}"])


def _save_pair(folder: str, sample: str, crop: np.ndarray, bbox, full_hw, tif_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = bbox
    full_h, full_w = full_hw
    with tifffile.TiffFile(tif_path) as tif:
        levels = list(tif.series[0].levels)
        thumb_level = min(5, len(levels) - 1)
        field = np.asarray(levels[thumb_level].asarray()).max(axis=0)
    lo, hi = np.percentile(field, [1, 99.5])
    show = np.clip((field.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1)
    sy, sx = show.shape[0] / full_h, show.shape[1] / full_w
    step = max(1, int(np.ceil(max(crop.shape) / 900)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=120)
    axes[0].imshow(show, cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy,
                                fill=False, edgecolor="#e4572e", linewidth=1.5))
    axes[0].set_title(f"original field\n{full_w} x {full_h} px")
    axes[1].imshow(crop[::step, ::step], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"crop kept for the model\n{x1 - x0} x {y1 - y0} px")
    for ax in axes:
        ax.set_axis_off()
    kept = (x1 - x0) * (y1 - y0) / float(max(full_w * full_h, 1))
    fig.suptitle(f"{sample}   {kept:.0%} of the field kept", fontsize=13)
    fig.tight_layout()
    out = os.path.join(folder, f"{sample}_original_vs_crop.png")
    fig.savefig(out)
    plt.close(fig)
    say(f"saved original-versus-crop picture {out}", indent=1)
    return out
