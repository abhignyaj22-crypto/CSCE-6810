"""Exports for pathology tools and batch inference over a slide folder (Phase 5).

- QuPath: tab-separated detections with x_um, y_um and one ``Measurement:`` column per program.
- Xenium Explorer: cell_id, x_centroid, y_centroid (microns) and one column per program.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from ..logging_utils import say, section, show_table


def _pred_columns(vmap: pd.DataFrame) -> List[str]:
    """Prediction columns to export: program means, their standard deviations and lineage probabilities."""
    return [c for c in vmap.columns if c.endswith("_pred") or c.endswith("_std") or c.startswith("lineage_prob_")]


def export_qupath(vmap: pd.DataFrame, path: str, um_per_px: float) -> None:
    df = pd.DataFrame({"x_um": vmap["x"] * um_per_px, "y_um": vmap["y"] * um_per_px})
    for c in _pred_columns(vmap):
        df[f"Measurement: {c}"] = vmap[c].values
    df.to_csv(path, sep="\t", index=False)
    say(f"QuPath detections written: {path} ({len(df)} rows, {len(df.columns)} columns)")


def export_xenium_explorer(vmap: pd.DataFrame, path: str, um_per_px: float) -> None:
    df = pd.DataFrame({
        "cell_id": [f"n{i}" for i in range(len(vmap))],
        "x_centroid": vmap["x"].values * um_per_px,
        "y_centroid": vmap["y"].values * um_per_px,
    })
    for c in _pred_columns(vmap):
        df[c] = vmap[c].values
    df.to_csv(path, index=False)
    say(f"Xenium Explorer table written: {path} ({len(df)} rows)")


def batch_inference(
    slide_dir: str, model: Any, fs: Any, selector: Any, cfg: Any, device: str, targets: List[str],
    n_samples: int, run: Any, main_target: str, burden_threshold: float,
    load_image_fn: Optional[Callable] = None, vmap_fn: Optional[Callable] = None, encoder: Optional[Any] = None,
) -> pd.DataFrame:
    """Score every ``*.tif*`` in ``slide_dir``; write per-slide CSVs and ``batch_burden.csv``."""
    section("S7b batch inference")
    if load_image_fn is None:
        from ..data.images import load_image as load_image_fn
    if vmap_fn is None:
        from .virtual_map import virtual_program_map as vmap_fn
    paths = sorted(glob.glob(os.path.join(slide_dir, "*.tif*")))
    say(f"{len(paths)} slides found in {slide_dir}")
    col = main_target.replace("_z", "_pred")
    rows = []
    for path in paths:
        sid = os.path.splitext(os.path.basename(path))[0]
        img = load_image_fn(path, cfg.img_level, None, cfg.z_project, cfg.stain_mode)
        vm = vmap_fn(img, model, fs, selector, cfg, device, targets, n_samples, encoder=encoder)
        vm.to_csv(run.art(f"vmap_{sid}.csv"), index=False)
        export_qupath(vm, run.art(f"vmap_{sid}_qupath.tsv"), cfg.um_per_px)
        export_xenium_explorer(vm, run.art(f"vmap_{sid}_xe.csv"), cfg.um_per_px)
        burden = float((vm[col] > burden_threshold).mean()) if len(vm) else np.nan
        rows.append(dict(slide=sid, n_nuclei=len(vm), burden=burden))
        say(f"{sid}: {len(vm)} nuclei, burden {burden:.3f}", indent=1)
    out = pd.DataFrame(rows, columns=["slide", "n_nuclei", "burden"])
    out.to_csv(run.art("batch_burden"), index=False)
    show_table(out, title="batch burden:")
    return out
