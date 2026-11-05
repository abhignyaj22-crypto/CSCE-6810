"""Synthetic CODEX-like fixture built from the synthetic cohort's planted lineages.

The real CODEX stage needs a marker table and a DAPI image of the same section.
This module fakes the marker table from ``lineage_true`` of one synthetic slide
(lognormal intensities, high for the lineage's own markers) and reuses the
slide's synthetic DAPI image, so the whole bridge can be tested end to end
without any proteomics data.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tifffile

from ..config import Config
from ..logging_utils import say
from .codex import load_marker_mapping

# Which canonical markers light up for each planted RNA lineage of the synthetic cohort.
_LINEAGE_MARKERS: Dict[str, list] = {
    "plasma": ["CD138", "CD38", "CD56"],
    "T": ["CD3", "CD4", "CD8"],
    "NK": ["CD56"],
    "B": ["CD20", "CD19", "PAX5"],
    "granulocyte": ["MPO", "CD15"],
    "monocyte": ["CD68", "CD14", "CD163"],
    "erythroid": ["CD71", "GlycophorinA"],
    "megakaryocyte": ["CD61", "CD41"],
    "stromal": ["SMA", "CD34", "Vimentin", "CollagenIV"],
}


def make_demo_codex(obs_one_sample: pd.DataFrame, demo_image: np.ndarray, lineages_true: np.ndarray,
                    cfg: Config, seed: int = 7, n_markers_extra: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return ``(table, dapi)``: a cell-by-marker table in microns and the DAPI image.

    ``obs_one_sample`` holds ``x, y`` in working-level pixels (the synthetic frame);
    coordinates are converted to microns with ``cfg.um_per_px`` so the loader's
    micron convention is exercised.
    """
    rng = np.random.default_rng(seed)
    mapping = load_marker_mapping()
    markers = list(mapping["aliases"].keys())
    markers = [m for m in markers if m != "DAPI"]
    n = len(obs_one_sample)
    # background: lognormal, mean around 20; positive markers: around 200
    X = rng.lognormal(mean=np.log(20.0), sigma=0.5, size=(n, len(markers))).astype(np.float32)
    col = {m: i for i, m in enumerate(markers)}
    for lin, ms in _LINEAGE_MARKERS.items():
        rows = np.where(lineages_true == lin)[0]
        for m in ms:
            if m in col:
                X[rows, col[m]] = rng.lognormal(mean=np.log(200.0), sigma=0.4, size=len(rows))
    table = pd.DataFrame(X, columns=markers)
    # use alias spellings for a few columns so alias matching is exercised
    table = table.rename(columns={"CD138": "Syndecan-1", "CD3": "CD3e", "SMA": "aSMA"})
    table.insert(0, "Centroid Y", (obs_one_sample["y"].values * cfg.um_per_px).astype(np.float32))
    table.insert(0, "Centroid X", (obs_one_sample["x"].values * cfg.um_per_px).astype(np.float32))
    table.insert(0, "Object ID", [f"obj{i}" for i in range(n)])
    table["Area"] = rng.uniform(30, 120, size=n).astype(np.float32)   # a non-marker column to be dropped
    say(f"synthetic CODEX table: {n} cells x {len(markers)} markers (+ geometry columns), coordinates in microns")
    return table, np.asarray(demo_image, dtype=np.uint8)


def write_demo_codex(out_dir: str, obs_one_sample: pd.DataFrame, demo_image: np.ndarray,
                     lineages_true: np.ndarray, cfg: Config, seed: int = 7) -> Tuple[str, str]:
    """Write ``codex_demo.csv`` and ``codex_demo_dapi.tif`` into ``out_dir``; return the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    table, dapi = make_demo_codex(obs_one_sample, demo_image, lineages_true, cfg, seed)
    csv_path = os.path.join(out_dir, "codex_demo.csv")
    tif_path = os.path.join(out_dir, "codex_demo_dapi.tif")
    table.to_csv(csv_path, index=False)
    tifffile.imwrite(tif_path, dapi)
    say(f"synthetic CODEX written: {csv_path}, {tif_path}")
    return csv_path, tif_path
