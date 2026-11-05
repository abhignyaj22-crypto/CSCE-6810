"""Morphometric features: patch intensity statistics and local cell density.

These are the quantities a haematopathologist reads at a glance (how dark, how
variable, how crowded) and they need only centroids and the image, so the
inference stage can compute them from detected nuclei without any transcriptomics.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from skimage import color
from sklearn.neighbors import KDTree

from ..config import Config
from ..logging_utils import say, timed
from .patches import extract_patch

_PATCH_STATS = ["p_mean", "p_std", "p_p10", "p_p90", "p_darkfrac"]


def MORPHO_NAMES(cfg: Config) -> List[str]:
    return _PATCH_STATS + [f"density_r{r}" for r in cfg.density_radii_px]


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else img.astype(np.uint8)
    return (color.rgb2gray(img) * 255).astype(np.uint8)


def morphometrics(obs: pd.DataFrame, images, cfg: Config) -> np.ndarray:
    """``(N, 5 + len(radii))`` float32: patch stats then density at each radius.

    Density comes from ``obs['density_r<r>']`` when present (computed on every cell
    of the slide before subsampling); otherwise it is computed from the centroids
    available in ``obs`` and that fact is printed.
    """
    radii = list(cfg.density_radii_px)
    feats = np.zeros((len(obs), len(_PATCH_STATS) + len(radii)), dtype=np.float32)
    samples = obs["sample"].to_numpy()
    with timed(f"morphometrics for {len(obs)} cells"):
        for sample in pd.unique(samples):
            idx = np.where(samples == sample)[0]
            gray = _to_gray_uint8(images.get(sample))
            ox, oy = images.offset(sample)
            xy = obs[["x", "y"]].to_numpy()[idx] - np.array([ox, oy])
            for k, i in enumerate(idx):
                patch = extract_patch(gray, xy[k, 0], xy[k, 1], cfg.patch_px)[..., 0].astype(np.float32)
                feats[i, :5] = [patch.mean(), patch.std(), np.percentile(patch, 10), np.percentile(patch, 90), (patch < 100).mean()]
            have_full = all(f"density_r{r}" in obs for r in radii)
            if have_full:
                for j, r in enumerate(radii):
                    feats[idx, 5 + j] = obs[f"density_r{r}"].to_numpy()[idx]
                say(f"{sample}: {len(idx)} cells, density taken from full-slide summaries", indent=1)
            else:
                tree = KDTree(xy)
                for j, r in enumerate(radii):
                    feats[idx, 5 + j] = tree.query_radius(xy, r=r, count_only=True) - 1
                say(f"{sample}: {len(idx)} cells, density computed from available centroids", indent=1)
    say(f"morphometrics shape {feats.shape}; columns {MORPHO_NAMES(cfg)}")
    return feats
