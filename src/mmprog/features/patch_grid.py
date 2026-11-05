"""Spatially located patch features for spatially regularized boosting.

The boosting algorithm of Xiang et al. (NIPS 2009, Advances in Neural Information Processing Systems 22) needs features that have a
known location, because its regulariser rewards selecting weak learners that lie
close to already selected ones. A ResNet embedding has no such geometry. The
patch itself does: the 64 px (27 um) window around a nucleus is divided into a
``grid x grid`` lattice of blocks, and each block contributes its mean and its
standard deviation of intensity. Every feature therefore has a position in
microns relative to the cell centre, and the learned importance map can be read
as "which part of the neighbourhood carries the program signal": the nucleus
itself (centre blocks), the immediate neighbours (middle ring), or the wider
surroundings (outer ring).

Feature layout: index ``k`` in ``[0, grid*grid)`` is the mean of block ``k`` (row
major), index ``grid*grid + k`` is the standard deviation of the same block. Both
share the block's location; the kernel treats them as co-located.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import say, timed
from .morphometrics import _to_gray_uint8
from .patches import extract_patch


def grid_locations_um(grid: int, patch_px: int, um_per_px: float) -> np.ndarray:
    """``(2*grid*grid, 2)`` block-centre coordinates in microns, cell centre at the origin."""
    block = patch_px / grid
    centres = (np.arange(grid) + 0.5) * block - patch_px / 2
    yy, xx = np.meshgrid(centres, centres, indexing="ij")
    loc = np.stack([xx.ravel(), yy.ravel()], axis=1) * um_per_px
    return np.vstack([loc, loc]).astype(np.float32)


def grid_feature_names(grid: int) -> list:
    names = [f"mean_r{i}_c{j}" for i in range(grid) for j in range(grid)]
    names += [f"std_r{i}_c{j}" for i in range(grid) for j in range(grid)]
    return names


def _block_stats(patch: np.ndarray, grid: int) -> Tuple[np.ndarray, np.ndarray]:
    p = patch.shape[0]
    blocks = patch.reshape(grid, p // grid, grid, p // grid).astype(np.float32)
    return blocks.mean(axis=(1, 3)).ravel(), blocks.std(axis=(1, 3)).ravel()


def patch_grid_features(obs: pd.DataFrame, images, cfg: Config, grid: int = 8) -> np.ndarray:
    """``(N, 2*grid*grid)`` float32 block means and standard deviations for every cell."""
    assert cfg.patch_px % grid == 0, f"patch_px {cfg.patch_px} must be divisible by grid {grid}"
    feats = np.zeros((len(obs), 2 * grid * grid), dtype=np.float32)
    samples = obs["sample"].to_numpy()
    with timed(f"patch grid features ({grid}x{grid} blocks of {cfg.patch_px // grid} px) for {len(obs)} cells"):
        for sample in pd.unique(samples):
            idx = np.where(samples == sample)[0]
            gray = _to_gray_uint8(images.get(sample))
            ox, oy = images.offset(sample)
            xy = obs[["x", "y"]].to_numpy()[idx] - np.array([ox, oy])
            for k, i in enumerate(idx):
                patch = extract_patch(gray, xy[k, 0], xy[k, 1], cfg.patch_px)[..., 0]
                m, s = _block_stats(patch, grid)
                feats[i, : grid * grid] = m
                feats[i, grid * grid :] = s
            say(f"{sample}: {len(idx)} cells gridded", indent=1)
    say(f"patch grid features shape {feats.shape}; block size {cfg.patch_px // grid} px = "
        f"{cfg.patch_px // grid * cfg.um_per_px:.1f} um")
    return feats
