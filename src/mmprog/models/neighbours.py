"""Spatial neighbour tables and coordinate normalisation.

Both functions work per slide: a cell's neighbours are always on its own slide,
and coordinates are normalised to [-1, 1] inside each slide so the spatial prior
sees the same coordinate range whatever the physical size of the trephine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..logging_utils import say


def knn_table(obs: pd.DataFrame, k: int) -> np.ndarray:
    """Return an (N, k) int64 table of the k nearest same-slide neighbours of each cell.

    Row order follows ``obs``. When a slide has fewer than k + 1 cells the missing
    slots are filled with the farthest available neighbour, so every row has k
    valid indices and the Laplacian term never indexes out of range.
    """
    n = len(obs)
    table = np.zeros((n, k), dtype=np.int64)
    samples = obs["sample"].values
    xy_all = obs[["x", "y"]].values.astype(np.float32)
    for s in pd.unique(samples):
        idx = np.where(samples == s)[0]
        kk = min(k + 1, len(idx))
        if kk < 2:
            # a one-cell slide has no neighbour but itself
            table[idx, :] = idx[:, None]
            continue
        _, nbr = NearestNeighbors(n_neighbors=kk).fit(xy_all[idx]).kneighbors(xy_all[idx])
        found = idx[nbr[:, 1:]]                       # drop self (column 0)
        table[idx, : kk - 1] = found
        if kk - 1 < k:
            table[idx, kk - 1:] = found[:, -1:]       # pad with the farthest neighbour
    say(f"knn table built: {n} cells, k={k}, {len(pd.unique(samples))} slides")
    return table


def norm_xy(obs: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) float32 coordinates scaled to [-1, 1] within each slide."""
    xy = obs[["x", "y"]].values.astype(np.float32).copy()
    samples = obs["sample"].values
    for s in pd.unique(samples):
        m = samples == s
        for c in (0, 1):
            lo = xy[m, c].min()
            span = np.ptp(xy[m, c]) + 1e-6
            xy[m, c] = (xy[m, c] - lo) / span * 2.0 - 1.0
    return xy
