"""Per-cell program counts, neighbourhood sums and local density.

Everything here is computed on EVERY QC-passing cell of a slide, before any
subsampling for the image model. Two reasons:

1. A 78,000-cell biopsy subsampled to 30,000 cells would look 2.6 times sparser
   than it is, which corrupts every density feature (the v3 bug).
2. The neighbourhood target for a kept cell must sum over all of its real
   neighbours, not only those that survived the subsample.

Counts are summed, never averaged, so that the split-half reliability estimate
(binomial thinning of integer counts) remains valid downstream.
"""
from __future__ import annotations

import time
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import KDTree

from ..config import Config
from ..logging_utils import say


def _counts_matrix(adata: ad.AnnData) -> csr_matrix:
    """Raw counts: the ``counts`` layer if normalisation already ran, else X."""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return X.tocsr() if issparse(X) else csr_matrix(X)


def count_summaries(adata_full: ad.AnnData, programs: Dict[str, List[str]], cfg: Config,
                    px_per_um: float | None = None) -> pd.DataFrame:
    """Compute count summaries for all cells of one slide.

    Returns a DataFrame indexed by ``obs_names`` with columns
    ``cnt_total, cnt_<program>, nb_n, nb_cnt_total, nb_cnt_<program>, density_r<r>``.
    ``nb_*`` sums run over every cell within ``cfg.nb_radius_um`` (including the
    cell itself); ``density_r<r>`` counts other cells within ``r`` pixels.
    ``px_per_um`` overrides the configured pixel size when the slide's OME metadata
    reports a different physical pixel size.
    """
    if px_per_um is None:
        px_per_um = cfg.px_per_um
    t0 = time.time()
    X = _counts_matrix(adata_full)
    n = adata_full.n_obs
    out = pd.DataFrame(index=adata_full.obs_names)
    out["cnt_total"] = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)

    for name, genes in programs.items():
        idx = [adata_full.var_names.get_loc(g) for g in genes if g in adata_full.var_names]
        if idx:
            out[f"cnt_{name}"] = np.asarray(X[:, idx].sum(axis=1)).ravel().astype(np.float32)
        else:
            out[f"cnt_{name}"] = np.float32(0.0)
        say(f"program {name}: {len(idx)}/{len(genes)} genes in panel, "
            f"mean {out[f'cnt_{name}'].mean():.2f} transcripts per cell", indent=1)

    xy = adata_full.obs[["x", "y"]].values.astype(np.float32)
    tree = KDTree(xy)
    radius_px = cfg.nb_radius_um * px_per_um
    neighbours = tree.query_radius(xy, r=radius_px)

    # Sparse adjacency A[i, j] = 1 when j lies within the radius of i.
    rows = np.repeat(np.arange(n), [len(nb) for nb in neighbours])
    cols = np.concatenate(neighbours) if n else np.zeros(0, dtype=np.int64)
    A = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    out["nb_n"] = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    out["nb_cnt_total"] = (A @ out["cnt_total"].values).astype(np.float32)
    for name in programs:
        out[f"nb_cnt_{name}"] = (A @ out[f"cnt_{name}"].values).astype(np.float32)

    for r in cfg.density_radii_px:
        out[f"density_r{r}"] = (tree.query_radius(xy, r=r, count_only=True) - 1).astype(np.float32)

    say(f"count summaries for {n} cells: neighbourhood radius {cfg.nb_radius_um} um = {radius_px:.1f} px, "
        f"median neighbours {np.median(out['nb_n']):.0f}, "
        f"median density_r{cfg.density_radii_px[0]} {np.median(out[f'density_r{cfg.density_radii_px[0]}']):.0f}, "
        f"{time.time() - t0:.1f}s")
    return out


def attach_summaries(adata_sub: ad.AnnData, summ: pd.DataFrame, adata_full: ad.AnnData) -> ad.AnnData:
    """Join summary rows onto a (possibly subsampled, renamed) subset by exact coordinate match.

    Cell ids can be renamed when slides are concatenated, so coordinates are the
    stable key. A match farther than 1e-3 px means the subset did not come from
    this slide and is treated as a hard error.
    """
    tree = KDTree(adata_full.obs[["x", "y"]].values.astype(np.float32))
    dist, j = tree.query(adata_sub.obs[["x", "y"]].values.astype(np.float32), k=1)
    assert (dist[:, 0] < 1e-3).all(), "coordinate match failed between subsample and full slide"
    for col in summ.columns:
        adata_sub.obs[col] = summ[col].values[j[:, 0]]
    say(f"attached {len(summ.columns)} summary columns to {adata_sub.n_obs} cells "
        f"(from {adata_full.n_obs} on the full slide)")
    return adata_sub


def subsample_cells(adata: ad.AnnData, max_cells: int | None, seed: int) -> ad.AnnData:
    """Random subsample without replacement, used only for embeddings and training."""
    if not max_cells or adata.n_obs <= max_cells:
        say(f"no subsampling: {adata.n_obs} cells <= limit {max_cells}")
        return adata
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(adata.n_obs, max_cells, replace=False))
    say(f"subsampled {max_cells}/{adata.n_obs} cells for the image model (seed {seed})")
    return adata[keep].copy()
