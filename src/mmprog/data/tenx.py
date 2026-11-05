"""Readers for 10x Genomics outputs (Xenium and Visium HD) without scanpy.

The 10x HDF5 matrix format stores a CSC matrix of genes x cells under the
``matrix`` group: ``data``, ``indices``, ``indptr``, ``shape`` (genes, cells),
``barcodes`` and ``features/{name,id}``. Transposing that CSC gives the CSR
cells x genes matrix anndata expects, with no densification.
"""
from __future__ import annotations

from typing import Any, Dict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from ..config import Config
from ..logging_utils import say


def _decode(arr: np.ndarray) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def read_10x_h5(path: str) -> ad.AnnData:
    """Read a 10x ``*.h5`` matrix into an AnnData with CSR float32 counts."""
    with h5py.File(path, "r") as fh:
        g = fh["matrix"]
        shape = tuple(int(v) for v in g["shape"][:])  # (n_genes, n_cells)
        data = g["data"][:].astype(np.float32)
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        barcodes = _decode(g["barcodes"][:])
        names = _decode(g["features"]["name"][:])
        ids = _decode(g["features"]["id"][:]) if "id" in g["features"] else names
    genes_by_cells = csc_matrix((data, indices, indptr), shape=shape)
    X = genes_by_cells.T.tocsr()
    var = pd.DataFrame({"gene_ids": ids}, index=pd.Index(names, name="gene"))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(barcodes, name="cell_id")), var=var)
    n_dup = int(var.index.duplicated().sum())
    if n_dup:
        adata.var_names_make_unique()
        say(f"made {n_dup} duplicate gene names unique", indent=1)
    say(f"read 10x h5 {path}: {adata.n_obs} cells x {adata.n_vars} genes, nnz {X.nnz}")
    return adata


def h5_matrix_shape(path: str) -> tuple[int, int]:
    """(n_cells, n_genes) from the h5 header alone, without reading the matrix.

    Used to exclude small-panel samples before their multi-gigabyte image is downloaded.
    """
    with h5py.File(path, "r") as fh:
        n_genes, n_cells = (int(v) for v in fh["matrix"]["shape"][:])
    return n_cells, n_genes


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def load_xenium(row: Dict[str, Any], cfg: Config) -> ad.AnnData:
    """Xenium cell matrix + ``cells`` table; centroids converted to working-level pixels."""
    adata = read_10x_h5(row["h5"])
    cells = _read_table(row["cells"]).set_index("cell_id").loc[adata.obs_names]
    um_per_px_l0 = float(row.get("um_per_px_l0") or cfg.um_per_px_level0)
    px_per_um = 1.0 / (um_per_px_l0 * 2 ** cfg.img_level)
    adata.obs["x"] = (cells["x_centroid"].values * px_per_um).astype(np.float32)
    adata.obs["y"] = (cells["y_centroid"].values * px_per_um).astype(np.float32)
    for key in ("sample", "patient", "stage"):
        adata.obs[key] = row[key]
    say(f"xenium {row['sample']}: centroids scaled by {px_per_um:.3f} px/um "
        f"(level {cfg.img_level}); x range [{adata.obs['x'].min():.0f}, {adata.obs['x'].max():.0f}], "
        f"y range [{adata.obs['y'].min():.0f}, {adata.obs['y'].max():.0f}]")
    return adata


def load_visium_hd(row: Dict[str, Any], cfg: Config) -> ad.AnnData:
    """Visium HD binned outputs: bins are treated as cells downstream."""
    adata = read_10x_h5(row["h5"])
    pos = _read_table(row["cells"]).set_index("barcode").loc[adata.obs_names]
    scale = float(row.get("scalefactor", 1.0)) / (2 ** cfg.img_level)
    adata.obs["x"] = (pos["pxl_col_in_fullres"].values * scale).astype(np.float32)
    adata.obs["y"] = (pos["pxl_row_in_fullres"].values * scale).astype(np.float32)
    if "in_tissue" in pos:
        n0 = adata.n_obs
        adata = adata[pos["in_tissue"].values == 1].copy()
        say(f"visium hd {row['sample']}: kept {adata.n_obs}/{n0} bins in tissue", indent=1)
    for key in ("sample", "patient", "stage"):
        adata.obs[key] = row[key]
    return adata


def load_row(row: Dict[str, Any], cfg: Config) -> ad.AnnData:
    """Dispatch on the row's platform and annotate cohort/platform."""
    platform = row.get("platform") or cfg.platform
    adata = load_xenium(row, cfg) if platform == "xenium" else load_visium_hd(row, cfg)
    adata.obs["platform"] = platform
    adata.obs["cohort"] = row.get("cohort", f"{platform}_default")
    return adata
