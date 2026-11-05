"""Cell-level quality control and count normalisation.

These are the three scanpy functions the pipeline used to depend on
(``calculate_qc_metrics``, cell filtering, ``normalize_total`` + ``log1p``),
re-implemented on the sparse count matrix so the package does not pull in
scanpy. Raw counts are preserved in ``adata.layers["counts"]`` because the
targets and the reliability estimate need integer counts, not log values.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from ..config import Config
from ..logging_utils import say, show_table


def _as_csr(adata: ad.AnnData) -> csr_matrix:
    """Return X as CSR without copying when it already is one."""
    if issparse(adata.X):
        return adata.X.tocsr()
    return csr_matrix(adata.X)


def calculate_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Add ``total_counts`` and ``n_genes_by_counts`` to ``adata.obs`` in place.

    Works on raw counts. If the matrix has already been log-normalised the
    totals are meaningless, so we warn when values look non-integer.
    """
    X = _as_csr(adata)
    total = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.diff(X.indptr)  # number of stored (non-zero) entries per row
    adata.obs["total_counts"] = total.astype(np.float32)
    adata.obs["n_genes_by_counts"] = n_genes.astype(np.int32)
    if X.nnz > 0 and not np.allclose(X.data[: min(1000, X.nnz)], np.round(X.data[: min(1000, X.nnz)])):
        say("WARNING: calculate_qc_metrics called on a matrix that does not look like raw counts")
    say(f"QC metrics: {adata.n_obs} cells, median transcripts {np.median(total):.0f}, "
        f"median genes {np.median(n_genes):.0f}")
    return adata


def apply_cell_qc(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Drop cells below the transcript and gene floors; print the per-sample tally.

    The floors (60 transcripts, 50 genes by default) follow the Xenium 5K QC used
    for the trephine cohort: cells below them are usually segmentation fragments
    or red-cell ghosts and would add noise to every count-based target.
    """
    if "total_counts" not in adata.obs:
        calculate_qc_metrics(adata)
    keep = (adata.obs["total_counts"].values >= cfg.min_transcripts) & \
           (adata.obs["n_genes_by_counts"].values >= cfg.min_genes)
    n_before = adata.n_obs
    tally = pd.DataFrame({
        "sample": adata.obs["sample"].values if "sample" in adata.obs else "all",
        "kept": keep,
    }).groupby("sample")["kept"].agg(cells="size", kept="sum")
    tally["dropped"] = tally["cells"] - tally["kept"]
    out = adata[keep].copy()
    dropped = adata.obs.loc[~keep, ["sample", "total_counts", "n_genes_by_counts"]].head(5)
    kept_rows = out.obs.loc[:, ["sample", "total_counts", "n_genes_by_counts"]].head(5)
    from ..observations import observe
    observe(cfg.run_dir, "cell_qc",
            before=f"{n_before} cells before the transcript and gene floors",
            after=f"{out.n_obs} cells kept, {n_before - out.n_obs} dropped",
            note=f"floors are {cfg.min_transcripts} transcripts and {cfg.min_genes} genes",
            examples=[f"kept {r['sample']} counts={r['total_counts']:.0f} genes={r['n_genes_by_counts']:.0f}"
                      for r in kept_rows.to_dict("records")]
                     + [f"dropped {r['sample']} counts={r['total_counts']:.0f} genes={r['n_genes_by_counts']:.0f}"
                        for r in dropped.to_dict("records")])
    say(f"cell QC (>= {cfg.min_transcripts} transcripts, >= {cfg.min_genes} genes): "
        f"kept {out.n_obs}/{n_before} ({100 * out.n_obs / max(n_before, 1):.1f}%)")
    show_table(tally, title="QC tally per sample")
    if out.n_obs:
        med = out.obs.groupby("sample")[["total_counts", "n_genes_by_counts"]].median() \
            if "sample" in out.obs else out.obs[["total_counts", "n_genes_by_counts"]].median().to_frame().T
        show_table(med, title="median transcripts and genes per cell after QC")
    return out


def normalize_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """Library-size normalise each cell to ``target_sum`` and apply log1p.

    Raw counts are copied to ``adata.layers["counts"]`` first (once). The
    normalised matrix is only used for lineage marker scoring; targets are built
    from the raw counts so that split-half thinning stays valid.
    """
    X = _as_csr(adata).astype(np.float32)
    if "counts" not in adata.layers:
        adata.layers["counts"] = X.copy()
        say("raw counts stored in adata.layers['counts']")
    total = np.asarray(X.sum(axis=1)).ravel()
    scale = target_sum / np.maximum(total, 1.0)
    X = csr_matrix(X.multiply(scale[:, None]))
    X.data = np.log1p(X.data)
    adata.X = X
    say(f"normalised to {target_sum:.0f} counts per cell and log1p-transformed "
        f"({adata.n_obs} cells x {adata.n_vars} genes)")
    return adata
