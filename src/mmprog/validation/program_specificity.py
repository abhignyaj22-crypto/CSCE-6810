"""Program specificity: does the morphology prediction track the program, or any gene set like it?

The question
------------
A morphology model can reach a respectable correlation with a gene-program score by
recovering total transcript abundance, cell density or cell identity, because every
expression score shares those. This validator asks whether the prediction tracks the
program specifically, with two checks that need no new data:

1. Expression-matched random gene sets. For each program, ``n_sets`` random gene sets are
   drawn from the same expression bins as the program's genes (one random gene per
   program gene, from the bin of that gene, outside the program), carrying the program's
   own weights. This is the control-gene-set construction of Tirosh et al. (2016, Science
   352:189-196, DOI 10.1126/science.aad0501; the same idea Seurat's AddModuleScore uses).
   Each random set is scored on the same cells, at the same scale (cell, or the 30 um
   neighbourhood mean) with the same neighbourhood construction as the real targets, and
   the out-of-fold prediction's per-patient Pearson with the real program is compared with
   its Pearson against the random sets (z-score and empirical p, per patient).
2. Split-gene halves. The program's weighted genes are split at random into two halves
   and each half is scored; a coordinated program should be tracked through either half,
   so both correlations should be positive and of similar size.

What is scored
--------------
The learned targets are NMF usages, which a random gene set cannot be pushed through, so
every set, the real program included, is scored as the weight-normalised mean of the
log-normalised expression of its genes (a linear proxy of the usage). The correlation of
the prediction with the real NMF target is reported next to the proxy so the reader sees
how close the proxy is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse

from ..logging_utils import say, show_table
from ..programs.targets_builder import neighbourhood_mean


def expression_bins(gene_mean: np.ndarray, n_bins: int = 25) -> np.ndarray:
    """Equal-size bins of genes ranked by mean expression (the binning score_lineages uses)."""
    order = np.argsort(gene_mean, kind="stable")
    bins = np.empty(len(gene_mean), dtype=np.int64)
    bins[order] = np.minimum(np.arange(len(gene_mean)) * n_bins // max(len(gene_mean), 1), n_bins - 1)
    return bins


PRESENCE_REL_WEIGHT = 0.01     # a gene counts as a program gene above this fraction of the program's maximum weight


def program_gene_indices(weights: np.ndarray, usable: np.ndarray, min_rel_weight: float = PRESENCE_REL_WEIGHT,
                         weight_mass: float = 0.9, max_genes: int = 100) -> np.ndarray:
    """Indices of the genes that carry the program.

    Usable in every sample, above ``min_rel_weight`` of the maximum weight, and, in descending
    weight order, no more than needed to reach ``weight_mass`` of the total weight (at most
    ``max_genes``). An unseeded NMF program can spread small weights over hundreds of genes;
    the random-set null needs spare genes in every expression bin, so the set is the genes that
    carry the bulk of the program, not its long tail.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.max() <= 0:
        return np.zeros(0, dtype=np.int64)
    cand = np.where(usable & (w > min_rel_weight * w.max()))[0]
    order = cand[np.argsort(-w[cand], kind="stable")]
    cum = np.cumsum(w[order]) / max(w[order].sum(), 1e-12)
    n_keep = int(np.searchsorted(cum, weight_mass) + 1)
    return np.sort(order[: max(2, min(n_keep, max_genes))])


def matched_random_sets(prog_idx: np.ndarray, prog_w: np.ndarray, bins: np.ndarray, usable: np.ndarray,
                        n_sets: int, rng: np.random.Generator, exclude: Optional[np.ndarray] = None) -> csr_matrix:
    """(G, n_sets) sparse weight matrix: each column is a random gene set matched bin by bin to the program.

    Every program gene is replaced by a random usable gene from the same expression bin,
    never a program gene, and inherits that program gene's weight. ``exclude`` is the
    boolean mask of every gene the program weights materially (above the presence
    threshold of its maximum weight, the long tail beyond the 90 % mass cap included), so
    a random set never contains a program gene; without it only the core genes in
    ``prog_idx`` are excluded. The NMF weight floor is numerically positive on every gene,
    so "non-zero weight" would exclude the whole panel. Bins with no spare gene fall back
    to the nearest bin with one.
    """
    n_genes = len(bins)
    in_prog = np.zeros(n_genes, dtype=bool)
    in_prog[prog_idx] = True
    if exclude is not None:
        in_prog |= np.asarray(exclude, dtype=bool)
    pools: Dict[int, np.ndarray] = {}
    for b in np.unique(bins):
        pools[b] = np.where((bins == b) & usable & ~in_prog)[0]
    rows, cols, vals = [], [], []
    for s in range(n_sets):
        for g, w in zip(prog_idx, prog_w):
            b = int(bins[g])
            pool = pools.get(b, np.zeros(0, dtype=np.int64))
            if len(pool) == 0:
                # nearest bin with a spare gene
                for d in range(1, bins.max() + 2):
                    for bb in (b - d, b + d):
                        if bb in pools and len(pools[bb]):
                            pool = pools[bb]
                            break
                    if len(pool):
                        break
            if len(pool) == 0:
                continue
            rows.append(int(rng.choice(pool)))
            cols.append(s)
            vals.append(float(w))
    W = csr_matrix((np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(n_genes, n_sets))
    return W


def weighted_mean_scores(X: csr_matrix, W: csr_matrix) -> np.ndarray:
    """(N, S) weight-normalised mean of log expression for every column of ``W``."""
    col_sum = np.asarray(W.sum(axis=0)).ravel()
    S = np.asarray((X @ W).todense()) if issparse(X @ W) else np.asarray(X @ W)
    return (S / np.maximum(col_sum, 1e-9)[None, :]).astype(np.float32)


def pearson_columns(pred: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Pearson of the vector ``pred`` with every column of ``M`` (nan where a column is constant)."""
    p = pred - pred.mean()
    Z = M - M.mean(axis=0, keepdims=True)
    num = p @ Z
    den = np.sqrt((p @ p) * (Z * Z).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def program_specificity(adata, gene_mask: np.ndarray, program_weights: pd.DataFrame, obs: pd.DataFrame, cfg,
                        um_by_sample: Dict[str, float], preds: np.ndarray, targets: Sequence[str], Y: np.ndarray,
                        groups: np.ndarray, test_targets: Sequence[str], n_sets: int, seed: int,
                        min_cells: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(per_patient_table, per_target_summary)`` for the learned targets in ``test_targets``."""
    rng = np.random.default_rng(seed)
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    X = X.tocsr()
    usable = np.asarray(gene_mask, dtype=bool)
    gene_mean = np.asarray(X.mean(axis=0)).ravel()
    bins = expression_bins(gene_mean)
    var_names = pd.Index(adata.var_names)
    H = program_weights.reindex(columns=var_names, fill_value=0.0)
    patients = np.unique(groups)
    say(f"program specificity: {len(test_targets)} targets, {n_sets} expression-matched random sets each, "
        f"{len(patients)} patients, {int(usable.sum())} usable genes in {bins.max() + 1} expression bins")
    rows: List[dict] = []
    for t in test_targets:
        if not t.startswith("prog_") or t not in targets:
            continue
        core = t[len("prog_"):-len("_z")]
        scale = "nb" if core.endswith("_nb") else "cell"
        program = core[: -len("_" + scale)]
        if program not in H.index:
            say(f"{t}: program {program!r} not in the weight table; skipped", indent=1)
            continue
        w_full = H.loc[program].to_numpy(dtype=np.float64)
        idx = program_gene_indices(w_full, usable)
        if len(idx) < 2:
            say(f"{t}: fewer than 2 usable weighted genes; skipped", indent=1)
            continue
        w = w_full[idx]
        j = list(targets).index(t)
        pred = np.asarray(preds[:, j], dtype=np.float64)
        ok_pred = np.isfinite(pred)
        # columns: 0 real proxy, 1 half A, 2 half B, 3.. random sets
        perm = rng.permutation(len(idx))
        half_a, half_b = idx[perm[: len(idx) // 2]], idx[perm[len(idx) // 2:]]
        wa, wb = w_full[half_a], w_full[half_b]
        n_g = X.shape[1]
        W_real = csr_matrix((w.astype(np.float32), (idx, np.zeros(len(idx), dtype=int))), shape=(n_g, 1))
        W_a = csr_matrix((wa.astype(np.float32), (half_a, np.zeros(len(half_a), dtype=int))), shape=(n_g, 1))
        W_b = csr_matrix((wb.astype(np.float32), (half_b, np.zeros(len(half_b), dtype=int))), shape=(n_g, 1))
        spare = int((usable & ~np.isin(np.arange(X.shape[1]), idx)).sum())
        if spare < len(idx):
            say(f"{t}: the program spans {len(idx)} of {int(usable.sum())} usable genes; too few spare genes for an "
                f"expression-matched null (random sets will be incomplete)", indent=1)
        # exclude every gene the program weights materially (above 1 % of its maximum weight, the same
        # presence threshold as program_gene_indices), long tail included; the NMF floor below that is
        # numerically non-zero on every gene and must stay available as spare genes for the null
        material = w_full > PRESENCE_REL_WEIGHT * float(w_full.max())
        W_null = matched_random_sets(idx, w, bins, usable, n_sets, rng, exclude=material)
        if W_null.nnz == 0:
            say(f"{t}: no expression-matched random set could be built ({int(material.sum())} program genes of "
                f"{int(usable.sum())} usable leave no spare genes); specificity not computable for this target", indent=1)
        W_all = hstack([W_real, W_a, W_b, W_null]).tocsr()
        S = weighted_mean_scores(X, W_all)                     # (N, 3 + n_sets) at the cell scale
        if scale == "nb":
            S = neighbourhood_mean(S, obs, cfg, um_by_sample)
        say(f"{t}: {len(idx)} weighted genes (halves {len(half_a)} / {len(half_b)}), scored at the {scale} scale", indent=1)
        for p in patients:
            m = (groups == p) & ok_pred
            if m.sum() <= min_cells:
                continue
            r_all = pearson_columns(pred[m], S[m])
            r_target = pearson_columns(pred[m], Y[m, j: j + 1].astype(np.float64))[0]
            null = r_all[3:]
            null = null[np.isfinite(null)]
            r_real = r_all[0]
            if len(null) < 5 or not np.isfinite(r_real):
                z, pe, q95 = np.nan, np.nan, np.nan
            else:
                # the null spread is floored at the sampling noise of a correlation on n cells (1/sqrt(n)): a null
                # narrower than that (small panels, heavily overlapping random sets) must not inflate z
                sd = max(float(null.std(ddof=1)), 1.0 / np.sqrt(float(m.sum())))
                z = (r_real - null.mean()) / sd
                pe = (1.0 + np.sum(null >= r_real)) / (len(null) + 1.0)
                q95 = float(np.quantile(null, 0.95))
            rows.append(dict(target=t, program=program, scale=scale, patient=str(p), n_cells=int(m.sum()), n_genes=len(idx),
                             r_target=r_target, r_proxy=r_real, null_mean=float(null.mean()) if len(null) else np.nan,
                             null_sd=float(null.std(ddof=1)) if len(null) > 1 else np.nan, null_q95=q95,
                             margin_over_null_q95=(r_real - q95) if np.isfinite(q95) else np.nan, z=z, p_empirical=pe,
                             r_half_a=r_all[1], r_half_b=r_all[2], n_random_sets=int(len(null))))
    table = pd.DataFrame(rows)
    if table.empty:
        return table, pd.DataFrame()
    summ = (table.groupby("target")
            .agg(program=("program", "first"), scale=("scale", "first"), n_patients=("patient", "nunique"),
                 median_r_target=("r_target", "median"), median_r_proxy=("r_proxy", "median"),
                 median_null_mean=("null_mean", "median"), median_z=("z", "median"),
                 median_margin_over_null_q95=("margin_over_null_q95", "median"),
                 fraction_patients_p_below_0_05=("p_empirical", lambda s: float(np.mean(s < 0.05)) if s.notna().any() else np.nan),
                 median_r_half_a=("r_half_a", "median"), median_r_half_b=("r_half_b", "median"))
            .reset_index())
    both = table.assign(both=(table["r_half_a"] > 0) & (table["r_half_b"] > 0)).groupby("target")["both"].mean()
    summ["fraction_patients_both_halves_positive"] = summ["target"].map(both).astype(float)
    show_table(summ.round(3), title="program specificity (per-patient medians; z against expression-matched random sets):")
    return table, summ


def specificity_headline(summ: pd.DataFrame, main_target: Optional[str]) -> dict:
    if summ.empty:
        return {}
    row = summ[summ["target"] == main_target] if main_target in set(summ["target"]) else summ.iloc[:1]
    r = row.iloc[0]
    return dict(main_target=str(r["target"]), median_z=float(r["median_z"]), median_r_target=float(r["median_r_target"]),
                median_r_proxy=float(r["median_r_proxy"]), median_null_mean=float(r["median_null_mean"]),
                median_margin_over_null_q95=float(r["median_margin_over_null_q95"]),
                fraction_patients_p_below_0_05=float(r["fraction_patients_p_below_0_05"]),
                median_r_half_a=float(r["median_r_half_a"]), median_r_half_b=float(r["median_r_half_b"]),
                targets_with_median_z_above_2=int((summ["median_z"] > 2).sum()), targets_tested=int(len(summ)))
