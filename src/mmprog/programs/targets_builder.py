"""Targets from learned programs and from the fixed literature sets, at two spatial scales.

Two target families share one builder so the gain from learned gene weights is
measured against the fixed-set scores rather than assumed:

- ``prog_<p>_<scale>_z``   learned program usage (anchored NMF), cell scale and the mean
                            over cells within ``nb_radius_um`` (the neighbourhood scale);
- ``anchor_<p>_<scale>_z`` the v2 count-based score of the fixed gene set (``program_score``
                            on summed counts), when the count summaries are present.

Both are standardised with ``targets.programs.standardise`` (cohort or slide, D8).
``expression_embedding`` is the standardised cell-scale learned usage (N, P), the
expression-side vector the unified model aligns morphology to.

Reliability of the learned targets: raw counts are thinned per cell and per gene into
two binomial halves, each half is re-normalised and projected with the fixed H, and the
two projections are correlated per sample (Spearman-Brown corrected). This is the
learned-weight analogue of the count split-half ceiling (D9).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import KDTree

from ..config import Config
from ..logging_utils import say, section, show_table
from ..targets.programs import program_score, standardise
from ..targets.reliability import _spearman_brown
from .anchored_nmf import AnchoredNMF


def neighbourhood_mean(values: np.ndarray, obs: pd.DataFrame, cfg: Config,
                       um_per_px_by_sample: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Mean of ``values`` (N, P) over all cells within ``nb_radius_um`` of each cell, per sample.

    The radius is converted to pixels with the slide's own pixel size when ``um_per_px_by_sample``
    is given (recorded from the OME metadata at ingest), else with the configured default, so the
    physical neighbourhood matches the one used for the ingest-time count summaries.
    """
    out = np.zeros_like(values, dtype=np.float32)
    samples = obs["sample"].to_numpy()
    xy_all = obs[["x", "y"]].to_numpy().astype(np.float32)
    for s in pd.unique(samples):
        idx = np.where(samples == s)[0]
        um = (um_per_px_by_sample or {}).get(str(s), cfg.um_per_px)
        radius_px = cfg.nb_radius_um / float(um)
        tree = KDTree(xy_all[idx])
        neigh = tree.query_radius(xy_all[idx], r=radius_px)
        rows = np.repeat(np.arange(len(idx)), [len(n) for n in neigh])
        cols = np.concatenate(neigh)
        A = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(idx), len(idx)))
        counts = np.asarray(A.sum(axis=1)).ravel()
        out[idx] = (A @ values[idx]) / np.maximum(counts, 1)[:, None]
    return out


def _lognorm(X: csr_matrix, target_sum: float = 1e4) -> csr_matrix:
    total = np.asarray(X.sum(axis=1)).ravel()
    Xn = csr_matrix(X.multiply((target_sum / np.maximum(total, 1.0))[:, None]))
    Xn.data = np.log1p(Xn.data)
    return Xn


def build_targets(adata, obs: pd.DataFrame, nmf: AnchoredNMF, programs: Dict[str, List[str]], cfg: Config,
                  gene_mask: np.ndarray, um_per_px_by_sample: Optional[Dict[str, float]] = None
                  ) -> Tuple[List[str], Dict[str, str], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return ``(targets, target_family, Y, program_scores (N, P, S), expression_embedding, obs_with_columns)``."""
    section("Targets: learned programs and fixed sets at two scales")
    say("note: learned neighbourhood-scale targets average the cell-scale usage over the cells kept for training "
        "(after max_cells_per_sample subsampling), whereas the anchor_only neighbourhood targets were summed at ingest "
        "over every QC-passing cell; the two families therefore measure neighbourhoods of different cell density. "
        "This is a stated limitation (DESIGN_V3.md D28); the fix path is to project all cells at ingest.")
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    mask = np.asarray(gene_mask, dtype=bool)
    usage = nmf.transform(X[:, mask], var_names=adata.var_names[mask])      # (N, P) cell scale
    say(f"learned usage: {usage.shape}, genes used {int(mask.sum())}/{len(mask)}")
    scales = list(cfg.target_spaces)
    program_scores = np.zeros((usage.shape[0], usage.shape[1], len(scales)), dtype=np.float32)
    groups = obs["sample"].to_numpy()
    obs = obs.copy()
    targets: List[str] = []
    family: Dict[str, str] = {}
    for si, scale in enumerate(scales):
        vals = usage if scale == "cell" else neighbourhood_mean(usage, obs, cfg, um_per_px_by_sample)
        program_scores[:, :, si] = vals
        for pi, name in enumerate(nmf.program_names_):
            raw = f"prog_{name}_{scale}"
            obs[raw] = vals[:, pi]
            obs[raw + "_z"] = standardise(vals[:, pi], groups, cfg.target_norm)
            targets.append(raw + "_z")
            family[raw + "_z"] = "learned"
    if cfg.include_anchor_only_targets:
        have_counts = all(c in obs for c in ("cnt_total", "nb_cnt_total"))
        if not have_counts:
            say("count summaries absent: anchor_only targets skipped")
        for name in programs:
            if not have_counts or f"cnt_{name}" not in obs:
                continue
            for scale in scales:
                prefix = "" if scale == "cell" else "nb_"
                raw = f"anchor_{name}_{scale}"
                obs[raw] = program_score(obs[f"{prefix}cnt_{name}"].to_numpy(), obs[f"{prefix}cnt_total"].to_numpy())
                obs[raw + "_z"] = standardise(obs[raw].to_numpy(), groups, cfg.target_norm)
                targets.append(raw + "_z")
                family[raw + "_z"] = "anchor_only"
    Y = obs[targets].to_numpy().astype(np.float32)
    expression_embedding = np.stack([standardise(usage[:, p], groups, cfg.target_norm) for p in range(usage.shape[1])], axis=1).astype(np.float32)
    say(f"targets ({len(targets)}): " + ", ".join(targets))
    say(f"families: learned={sum(v == 'learned' for v in family.values())}, anchor_only={sum(v == 'anchor_only' for v in family.values())}")
    return targets, family, Y, program_scores, expression_embedding, obs


def learned_reliability(counts: csr_matrix, nmf: AnchoredNMF, var_names: pd.Index, gene_mask: np.ndarray,
                        obs: pd.DataFrame, cfg: Config, seed: int,
                        um_per_px_by_sample: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Split-half reliability of learned targets by thinning raw counts and re-projecting with fixed H."""
    section("Target reliability of learned programs (count thinning, fixed gene weights)")
    rng = np.random.default_rng(seed)
    counts = counts.tocsr().astype(np.int64)
    a_data = rng.binomial(counts.data, 0.5)
    A = csr_matrix((a_data.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)
    B = csr_matrix(((counts.data - a_data).astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)
    mask = np.asarray(gene_mask, dtype=bool)
    usage_a = nmf.transform(_lognorm(A)[:, mask], var_names=var_names[mask])
    usage_b = nmf.transform(_lognorm(B)[:, mask], var_names=var_names[mask])
    rows = []
    samples = obs["sample"].to_numpy()
    for scale in cfg.target_spaces:
        va = usage_a if scale == "cell" else neighbourhood_mean(usage_a, obs, cfg, um_per_px_by_sample)
        vb = usage_b if scale == "cell" else neighbourhood_mean(usage_b, obs, cfg, um_per_px_by_sample)
        for pi, name in enumerate(nmf.program_names_):
            for s in pd.unique(samples):
                m = samples == s
                r = stats.pearsonr(va[m, pi], vb[m, pi])[0] if va[m, pi].std() > 0 and vb[m, pi].std() > 0 else np.nan
                rows.append(dict(target=f"prog_{name}_{scale}_z", sample=s, r_half=r, r_sb=_spearman_brown(r), n_cells=int(m.sum())))
    tab = pd.DataFrame(rows)
    say(f"learned-target reliability: {tab['target'].nunique()} targets x {tab['sample'].nunique()} samples")
    return tab


def panel_coverage(programs: Dict[str, List[str]], var_names: pd.Index, gene_mask: np.ndarray) -> pd.DataFrame:
    present = set(var_names[np.asarray(gene_mask, dtype=bool)])
    rows = [dict(program=p, n_genes=len(g), in_panel=sum(x in present for x in g),
                 missing=", ".join(x for x in g if x not in present)) for p, g in programs.items()]
    tab = pd.DataFrame(rows)
    show_table(tab, title="panel coverage of the anchor gene sets:")
    return tab
