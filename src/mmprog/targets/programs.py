"""Program scores and lineage labels.

A program score is a count-based quantity: the transcripts of the program's genes
divided by all transcripts of the cell (or of its neighbourhood), scaled and
log-transformed. Counts are used, not per-gene z-scores, so that the same number
can later be split in half to measure how reliable it is (see ``reliability.py``).

Lineage labels follow the marker-scoring idea of scanpy's ``score_genes``: the
mean log expression of the markers minus the mean of a background gene set drawn
from the same expression bins, so that highly expressed housekeeping genes do not
dominate.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from ..config import Config
from ..logging_utils import say, section, show_table


def program_score(cnt: np.ndarray, total: np.ndarray) -> np.ndarray:
    """``log1p(1e4 * cnt / max(total, 1))``: counts per ten thousand, log scaled."""
    cnt = np.asarray(cnt, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    return np.log1p(1e4 * cnt / np.maximum(total, 1.0))


def standardise(values: np.ndarray, groups: np.ndarray, mode: str) -> np.ndarray:
    """Z-score a target either across the whole cohort or within each slide.

    ``cohort`` keeps between-patient differences, which the stage endpoint needs.
    ``slide`` removes them, which is the older behaviour kept for comparison.
    """
    v = pd.Series(np.asarray(values, dtype=np.float64))
    if mode == "cohort":
        return ((v - v.mean()) / (v.std() + 1e-6)).to_numpy()
    if mode == "slide":
        g = pd.Series(np.asarray(groups))
        return v.groupby(g.values).transform(lambda s: (s - s.mean()) / (s.std() + 1e-6)).to_numpy()
    raise ValueError(f"unknown standardisation mode {mode!r}")


def build_targets(adata, programs: Dict[str, List[str]], cfg: Config):
    """Add ``prog_<p>_<space>`` and ``prog_<p>_<space>_z`` columns to ``adata.obs``.

    Returns ``(adata, targets, kept_programs)``. Programs with fewer than
    ``cfg.min_program_genes`` genes in the panel, or without count summaries, are
    skipped and the reason is printed.
    """
    section("Targets: program scores")
    panel = set(adata.var_names)
    kept: Dict[str, List[str]] = {}
    targets: List[str] = []
    groups = adata.obs["sample"].to_numpy()
    for name, genes in programs.items():
        in_panel = [g for g in genes if g in panel]
        say(f"{name}: {len(in_panel)}/{len(genes)} genes in panel")
        if len(in_panel) < cfg.min_program_genes:
            say(f"skipping {name}: fewer than {cfg.min_program_genes} genes in panel", indent=1)
            continue
        if f"cnt_{name}" not in adata.obs:
            say(f"skipping {name}: no count summary column cnt_{name}", indent=1)
            continue
        kept[name] = in_panel
        for space in cfg.target_spaces:
            prefix = "" if space == "cell" else "nb_"
            raw = program_score(adata.obs[f"{prefix}cnt_{name}"].to_numpy(), adata.obs[f"{prefix}cnt_total"].to_numpy())
            col = f"prog_{name}_{space}"
            adata.obs[col] = raw
            adata.obs[f"{col}_z"] = standardise(raw, groups, cfg.target_norm)
            targets.append(f"{col}_z")
            say(f"{col}: mean {raw.mean():.3f} sd {raw.std():.3f} | zero fraction {(raw == 0).mean():.3f}", indent=1)
    say(f"{len(targets)} targets built, standardised by {cfg.target_norm}: {targets}")
    return adata, targets, kept


def _dense_columns(X, cols: np.ndarray) -> np.ndarray:
    sub = X[:, cols]
    return sub.toarray() if issparse(sub) else np.asarray(sub)


def _marker_score(X, var_names: pd.Index, markers: List[str], gene_mean: np.ndarray,
                  bins: np.ndarray, rng: np.random.Generator, ctrl_size: int) -> np.ndarray:
    """Mean marker expression minus mean of expression-matched background genes."""
    marker_idx = np.array([var_names.get_loc(g) for g in markers])
    background: List[int] = []
    for b in np.unique(bins[marker_idx]):
        pool = np.where((bins == b) & ~np.isin(np.arange(len(bins)), marker_idx))[0]
        if len(pool) == 0:
            continue
        background.extend(rng.choice(pool, size=min(ctrl_size, len(pool)), replace=False).tolist())
    score = _dense_columns(X, marker_idx).mean(axis=1)
    if background:
        score = score - _dense_columns(X, np.array(background)).mean(axis=1)
    return score


def score_lineages(adata, lineages: Dict[str, List[str]], cfg: Config, n_bins: int = 25, ctrl_size: int = 50):
    """Score each lineage marker set and assign every cell to its argmax lineage.

    ``adata.X`` must be log-normalised. Adds ``lin_<name>``, ``lineage`` (int) and
    ``lineage_name`` to ``adata.obs``; returns ``(adata, lin_names)``.
    """
    section("Targets: lineage labels")
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    gene_mean = np.asarray(X.mean(axis=0)).ravel()
    # rank genes by mean expression and cut into equal-size bins so background genes match marker expression levels
    order = np.argsort(gene_mean, kind="stable")
    bins = np.empty(len(gene_mean), dtype=np.int64)
    bins[order] = np.minimum(np.arange(len(gene_mean)) * n_bins // max(len(gene_mean), 1), n_bins - 1)
    rng = np.random.default_rng(cfg.seed)
    lin_names: List[str] = []
    for name, genes in lineages.items():
        markers = [g for g in genes if g in adata.var_names]
        say(f"{name}: {len(markers)}/{len(genes)} markers in panel")
        if len(markers) < cfg.min_program_genes:
            say(f"skipping {name}: fewer than {cfg.min_program_genes} markers", indent=1)
            continue
        adata.obs[f"lin_{name}"] = _marker_score(X, adata.var_names, markers, gene_mean, bins, rng, ctrl_size)
        lin_names.append(name)
    if not lin_names:
        raise ValueError("no lineage had enough markers in the panel")
    scores = adata.obs[[f"lin_{n}" for n in lin_names]].to_numpy()
    adata.obs["lineage"] = scores.argmax(axis=1).astype(np.int64)
    adata.obs["lineage_name"] = [lin_names[i] for i in adata.obs["lineage"]]
    comp = adata.obs.groupby("sample", observed=True)["lineage_name"].value_counts().unstack(fill_value=0)
    show_table(comp, title="cells per lineage and sample:")
    if "lineage_true" in adata.obs:
        agreement = (adata.obs["lineage_true"].to_numpy() == adata.obs["lineage_name"].to_numpy()).mean()
        say(f"agreement with planted lineage (synthetic data): {agreement:.3f}")
    return adata, lin_names
