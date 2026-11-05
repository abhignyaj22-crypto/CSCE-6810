"""Scoring held-out predictions.

The headline number is the per-patient median Pearson correlation: predictions
and measured scores are correlated inside each held-out patient, and the median
over patients is reported. Pooled correlations are also given, but they mix
between-patient and within-patient structure and are easier to inflate.

The reliability-adjusted correlation divides each patient's r by the square
root of the target's split-half reliability for that patient (the noise
ceiling). A value near 1 means the model reproduces the target as well as an
independent re-measurement would.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(stats.pearsonr(a, b)[0])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b)[0])


def score(
    preds: np.ndarray,
    Y: np.ndarray,
    targets: List[str],
    groups: np.ndarray,
    rel_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    min_cells: int = 30,
    sids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """One row per target with pooled and per-patient correlations.

    ``groups`` is the patient of every cell and ``sids`` the sample (slide) of every cell.
    The reliability table is keyed by sample, so the ceiling for a patient is the median
    ``r_sb`` over the samples that patient's cells come from. When ``sids`` is omitted the
    sample is assumed to equal the patient (one slide per patient).
    """
    rel_lookup = rel_lookup or {}
    sids = groups if sids is None else sids
    rows = []
    patients = np.unique(groups)
    for j, t in enumerate(targets):
        per: Dict[str, float] = {}
        per_adj: Dict[str, float] = {}
        for p in patients:
            m = groups == p
            if m.sum() <= min_cells:
                continue
            r = _safe_pearson(preds[m, j], Y[m, j])
            per[str(p)] = r
            ceilings = [rel_lookup.get((t, str(s)), np.nan) for s in np.unique(sids[m])]
            ceilings = [c for c in ceilings if c == c]
            ceiling = float(np.median(ceilings)) if ceilings else np.nan
            if np.isnan(r) or np.isnan(ceiling):
                per_adj[str(p)] = np.nan
            else:
                # floor the ceiling at 0.05 so a near-zero reliability does not explode the ratio
                per_adj[str(p)] = r / np.sqrt(max(ceiling, 0.05))
        vals = np.array(list(per.values()), dtype=float)
        adj_vals = np.array(list(per_adj.values()), dtype=float)
        rows.append(dict(
            target=t,
            pearson_pooled=_safe_pearson(preds[:, j], Y[:, j]),
            spearman_pooled=_safe_spearman(preds[:, j], Y[:, j]),
            pearson_patient_median=float(np.nanmedian(vals)) if len(vals) and not np.all(np.isnan(vals)) else np.nan,
            pearson_patient_median_adj=float(np.nanmedian(adj_vals)) if len(adj_vals) and not np.all(np.isnan(adj_vals)) else np.nan,
            n_patients_scored=int(len(vals)),
            per_patient=per,
            per_patient_adj=per_adj,
        ))
    return pd.DataFrame(rows)


def bootstrap_ci(per_patient: Dict[str, float], n_boot: int, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap interval of the per-patient median (resampling patients)."""
    rng = np.random.default_rng(seed)
    v = np.array(list(per_patient.values()), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    meds = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return (round(float(lo), 3), round(float(hi), 3))
