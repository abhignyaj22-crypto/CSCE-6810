"""Calibration of the predictive uncertainty.

Two views of the same question, "does the stated uncertainty mean what it says":

1. Gaussian coverage: the fraction of held-out cells whose measured target falls inside
   the symmetric interval mean +/- z_level * std, per target and level, pooled and as the
   per-patient median. A calibrated model gives coverage close to the nominal level.
2. Split conformal intervals: the quantile of the normalised residual |y - mean| / std on
   the inner validation patient scales the interval, mean +/- q * std, so that coverage on
   exchangeable data is at least 1 - alpha (split conformal; Conformal Risk Control,
   Angelopoulos et al., ICLR 2024, generalises the guarantee to other losses).

Caveat (DESIGN_V3.md D30): cells of one patient are not exchangeable with cells of
another, so the conformal guarantee holds only approximately when the calibration unit is
one patient and the test unit is a different patient. Coverage is therefore reported per
patient as well as pooled, and the reader should look at the spread, not the pooled value.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say, warn, show_table


def coverage_table(mean: np.ndarray, std: np.ndarray, Y: np.ndarray, groups: np.ndarray, targets: List[str],
                   levels: Iterable[float] = (0.5, 0.9), min_cells: int = 30) -> pd.DataFrame:
    """Empirical coverage of symmetric Gaussian intervals per target and nominal level."""
    rows = []
    patients = np.unique(groups)
    for level in levels:
        z = stats.norm.ppf(0.5 + level / 2)
        for j, t in enumerate(targets):
            ok = np.isfinite(mean[:, j]) & np.isfinite(std[:, j]) & np.isfinite(Y[:, j])
            if ok.sum() < min_cells:
                continue
            inside = np.abs(Y[ok, j] - mean[ok, j]) <= z * std[ok, j]
            per = []
            for p in patients:
                m = ok & (groups == p)
                if m.sum() >= min_cells:
                    per.append(float((np.abs(Y[m, j] - mean[m, j]) <= z * std[m, j]).mean()))
            rows.append(dict(target=t, level=level, coverage_pooled=float(inside.mean()),
                             coverage_patient_median=float(np.median(per)) if per else np.nan,
                             coverage_patient_min=float(np.min(per)) if per else np.nan,
                             coverage_patient_max=float(np.max(per)) if per else np.nan,
                             mean_std=float(np.nanmean(std[ok, j])), n_cells=int(ok.sum()), n_patients=len(per)))
    tab = pd.DataFrame(rows)
    if len(tab):
        show_table(tab.round(3), title="Gaussian interval coverage (nominal vs empirical, pooled and per-patient median):")
        for level in levels:
            d = tab[tab["level"] == level]
            if len(d):
                say(f"level {level:.0%}: pooled coverage median over targets {d['coverage_pooled'].median():.3f}, "
                    f"patient-median {d['coverage_patient_median'].median():.3f} "
                    f"({'over' if d['coverage_pooled'].median() > level else 'under'}-covering)")
    return tab


def expected_calibration_error(mean: np.ndarray, std: np.ndarray, Y: np.ndarray, n_bins: int = 10) -> float:
    """Regression ECE: mean absolute gap between nominal and empirical coverage over a grid of levels."""
    ok = np.isfinite(mean) & np.isfinite(std) & np.isfinite(Y)
    if ok.sum() < 10:
        return float("nan")
    levels = np.linspace(0.05, 0.95, n_bins)
    gaps = []
    r = np.abs(Y[ok] - mean[ok]) / np.maximum(std[ok], 1e-8)
    for lv in levels:
        z = stats.norm.ppf(0.5 + lv / 2)
        gaps.append(abs(float((r <= z).mean()) - lv))
    return float(np.mean(gaps))


def conformal_quantile(resid_norm_val: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal quantile of the normalised residuals on the calibration set."""
    r = np.asarray(resid_norm_val, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(r)[k - 1])


def conformal_coverage(mean: np.ndarray, std: np.ndarray, Y: np.ndarray, groups: np.ndarray, targets: List[str],
                       q: Dict[str, float], alpha: float = 0.1, min_cells: int = 30) -> pd.DataFrame:
    """Coverage of mean +/- q_t * std on test patients, per target, pooled and per patient."""
    rows = []
    for j, t in enumerate(targets):
        qt = q.get(t, np.nan)
        ok = np.isfinite(mean[:, j]) & np.isfinite(std[:, j]) & np.isfinite(Y[:, j])
        if not np.isfinite(qt) or ok.sum() < min_cells:
            continue
        inside = np.abs(Y[ok, j] - mean[ok, j]) <= qt * std[ok, j]
        per = {}
        for p in np.unique(groups):
            m = ok & (groups == p)
            if m.sum() >= min_cells:
                per[str(p)] = float((np.abs(Y[m, j] - mean[m, j]) <= qt * std[m, j]).mean())
        rows.append(dict(target=t, alpha=alpha, nominal=1 - alpha, q=qt, coverage_pooled=float(inside.mean()),
                         coverage_patient_median=float(np.median(list(per.values()))) if per else np.nan,
                         coverage_patient_min=float(np.min(list(per.values()))) if per else np.nan,
                         n_patients=len(per)))
    tab = pd.DataFrame(rows)
    if len(tab):
        show_table(tab.round(3), title=f"split-conformal coverage at nominal {1 - alpha:.0%} (patient-level caveat applies):")
    return tab


def calibration_from_folds(mean: np.ndarray, std: np.ndarray, Y: np.ndarray, groups: np.ndarray, targets: List[str],
                           folds, alpha: float = 0.1, levels: Iterable[float] = (0.5, 0.9),
                           val_preds: Optional[Dict[int, Dict[str, np.ndarray]]] = None) -> pd.DataFrame:
    """Gaussian coverage on all out-of-fold cells plus conformal coverage calibrated per fold.

    Split conformal protocol: for each fold the quantile of normalised residuals is taken from
    the fold's inner validation patient, predicted by THAT fold's model (``val_preds``, saved
    by Training), and applied to the fold's test patients, which the model never saw. The
    calibration patient and the test patients are different people, which is the setting the
    patient-level caveat describes. When ``val_preds`` is absent the out-of-fold rows of the
    validation patient are used instead; those come from another fold's model, so that path
    is only a fallback and is printed as such.
    """
    gauss = coverage_table(mean, std, Y, groups, targets, levels)
    gauss["kind"] = "gaussian"
    conf_rows = []
    gs = groups.astype(str)
    if val_preds:
        say(f"conformal quantiles from in-fold validation predictions of {len(val_preds)} folds (split conformal)")
    else:
        warn("conformal quantiles fall back to out-of-fold rows of the validation patient (another fold's model); "
             "this is not the split-conformal protocol")
    for f in folds or []:
        te = np.isin(gs, f.test_patients)
        q = {}
        vp = (val_preds or {}).get(f.fold_id)
        if vp is not None:
            va_rows, va_mean, va_std = np.asarray(vp["rows"]), np.asarray(vp["mean"]), np.asarray(vp["std"])
            for j, t in enumerate(targets):
                r = np.abs(Y[va_rows, j] - va_mean[:, j]) / np.maximum(va_std[:, j], 1e-8)
                q[t] = conformal_quantile(r, alpha)
        else:
            va = np.isin(gs, [f.val_patient])
            for j, t in enumerate(targets):
                r = np.abs(Y[va, j] - mean[va, j]) / np.maximum(std[va, j], 1e-8)
                q[t] = conformal_quantile(r, alpha)
        tab = conformal_coverage(mean[te], std[te], Y[te], groups[te], targets, q, alpha)
        if len(tab):
            tab["fold"] = f.fold_id
            conf_rows.append(tab)
    if conf_rows:
        conf = pd.concat(conf_rows, ignore_index=True)
        agg = conf.groupby("target").agg(q=("q", "median"), coverage_pooled=("coverage_pooled", "mean"),
                                         coverage_patient_median=("coverage_patient_median", "median"),
                                         coverage_patient_min=("coverage_patient_min", "min")).reset_index()
        agg["level"] = 1 - alpha
        agg["kind"] = "conformal"
        out = pd.concat([gauss, agg], ignore_index=True)
    else:
        out = gauss
    ece = {t: expected_calibration_error(mean[:, j], std[:, j], Y[:, j]) for j, t in enumerate(targets)}
    out["ece"] = out["target"].map(ece)
    return out
