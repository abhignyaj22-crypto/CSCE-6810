"""Patient-level endpoints.

- program burden: the fraction of a patient's plasma cells whose predicted main
  program score is in the top quartile of the cohort. One number per patient.
- stage trend: Spearman correlation of burden with disease stage
  (control < MGUS < SMM < MM). This needs cohort-standardised targets; slide
  standardisation removes the between-patient signal by construction.
- Cox PFS: optional, when a clinical table is supplied (lifelines imported lazily).
- signature concordance (Phase 3): scores a published signature in bulk RNA.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..logging_utils import say, show_table


def program_burden(
    obs: pd.DataFrame, pred_full: np.ndarray, main_target_idx: int, cfg: Any,
    stage_order: Dict[str, int], plasma_label: str = "plasma",
) -> pd.DataFrame:
    """Per-patient burden of predicted high-program plasma cells."""
    pred = pred_full[:, main_target_idx]
    is_pc = obs["lineage_name"].values == plasma_label
    if is_pc.sum() == 0:
        say("program burden: no plasma cells labelled; returning empty table")
        return pd.DataFrame(columns=["patient", "stage", "n_plasma", "burden", "stage_ord"])
    thr = float(np.percentile(pred[is_pc], 75))
    pc = obs.loc[is_pc, ["patient", "stage"]].copy()
    pc["pred"] = pred[is_pc]
    burden = (pc.groupby(["patient", "stage"], observed=True)
                .agg(n_plasma=("pred", "size"), burden=("pred", lambda v: float((v > thr).mean())))
                .reset_index())
    too_few = burden["n_plasma"] < cfg.min_plasma_for_burden
    burden.loc[too_few, "burden"] = np.nan
    burden["stage_ord"] = burden["stage"].map(stage_order)
    say(f"program burden: threshold {thr:.3f} (75th percentile over {int(is_pc.sum())} plasma cells); "
        f"{int(too_few.sum())} patients below {cfg.min_plasma_for_burden} plasma cells set to NaN")
    show_table(burden, title="burden per patient:")
    return burden


def stage_trend(burden: pd.DataFrame) -> Tuple[float, float, int]:
    """Spearman rho of burden vs ordinal stage; (nan, nan, n) when fewer than 4 patients."""
    b = burden.dropna(subset=["burden", "stage_ord"])
    n = int(len(b))
    if n < 4:
        say(f"stage trend: only {n} patients with a burden value; need at least 4")
        return (np.nan, np.nan, n)
    rho, p = stats.spearmanr(b["stage_ord"], b["burden"])
    say(f"stage trend: Spearman rho={rho:.3f} p={p:.3g} (n={n})")
    return (float(rho), float(p), n)


def cox_pfs(burden: pd.DataFrame, clinical_csv: Optional[str], out_path: str) -> Optional[pd.DataFrame]:
    """Cox model of PFS on burden. Needs columns patient, pfs_months, pfs_event in the clinical table."""
    if not clinical_csv or not os.path.exists(clinical_csv):
        say("Cox PFS gate: no clinical table (set clinical_csv with columns patient, pfs_months, pfs_event)")
        return None
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        say("Cox PFS gate: lifelines is not installed (pip install lifelines)")
        return None
    clin = pd.read_csv(clinical_csv).merge(burden[["patient", "burden"]].dropna(), on="patient")
    if len(clin) < 5:
        say(f"Cox PFS gate: only {len(clin)} patients with both burden and clinical data")
        return None
    cph = CoxPHFitter().fit(clin[["pfs_months", "pfs_event", "burden"]], "pfs_months", "pfs_event")
    summary = cph.summary
    summary.to_csv(out_path)
    say(f"Cox PFS on burden (n={len(clin)}): HR={np.exp(summary.loc['burden', 'coef']):.3f} "
        f"p={summary.loc['burden', 'p']:.3g}")
    return summary


def signature_concordance(signature: List[str], commpass_csv: Optional[str], out_path: str,
                          enabled: bool = True) -> Optional[pd.DataFrame]:
    """Phase 3 gate: score a published signature in bulk RNA and fit a Cox model."""
    if not enabled or not signature or not commpass_csv or not os.path.exists(commpass_csv):
        say("signature concordance gate: needs Phase 3 enabled, a signature list, and commpass_csv")
        return None
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        say("signature concordance gate: lifelines is not installed")
        return None
    bulk = pd.read_csv(commpass_csv, index_col=0)
    genes = [g for g in signature if g in bulk.columns]
    say(f"signature genes found in bulk: {len(genes)}/{len(signature)}")
    if len(genes) < 3:
        say("signature concordance: fewer than 3 signature genes in the bulk table; skipped")
        return None
    bulk["sig_score"] = StandardScaler().fit_transform(bulk[genes]).mean(1)
    cph = CoxPHFitter().fit(bulk[["pfs_months", "pfs_event", "sig_score"]], "pfs_months", "pfs_event")
    cph.summary.to_csv(out_path)
    say(f"signature Cox (n={len(bulk)}): HR={np.exp(cph.summary.loc['sig_score', 'coef']):.3f} "
        f"p={cph.summary.loc['sig_score', 'p']:.3g}")
    return cph.summary
