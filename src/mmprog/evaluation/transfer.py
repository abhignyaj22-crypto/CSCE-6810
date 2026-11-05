"""Cross-cohort transfer (Phase 2 gate).

Train on one cohort, test on another, shared targets only. The test slides are
unseen, so their slide embedding is the reserved zero row. This is the check
that the morphology-to-program mapping is not a platform artefact.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say
from ..models.train import predict, train_model
from .cross_validation import CVData, build_features


def run_transfer(
    train_cohort: str, test_cohort: str, data: CVData, EMB: np.ndarray, MOR: np.ndarray,
    cfg: Any, device: str, seed: int = 0,
    feature_space_factory: Optional[Callable] = None, selector_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    cohorts = data.obs["cohort"].values
    tr_all = np.where(cohorts == train_cohort)[0]
    te = np.where(cohorts == test_cohort)[0]
    if len(tr_all) == 0 or len(te) == 0:
        raise ValueError(f"cohort missing: train={train_cohort} ({len(tr_all)} cells) test={test_cohort} ({len(te)} cells)")
    pats = np.unique(data.GROUPS[tr_all])
    va_pat = pats[-1]
    va = tr_all[data.GROUPS[tr_all] == va_pat]
    tr = tr_all[data.GROUPS[tr_all] != va_pat]
    say(f"transfer {train_cohort} -> {test_cohort}: train {len(pats) - 1} patients, early-stop on {va_pat}, "
        f"test {len(np.unique(data.GROUPS[te]))} patients")

    X, _, _ = build_features(EMB, MOR, data, tr, cfg, seed, bool(cfg.use_morpho),
                             feature_space_factory, selector_factory)
    model, _, _, _ = train_model(X, data.Y, data.XY, data.SID, data.LIN, data.NB, tr, va, data.n_samples,
                                 len(data.lin_names), cfg, device, seed=seed, verbose=False)
    sid_unseen = np.full(len(te), data.n_samples, dtype=np.int64)   # zero slide embedding
    p = predict(model, X[te], data.XY[te], sid_unseen, device)

    rows = []
    for j, t in enumerate(data.targets):
        per = {}
        for pt in np.unique(data.GROUPS[te]):
            m = data.GROUPS[te] == pt
            if m.sum() > cfg.min_cells_per_patient_metric and np.std(p[m, j]) > 0 and np.std(data.Y[te][m, j]) > 0:
                per[str(pt)] = float(stats.pearsonr(p[m, j], data.Y[te][m, j])[0])
        rows.append(dict(target=t, train=train_cohort, test=test_cohort,
                         pearson_patient_median=float(np.nanmedian(list(per.values()))) if per else np.nan,
                         n_test_patients=len(per)))
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        say(f"{r.target}: r={r.pearson_patient_median:.3f} over {r.n_test_patients} test patients", indent=1)
    return out
