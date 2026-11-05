"""Patient-level evaluation of spatially regularized boosting on patch-grid features.

Two questions, answered against the plain AdaBoost control (lambda = 0) on the same
stumps:
1. Does the regularised model lose or gain held-out accuracy? (Per-patient AUC
   for the top-quartile classification; per-patient Pearson for the squared-loss
   variant.) The original paper found a small accuracy cost on fMRI and a small
   gain on faces; both outcomes are reported, not assumed.
2. Is the importance map spatially coherent? Measured by Moran's I of the
   importance map over the block grid: about 0 for scattered selection, positive
   for clustered selection. This is the property the regulariser buys, and it is
   what makes the map readable as centre, inner ring, outer ring of the patch.

Everything is fit inside patient-level folds; thresholds for the top-quartile
label are computed within slide so a slide's overall level does not leak.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..config import Config
from ..logging_utils import say, section, show_table
from ..models.spatial_boost import SpatiallyRegularizedBoosting


def top_quartile_labels(y: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """+1 for cells in the top quartile of their own slide, -1 otherwise."""
    lab = -np.ones(len(y))
    for s in np.unique(sid):
        m = sid == s
        lab[m] = np.where(y[m] >= np.quantile(y[m], 0.75), 1.0, -1.0)
    return lab


def _per_patient(metric, pred: np.ndarray, truth: np.ndarray, groups: np.ndarray, min_cells: int) -> Dict[str, float]:
    out = {}
    for p in np.unique(groups):
        m = groups == p
        if m.sum() < min_cells:
            continue
        try:
            out[str(p)] = float(metric(truth[m], pred[m]))
        except ValueError:
            out[str(p)] = float("nan")
    return out


def run_spatial_boost_cv(G: np.ndarray, locations: np.ndarray, y: np.ndarray, sid: np.ndarray, groups: np.ndarray,
                         cfg: Config, grid: int, lambdas: Tuple[float, ...] = (0.0, 0.1, 0.5),
                         radius_um: float = 5.0, n_rounds: int = 150, max_train_cells: int = 40000,
                         seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, list]]:
    """Return (results table, importance maps per lambda, per-fold histories per lambda)."""
    section("Spatially regularized boosting: patient-level cross-validation")
    rng = np.random.default_rng(seed)
    labels = top_quartile_labels(y, sid)
    n_splits = min(cfg.n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    maps: Dict[str, np.ndarray] = {}
    histories: Dict[str, list] = {}
    say(f"grid features {G.shape}, {n_splits} folds, lambdas {list(lambdas)}, radius {radius_um} um, {n_rounds} rounds")
    for loss in ("exponential", "squared"):
        target = labels if loss == "exponential" else y
        for lam in lambdas:
            t0 = time.time()
            pred = np.zeros(len(y))
            coherence, hist = [], []
            beta_sum = None
            for fold, (tr, te) in enumerate(gkf.split(G, target, groups)):
                if len(tr) > max_train_cells:
                    tr = rng.choice(tr, max_train_cells, replace=False)
                model = SpatiallyRegularizedBoosting(lam=lam, radius=radius_um, n_rounds=n_rounds, loss=loss,
                                                     verbose_every=0).fit(G[tr], target[tr], locations)
                pred[te] = model.decision_function(G[te])
                coherence.append(model.spatial_coherence())
                hist.append(model.history)
                beta = model.importance()
                beta_sum = beta if beta_sum is None else beta_sum + beta
                say(f"{loss} lambda={lam} fold {fold}: {len(tr)} train cells, {len(te)} test cells, "
                    f"{len(model.history)} rounds, coherence {coherence[-1]:.2f}", indent=1)
            if loss == "exponential":
                per = _per_patient(roc_auc_score, pred, (labels > 0).astype(int), groups, cfg.min_cells_per_patient_metric)
                metric_name = "auc_patient_median"
            else:
                per = _per_patient(lambda a, b: stats.pearsonr(a, b)[0], pred, y, groups, cfg.min_cells_per_patient_metric)
                metric_name = "pearson_patient_median"
            key = f"{loss}_lam{lam}"
            maps[key] = (beta_sum / n_splits).reshape(2, grid, grid)
            histories[key] = hist
            rows.append(dict(loss=loss, lam=lam, metric=metric_name, value=float(np.nanmedian(list(per.values()))),
                             coherence_mean=float(np.nanmean(coherence)),
                             n_active_components=int((beta_sum > 0).sum()), seconds=round(time.time() - t0, 1)))
            say(f"{loss} lambda={lam}: {metric_name}={rows[-1]['value']:.3f}, coherence={rows[-1]['coherence_mean']:.2f}, "
                f"active components={rows[-1]['n_active_components']}/{G.shape[1]}, {rows[-1]['seconds']}s")
    res = pd.DataFrame(rows)
    show_table(res, title="spatially regularized boosting vs AdaBoost control (lambda = 0):")
    _interpret(res)
    return res, maps, histories


def _interpret(res: pd.DataFrame) -> None:
    for loss in res["loss"].unique():
        d = res[res["loss"] == loss].sort_values("lam")
        base = d.iloc[0]
        for _, r in d.iloc[1:].iterrows():
            acc = r["value"] - base["value"]
            coh = r["coherence_mean"] - base["coherence_mean"]
            verdict = ("coherence gained at no accuracy cost" if coh > 0.05 and acc > -0.02 else
                       "coherence gained at an accuracy cost" if coh > 0.05 else
                       "no coherence gain")
            say(f"{loss}: lambda {r['lam']} vs 0: accuracy {acc:+.3f}, coherence {coh:+.2f} -> {verdict}")


def ring_summary(importance_map: np.ndarray, grid: int, block_um: float) -> pd.DataFrame:
    """Importance by block-centre distance from the nucleus: centre (< 5 um), inner (5 to 10 um), outer (>= 10 um).

    Both the share of total importance per band and the share divided by the number of
    blocks in the band are reported, because the bands hold different numbers of blocks
    and a uniform map would otherwise look outer-ring dominated.
    """
    centres = (np.arange(grid) + 0.5) * block_um - grid * block_um / 2
    yy, xx = np.meshgrid(centres, centres, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2)
    bands = {"centre": radius < 5.0, "inner_ring": (radius >= 5.0) & (radius < 10.0), "outer_ring": radius >= 10.0}
    rows = []
    for plane, name in enumerate(("mean", "std")):
        total = importance_map[plane].sum() + 1e-12
        row = {"plane": name}
        for b, m in bands.items():
            n_blocks = int(m.sum())
            share = float(importance_map[plane][m].sum() / total) if n_blocks else float("nan")
            row[b] = share
            row[f"n_blocks_{b}"] = n_blocks
            row[f"per_block_{b}"] = share / n_blocks if n_blocks else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
