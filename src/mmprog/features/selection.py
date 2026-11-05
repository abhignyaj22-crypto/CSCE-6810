"""Feature selection with a deliberately narrow mandate.

Only two steps change what the model sees, and neither looks at the targets:

1. a variance floor on embedding dimensions (dead channels carry nothing), and
2. correlation pruning among morphometrics (a near-duplicate column adds noise to
   the scaler and nothing to the model).

Everything else is a report: a univariate Spearman screen of morphometrics against
each target, the PCA retained-variance table, and permutation importance of a
ridge model. Those use the targets, so using them to drop features inside a fold
would be selection on the outcome and would bias the held-out estimate upward.
They are printed and saved so a reader can see which features carry signal.
The selector is fit on training cells only, inside each cross-validation fold.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..logging_utils import say, section, show_table, timed
from .morphometrics import MORPHO_NAMES


class FeatureSelector:
    """Variance floor + morphometric redundancy pruning, with diagnostic reports."""

    REPORT_SUBSAMPLE = 20000
    REPORT_PCS = 32

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keep_emb_dims: Optional[np.ndarray] = None
        self.keep_mor_dims: Optional[np.ndarray] = None
        self.report: Dict[str, Any] = {}

    # ------------------------------------------------------------------ steps
    def _variance_floor(self, emb_sub: np.ndarray) -> np.ndarray:
        var = emb_sub.var(axis=0)
        keep = np.where(var > self.cfg.selection_variance_floor)[0]
        dropped = emb_sub.shape[1] - len(keep)
        say(f"step 1 variance floor ({self.cfg.selection_variance_floor:g}): keep {len(keep)}/{emb_sub.shape[1]} embedding dims, drop {dropped} dead dims")
        self.report["embedding_dims_total"] = int(emb_sub.shape[1])
        self.report["embedding_dims_kept"] = int(len(keep))
        self.report["embedding_dims_dropped"] = int(dropped)
        return keep

    def _correlation_pruning(self, mor_sub: np.ndarray, names: List[str]) -> np.ndarray:
        """Greedy: walk columns in order, drop any column whose |r| with a kept column exceeds the threshold."""
        thr = self.cfg.selection_corr_threshold
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(mor_sub, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        kept: List[int] = []
        dropped: List[Dict[str, Any]] = []
        for j in range(mor_sub.shape[1]):
            if mor_sub[:, j].std() == 0:
                dropped.append(dict(column=names[j], reason="constant"))
                continue
            partner = next((k for k in kept if abs(corr[j, k]) > thr), None)
            if partner is None:
                kept.append(j)
            else:
                dropped.append(dict(column=names[j], reason=f"|r|={abs(corr[j, partner]):.3f} with {names[partner]}"))
        say(f"step 2 correlation pruning (|r| > {thr}): keep {len(kept)}/{mor_sub.shape[1]} morphometrics")
        for d in dropped:
            say(f"dropped {d['column']}: {d['reason']}", indent=1)
        self.report["morphometrics_kept"] = [names[k] for k in kept]
        self.report["morphometrics_dropped"] = dropped
        return np.array(kept, dtype=np.int64)

    def _univariate_screen(self, mor_sub: np.ndarray, Y_sub: np.ndarray, names: List[str], targets: List[str]) -> None:
        rows = []
        for j, name in enumerate(names):
            row = {"morphometric": name}
            for t, target in enumerate(targets):
                rho = stats.spearmanr(mor_sub[:, j], Y_sub[:, t])[0] if mor_sub[:, j].std() > 0 else np.nan
                row[target] = round(float(rho), 3) if np.isfinite(rho) else None
            rows.append(row)
        tab = pd.DataFrame(rows).set_index("morphometric")
        show_table(tab, title="step 3 univariate Spearman of morphometrics vs targets (report only, training cells):")
        self.report["univariate_spearman"] = rows

    def _pca_variance_report(self, emb_sub: np.ndarray) -> PCA:
        n_comp = min(self.cfg.pca_dim, emb_sub.shape[0], emb_sub.shape[1])
        pca = PCA(n_comp, random_state=self.cfg.seed).fit(StandardScaler().fit_transform(emb_sub))
        cum = np.cumsum(pca.explained_variance_ratio_)
        milestones = {}
        for q in (0.5, 0.8, 0.9, 0.95):
            milestones[f"pcs_for_{int(q * 100)}pct"] = int(np.searchsorted(cum, q) + 1) if cum[-1] >= q else None
        say(f"step 4 PCA retained variance on a {emb_sub.shape[0]}-cell subsample: {n_comp} PCs keep {100 * cum[-1]:.1f}%; " +
            ", ".join(f"{k}={v}" for k, v in milestones.items()))
        self.report["pca_variance"] = dict(n_components=int(n_comp), cumulative_at_last=float(cum[-1]),
                                           explained_ratio_first10=[float(v) for v in pca.explained_variance_ratio_[:10]], **milestones)
        return pca

    def _permutation_importance(self, emb_sub: np.ndarray, mor_sub: np.ndarray, Y_sub: np.ndarray,
                                mor_names: List[str], targets: List[str]) -> None:
        n_pcs = min(self.REPORT_PCS, emb_sub.shape[0], emb_sub.shape[1])
        z = PCA(n_pcs, random_state=self.cfg.seed).fit_transform(StandardScaler().fit_transform(emb_sub))
        X = np.hstack([z, StandardScaler().fit_transform(mor_sub)])
        feat_names = [f"PC{i + 1}" for i in range(n_pcs)] + mor_names
        model = Ridge(alpha=1.0).fit(X, Y_sub)
        r2 = model.score(X, Y_sub)
        imp = permutation_importance(model, X, Y_sub, n_repeats=5, random_state=self.cfg.seed, scoring="r2")
        order = np.argsort(imp.importances_mean)[::-1][: self.cfg.selection_report_top]
        tab = pd.DataFrame(dict(feature=[feat_names[i] for i in order],
                                importance_mean=np.round(imp.importances_mean[order], 4),
                                importance_std=np.round(imp.importances_std[order], 4)))
        say(f"step 5 ridge on [{n_pcs} PCs + {len(mor_names)} morphometrics] -> targets {targets}: in-sample R2 {r2:.3f}")
        show_table(tab, title="permutation importance (report only; drop in R2 when a feature is shuffled):")
        self.report["ridge_in_sample_r2"] = float(r2)
        self.report["permutation_importance_top"] = tab.to_dict("records")

    # ------------------------------------------------------------------ public
    def fit(self, emb, mor, Y: np.ndarray, obs: pd.DataFrame, train_idx) -> "FeatureSelector":
        section("Feature selection (training cells only)")
        train_idx = np.asarray(train_idx)
        rng = np.random.default_rng(self.cfg.seed)
        sub = np.sort(rng.choice(train_idx, min(self.REPORT_SUBSAMPLE, len(train_idx)), replace=False))
        say(f"fitting on {len(train_idx)} training cells; diagnostics on a {len(sub)}-cell subsample")
        mor_names = MORPHO_NAMES(self.cfg)
        targets = [c for c in obs.columns if c.startswith("prog_") and c.endswith("_z")]
        if len(targets) != Y.shape[1]:
            targets = [f"target_{t}" for t in range(Y.shape[1])]
        with timed("feature selection"):
            emb_sub = np.asarray(emb[sub], dtype=np.float32)
            mor_sub = np.asarray(mor[sub], dtype=np.float32)
            Y_sub = np.asarray(Y[sub], dtype=np.float32)
            self.keep_emb_dims = self._variance_floor(emb_sub)
            self.keep_mor_dims = self._correlation_pruning(mor_sub, mor_names)
            self._univariate_screen(mor_sub, Y_sub, mor_names, targets)
            self._pca_variance_report(emb_sub[:, self.keep_emb_dims])
            self._permutation_importance(emb_sub[:, self.keep_emb_dims], mor_sub[:, self.keep_mor_dims], Y_sub,
                                         [mor_names[k] for k in self.keep_mor_dims], targets)
        self.report["n_train_cells"] = int(len(train_idx))
        self.report["n_report_cells"] = int(len(sub))
        say("decision: the model receives the variance-filtered embedding dims and the pruned morphometrics; "
            "the Spearman screen and permutation importance are reports and do not drop features")
        return self


def save_selection_report(selector: FeatureSelector, path: str) -> None:
    payload = dict(keep_emb_dims=None if selector.keep_emb_dims is None else selector.keep_emb_dims.tolist(),
                   keep_mor_dims=None if selector.keep_mor_dims is None else selector.keep_mor_dims.tolist(),
                   report=selector.report)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    say(f"feature selection report written to {path}")
