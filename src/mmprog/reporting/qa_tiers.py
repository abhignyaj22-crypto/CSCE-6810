"""Quality-assurance tiers for pathologist review.

Tier 1 is the overlay figure (see ``figures.py``).
Tier 2 flags the cells whose held-out prediction disagrees most with the
measured program score (top 5 percent of absolute residual within each slide).
Tier 3 clusters the flagged cells with DBSCAN; a cluster of at least 100 cells is
a region where the model is systematically wrong and is exported as a re-image
request. The thresholds are deliberately simple so a reviewer can reproduce them.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ..logging_utils import say, section, show_table
from ..paths import RunDir


def flag_tiers(obs: pd.DataFrame, pred_full: np.ndarray, main_idx: int, main_target: str, run: RunDir,
               tier2_quantile: float = 0.95, eps_px: float = 60.0, min_samples: int = 25,
               min_cluster_cells: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    section("S8a QA tiers: residual-based review lists")
    obs = obs.copy()
    obs["resid"] = np.abs(pred_full[:, main_idx] - obs[main_target].values)
    obs["flag_tier2"] = obs.groupby("sample")["resid"].transform(lambda v: v > v.quantile(tier2_quantile))
    say(f"tier 2 rule: |pred - measured| above the {tier2_quantile:.0%} quantile within each slide, target {main_target}")

    cols = ["sample", "patient", "x", "y", "lineage_name", main_target, "resid"]
    flags = obs.loc[obs["flag_tier2"], cols].copy()
    flags.to_csv(run.art("tier2"), index=False)
    say(f"tier 2: {len(flags)} cells flagged across {flags['sample'].nunique()} slides -> {run.art('tier2')}")
    show_table(flags.groupby("sample").size().rename("flagged_cells").reset_index(), title="flagged cells per slide")

    tier3_rows = []
    for sample, d in flags.groupby("sample"):
        if len(d) < 50:
            say(f"{sample}: fewer than 50 flagged cells, no clustering", indent=1)
            continue
        labels = DBSCAN(eps=eps_px, min_samples=min_samples).fit_predict(d[["x", "y"]].values)
        n_clusters = len(set(labels) - {-1})
        say(f"{sample}: DBSCAN found {n_clusters} clusters of flagged cells", indent=1)
        for c in sorted(set(labels) - {-1}):
            dd = d[labels == c]
            if len(dd) >= min_cluster_cells:
                tier3_rows.append(dict(sample=sample, n_cells=int(len(dd)),
                                       x_min=float(dd.x.min()), x_max=float(dd.x.max()),
                                       y_min=float(dd.y.min()), y_max=float(dd.y.max())))
    tier3 = pd.DataFrame(tier3_rows, columns=["sample", "n_cells", "x_min", "x_max", "y_min", "y_max"])
    tier3.to_csv(run.art("tier3"), index=False)
    say(f"tier 3: {len(tier3)} regions with >= {min_cluster_cells} systematically mispredicted cells -> {run.art('tier3')}")
    if len(tier3):
        show_table(tier3, title="re-image requests")
    return flags, tier3
