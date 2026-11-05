"""Perturbation robustness: do morphology-only predictions survive scanner-like changes?

The morphology-only path is run once on a slide's clean image and once per perturbation
(``validation_perturbations``) on the same nuclei, and the predictions are compared
target by target: Spearman correlation between perturbed and clean predictions, and the
mean absolute shift in standardised units. The perturbation set is a proxy for exposure,
focus, orientation and sensor noise differences between scanners and sites; it does not
replace a multi-site study with real slides.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say, show_table, timed
from .perturbations import apply_perturbation, describe, parse_perturbation


def perturbation_robustness(img: np.ndarray, xy_clean: np.ndarray, checkpoint: Dict[str, Any], cfg, device: str,
                            tokens: Sequence[str], targets: Sequence[str], vmap_fn: Callable, sample: str = "check",
                            seed: int = 0, um_per_px: Optional[float] = None) -> Tuple[pd.DataFrame, dict]:
    """``vmap_fn(img, checkpoint, cfg, device, sample_name, detect_fn, um_per_px=...)`` must return the virtual map
    frame; ``um_per_px`` is the slide's pixel size so the neighbour offsets match training for that slide."""
    for t in tokens:
        parse_perturbation(t)             # fail early on a typo, before any inference
    pred_cols = [t.replace("_z", "_pred") for t in targets]
    with timed(f"clean inference on {sample} ({len(xy_clean)} nuclei)"):
        clean = vmap_fn(img, checkpoint, cfg, device, sample, lambda _im: xy_clean, um_per_px=um_per_px)
    pred_cols = [c for c in pred_cols if c in clean]
    rows: List[dict] = []
    for token in tokens:
        img_p, xy_p = apply_perturbation(img, xy_clean, token, seed=seed)
        with timed(f"inference under {describe(token)}"):
            pert = vmap_fn(img_p, checkpoint, cfg, device, f"{sample}_{token}", lambda _im, _xy=xy_p: _xy, um_per_px=um_per_px)
        if len(pert) != len(clean):
            say(f"{token}: {len(pert)} rows vs {len(clean)} clean rows; nuclei were not kept fixed", indent=1)
            continue
        for c in pred_cols:
            a, b = clean[c].to_numpy(dtype=np.float64), pert[c].to_numpy(dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            rho = float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 3 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan
            rows.append(dict(perturbation=token, description=describe(token), target=c.replace("_pred", "_z"), n_nuclei=int(ok.sum()),
                             spearman_vs_clean=rho, mean_abs_shift=float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else np.nan,
                             mean_shift=float((b[ok] - a[ok]).mean()) if ok.any() else np.nan,
                             clean_sd=float(a[ok].std()) if ok.any() else np.nan))
    table = pd.DataFrame(rows)
    headline: dict = {}
    if not table.empty:
        main = targets[0].replace("_z", "_pred") if len(targets) else None
        tm = table[table["target"] == (main.replace("_pred", "_z") if main else "")]
        headline = dict(main_target=main.replace("_pred", "_z") if main else None, n_perturbations=int(table["perturbation"].nunique()),
                        min_spearman_main=float(tm["spearman_vs_clean"].min()) if len(tm) else np.nan,
                        median_spearman_all=float(table["spearman_vs_clean"].median()),
                        max_mean_abs_shift_main=float(tm["mean_abs_shift"].max()) if len(tm) else np.nan,
                        worst_perturbation_main=str(tm.loc[tm["spearman_vs_clean"].idxmin(), "perturbation"]) if len(tm) and tm["spearman_vs_clean"].notna().any() else None)
        show_table(table.pivot_table(index="perturbation", columns="target", values="spearman_vs_clean").round(3),
                   title="Spearman between perturbed and clean predictions:")
    return table, headline
