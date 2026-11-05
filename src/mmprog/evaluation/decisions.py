"""Decision rules applied to the cross-validation table.

These rules turn ``cv_results.csv`` into statements that can be written into a
report. They are applied to the numbers, never to a figure, and every statement
carries the values it was derived from.

Rules (margin is an absolute difference in per-patient median Pearson):
1. above_chance:               full > shuffled + margin
2. morphology_beyond_identity: full > composition_only + margin (and above chance)
3. identity_bottleneck:        oracle > full + margin
4. spatial_prior_useful:       full > no_spatial + margin
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logging_utils import say


def _get(RES: pd.DataFrame, target: str, model: str) -> float:
    if model in RES.columns and target in RES.index:
        v = RES.loc[target, model]
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def apply_decision_rules(
    RES: pd.DataFrame, RES_ADJ: Optional[pd.DataFrame], main_target: str, margin: float = 0.05
) -> Tuple[List[str], Dict[str, bool]]:
    """Return (plain-language statements, verdict flags) for ``main_target``."""
    full = _get(RES, main_target, "full")
    shuffled = _get(RES, main_target, "shuffled")
    comp = _get(RES, main_target, "composition_only")
    oracle = _get(RES, main_target, "oracle")
    no_spatial = _get(RES, main_target, "no_spatial")
    # v3 maps its no_neighbours ablation onto this rule; name the variant that was actually run
    spatial_label = "no_neighbours" if "no_neighbours" in RES.columns else "no_spatial"
    adj = _get(RES_ADJ, main_target, "full") if RES_ADJ is not None else np.nan

    def gt(a: float, b: float) -> bool:
        return bool(a == a and b == b and a > b + margin)

    verdict = {
        "above_chance": gt(full, shuffled),
        "morphology_beyond_identity": gt(full, shuffled) and gt(full, comp),
        "identity_bottleneck": gt(oracle, full),
        "spatial_prior_useful": gt(full, no_spatial),
    }

    statements: List[str] = []
    statements.append(f"Main target: {main_target}. Full model per-patient median r = {full:.3f}"
                      + (f" (reliability-adjusted {adj:.3f})." if adj == adj else "."))
    if verdict["above_chance"]:
        statements.append(f"ABOVE CHANCE: full ({full:.3f}) exceeds the shuffled-label baseline ({shuffled:.3f}) by more than {margin}.")
    else:
        statements.append(f"NOT ABOVE CHANCE: full ({full:.3f}) does not exceed shuffled ({shuffled:.3f}) by {margin}; "
                          "the model has not learned a cell-to-program link on this data.")
    if verdict["morphology_beyond_identity"]:
        statements.append(f"MORPHOLOGY BEYOND IDENTITY: full ({full:.3f}) exceeds composition_only ({comp:.3f}); "
                          "morphology carries program information beyond cell type. Report as the main finding.")
    elif full == full and comp == comp and abs(full - comp) <= margin:
        statements.append(f"CELL-TYPE DETECTOR: full ({full:.3f}) is within {margin} of composition_only ({comp:.3f}); "
                          "the model is recovering cell identity, not a program. Report as such.")
    elif full == full and comp == comp:
        statements.append(f"BELOW IDENTITY BASELINE: full ({full:.3f}) is below composition_only ({comp:.3f}); "
                          "the image features do not yet recover even cell identity for this target.")
    if verdict["identity_bottleneck"]:
        statements.append(f"IDENTITY BOTTLENECK: oracle ({oracle:.3f}) exceeds full ({full:.3f}); "
                          "the next lever is the encoder (pathology-pretrained) or a larger patch, not the head.")
    elif oracle == oracle:
        statements.append(f"NO IDENTITY BOTTLENECK: oracle ({oracle:.3f}) is not above full ({full:.3f}) by {margin}.")
    if verdict["spatial_prior_useful"]:
        statements.append(f"SPATIAL PRIOR USEFUL: full ({full:.3f}) exceeds {spatial_label} ({no_spatial:.3f}).")
    elif no_spatial == no_spatial:
        statements.append(f"SPATIAL PRIOR NOT USEFUL: full ({full:.3f}) vs {spatial_label} ({no_spatial:.3f}); "
                          "the prior earns nothing on this data. Drop it or say so.")

    # one line over all targets
    if "full" in RES.columns:
        n_above = 0
        n_total = 0
        for t in RES.index:
            f, s = _get(RES, t, "full"), _get(RES, t, "shuffled")
            if f == f and s == s:
                n_total += 1
                n_above += int(f > s + margin)
        statements.append(f"Across all targets: {n_above}/{n_total} exceed the shuffled baseline by {margin}.")

    say("decision rules:")
    for s in statements:
        say(s, indent=1)
    return statements, verdict
