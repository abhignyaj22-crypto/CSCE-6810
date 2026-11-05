"""Split-half reliability of every target: the noise ceiling.

Each target is built from summed transcript counts, so it can be split into two
statistically independent halves by binomial thinning (each transcript lands in
half A or half B with probability one half). The correlation between the two
halves, corrected by the Spearman-Brown formula, estimates how well the target
agrees with an independent re-measurement of the same cells. No image model can
be expected to exceed the square root of that value, so every later correlation
is read against this table.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import say, section, show_table
from .programs import program_score


def split_half(cnt_p: np.ndarray, total: np.ndarray, rng: np.random.Generator):
    """Thin program counts and the remaining counts separately.

    Returns ``((a, total_a), (b, total_b))`` so each half is a valid
    (program count, total count) pair for ``program_score``.
    """
    cnt_p = np.asarray(cnt_p).astype(np.int64)
    total = np.asarray(total).astype(np.int64)
    rest = np.maximum(total - cnt_p, 0)
    a = rng.binomial(cnt_p, 0.5)
    b_rest = rng.binomial(rest, 0.5)
    half_a = (a, a + b_rest)
    half_b = (cnt_p - a, (cnt_p - a) + (rest - b_rest))
    return half_a, half_b


def _spearman_brown(r: float) -> float:
    if not np.isfinite(r) or r <= -1:
        return np.nan
    return 2 * r / (1 + r)


def reliability_table(obs: pd.DataFrame, kept_programs: Dict[str, List[str]], cfg: Config) -> pd.DataFrame:
    """One row per (target, sample): split-half r and its Spearman-Brown correction."""
    section("Target reliability (split-half, binomial thinning)")
    rng = np.random.default_rng(cfg.reliability_seed)
    rows = []
    for name in kept_programs:
        for space in cfg.target_spaces:
            prefix = "" if space == "cell" else "nb_"
            if f"{prefix}cnt_{name}" not in obs:
                say(f"no counts for {name} in space {space}; skipped")
                continue
            for sample in pd.unique(obs["sample"]):
                mask = obs["sample"].to_numpy() == sample
                counts = obs.loc[mask, f"{prefix}cnt_{name}"].to_numpy()
                totals = obs.loc[mask, f"{prefix}cnt_total"].to_numpy()
                (a, ta), (b, tb) = split_half(counts, totals, rng)
                sa, sb = program_score(a, ta), program_score(b, tb)
                r = stats.pearsonr(sa, sb)[0] if sa.std() > 0 and sb.std() > 0 else np.nan
                rows.append(dict(target=f"prog_{name}_{space}_z", sample=sample, r_half=r,
                                 r_sb=_spearman_brown(r), n_cells=int(mask.sum())))
    tab = pd.DataFrame(rows)
    say(f"reliability computed for {tab['target'].nunique() if len(tab) else 0} targets x {tab['sample'].nunique() if len(tab) else 0} samples")
    return tab


def reliability_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Median / min / max of r_sb per target and the attainable correlation ceiling."""
    summary = tab.groupby("target")["r_sb"].agg(["median", "min", "max"]).round(3)
    summary["ceiling_sqrt"] = np.sqrt(summary["median"].clip(lower=0)).round(3)
    show_table(summary, title="target reliability (Spearman-Brown r_sb across samples):")
    say("Read: a model correlating at 'ceiling_sqrt' reproduces the target as well as an independent re-measurement would.")
    return summary


def reliability_lookup(tab: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Map ``(target, sample)`` to the Spearman-Brown reliability.

    Keyed by sample (slide), not patient: reliability is a property of one slide's counts.
    ``evaluation.metrics.score`` aggregates over a patient's samples when they differ.
    """
    return {(str(r.target), str(r.sample)): float(r.r_sb) for r in tab.itertuples()}
