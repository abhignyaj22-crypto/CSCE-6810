"""Patient-level association of predicted program burdens with disease stage: PERMANOVA and a trend test.

Why these two tests and not MANOVA. The cohort has about 21 patients in four stage groups
and seven program variables. A classical one-way MANOVA (Pillai's trace) needs more patients
than variables plus groups, assumes multivariate normality and a common covariance, and the
synthetic run already skipped it as degenerate. The two tests here ask the same two
questions without those requirements:

1. Does the vector of program burdens differ between stages at all?
   PERMANOVA (Anderson 2001, Austral Ecology 26:32-46): a pseudo-F statistic computed from
   the Euclidean distance matrix of z-scored patient vectors, with its null distribution
   obtained by permuting stage labels across patients. It works with more variables than
   patients and makes no distributional assumption beyond exchangeability under the null.
2. Does each program's burden increase (or decrease) along the ordinal disease axis
   control < MGUS < SMM < MM?
   Jonckheere-Terpstra trend test (Jonckheere 1954, Biometrika 41:133-145; Terpstra 1952):
   the sum over ordered group pairs of Mann-Whitney counts, with a two-sided permutation
   p-value, and Benjamini-Hochberg q-values across programs (Benjamini and Hochberg 1995,
   JRSS-B 57:289-300). Spearman of burden against stage rank is printed alongside for
   continuity with the earlier stage_trend summary.

The unit of replication is the patient. Cells are nested in patients and a cell-level test
would report meaningless p-values from pseudo-replication. The classical MANOVA remains
available (``stage_test: manova`` or ``both``) for comparison.

Seeding: PERMANOVA uses the run seed; each program's trend test draws its permutations from
its own generator derived from the run seed (seed * 1000 + program index + 1), so the nulls
are independent across programs rather than one shared permutation sequence. Both p-values
use the (b + 1) / (m + 1) correction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say, section, show_table

RESULT_COLUMNS = ["test", "program", "statistic", "p_value", "q_value", "r2_or_rho", "n_patients", "n_groups",
                  "groups", "note"]


# ----------------------------------------------------------------------------- statistics
def _pseudo_f(D2: np.ndarray, labels: np.ndarray) -> tuple:
    """PERMANOVA pseudo-F and R-squared from a squared Euclidean distance matrix and group labels.

    SS_total = sum of squared distances over all pairs / N; SS_within = sum over groups of the
    same quantity inside the group (Anderson 2001, equations 2 and 3).
    """
    n = len(labels)
    ss_total = D2[np.triu_indices(n, 1)].sum() / n
    ss_within = 0.0
    groups = np.unique(labels)
    for g in groups:
        idx = np.where(labels == g)[0]
        if len(idx) > 1:
            sub = D2[np.ix_(idx, idx)]
            ss_within += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    ss_between = ss_total - ss_within
    a = len(groups)
    df_between, df_within = a - 1, n - a
    if df_within <= 0 or ss_within <= 0:
        return float("nan"), float("nan")
    f = (ss_between / df_between) / (ss_within / df_within)
    r2 = ss_between / ss_total if ss_total > 0 else float("nan")
    return float(f), float(r2)


def permanova(X: np.ndarray, labels: Sequence, n_perm: int = 999, seed: int = 0) -> Dict:
    """PERMANOVA on rows of ``X`` (one patient per row) against ``labels``.

    Columns are z-scored first so every program contributes on the same scale; the distance
    is Euclidean. Returns pseudo-F, permutation p-value, R-squared and the null summary.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray([str(v) for v in labels])
    n, p = X.shape
    group_sizes = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    say(f"PERMANOVA: {n} patients x {p} variables, groups {group_sizes}, "
        f"{n_perm} permutations, Euclidean distance on z-scored columns")
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    sq = (Z ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    observed, r2 = _pseudo_f(D2, labels)
    if not np.isfinite(observed):
        say("PERMANOVA: pseudo-F undefined (no within-group variation or no residual degrees of freedom)")
        return {"status": "skipped", "reason": "pseudo-F undefined", "n_patients": int(n), "n_groups": int(len(np.unique(labels)))}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _pseudo_f(D2, rng.permutation(labels))[0]
    null = null[np.isfinite(null)]
    # +1 corrections keep the p-value away from exactly zero (Phipson and Smyth 2010)
    p_value = (np.sum(null >= observed) + 1) / (len(null) + 1)
    out = {"status": "ok", "pseudo_F": observed, "p_value": float(p_value), "r2": r2, "n_patients": int(n),
           "n_groups": int(len(np.unique(labels))), "n_perm": int(len(null)),
           "null_median": float(np.median(null)), "null_95": float(np.percentile(null, 95))}
    say(f"PERMANOVA: pseudo-F = {observed:.3f} (null median {out['null_median']:.3f}, 95th pct {out['null_95']:.3f}), "
        f"R2 = {r2:.3f}, permutation p = {p_value:.3g}")
    return out


def _jt_statistic(values: np.ndarray, ranks: np.ndarray) -> float:
    """Sum over ordered group pairs (i < j) of the Mann-Whitney count of value_j > value_i (ties count one half)."""
    stat = 0.0
    levels = np.unique(ranks)
    for a_idx in range(len(levels)):
        va = values[ranks == levels[a_idx]]
        for b_idx in range(a_idx + 1, len(levels)):
            vb = values[ranks == levels[b_idx]]
            diff = vb[:, None] - va[None, :]
            stat += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return stat


def jonckheere_terpstra(values: Sequence[float], ordinal_labels: Sequence[int], n_perm: int = 999, seed: int = 0) -> Dict:
    """Jonckheere-Terpstra test for a monotone trend of ``values`` across ordered groups.

    ``ordinal_labels`` are integers whose order is the hypothesised order (control 0 < MGUS 1 < ...).
    The two-sided permutation p-value counts permutations whose statistic is at least as far from
    its expectation (half the number of cross-group pairs) as the observed one.
    """
    values = np.asarray(values, dtype=float)
    ranks = np.asarray(ordinal_labels, dtype=int)
    ok = np.isfinite(values)
    values, ranks = values[ok], ranks[ok]
    levels, counts = np.unique(ranks, return_counts=True)
    n = len(values)
    if len(levels) < 2 or n < 3:
        say(f"Jonckheere-Terpstra: skipped ({len(levels)} ordered group(s), {n} patients)")
        return {"status": "skipped", "reason": f"{len(levels)} ordered group(s), {n} patients", "n_patients": int(n),
                "n_groups": int(len(levels))}
    n_pairs = (n * n - (counts ** 2).sum()) / 2.0
    expectation = n_pairs / 2.0
    observed = _jt_statistic(values, ranks)
    rng = np.random.default_rng(seed)
    null = np.array([_jt_statistic(rng.permutation(values), ranks) for _ in range(n_perm)])
    p_value = (np.sum(np.abs(null - expectation) >= abs(observed - expectation)) + 1) / (n_perm + 1)
    rho = float(stats.spearmanr(ranks, values)[0]) if n >= 4 and values.std() > 0 else float("nan")
    direction = "increasing" if observed > expectation else ("decreasing" if observed < expectation else "flat")
    say(f"Jonckheere-Terpstra: JT = {observed:.1f} (expected {expectation:.1f} under no trend, {n_pairs:.0f} cross-group pairs), "
        f"{direction}, two-sided permutation p = {p_value:.3g}, Spearman rho vs stage rank = {rho:.3f}")
    return {"status": "ok", "JT": observed, "expected": expectation, "p_value": float(p_value), "spearman_rho": rho,
            "direction": direction, "n_patients": int(n), "n_groups": int(len(levels)), "n_perm": int(n_perm)}


def bh_qvalues(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone, capped at 1); NaN p-values stay NaN and do not count."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ----------------------------------------------------------------------------- the table
def _collapse_stages(stages: pd.Series, collapse_rule: Optional[Dict[str, str]]) -> pd.Series:
    if not collapse_rule:
        return stages.astype(str)
    return stages.astype(str).map(lambda s: collapse_rule.get(s, s))


def stage_association_table(burden_table: pd.DataFrame, program_columns: Sequence[str], stage_col: str = "stage",
                            stage_order: Optional[Dict[str, int]] = None, n_perm: int = 999, seed: int = 0,
                            collapse_rule: Optional[Dict[str, str]] = None, min_per_group: int = 2) -> pd.DataFrame:
    """PERMANOVA over all programs plus one trend test per program; one row per test.

    ``burden_table`` has one row per patient with a stage column and one column per program.
    ``stage_order`` maps stage labels (after collapsing) to their ordinal rank; stages without a
    rank are excluded from the trend test but kept in the PERMANOVA. Groups with fewer than
    ``min_per_group`` patients are dropped from both tests and named in the note column.
    """
    section("stage association: PERMANOVA (all programs) and Jonckheere-Terpstra trend (per program)")
    cols = [c for c in program_columns if c in burden_table.columns]
    tab = burden_table.dropna(subset=cols, how="all").copy()
    tab["_stage"] = _collapse_stages(tab[stage_col], collapse_rule)
    if collapse_rule:
        say("stages collapsed: " + ", ".join(f"{k}->{v}" for k, v in collapse_rule.items()))
    counts = tab["_stage"].value_counts()
    small = counts[counts < min_per_group].index.tolist()
    if small:
        say(f"dropping groups with fewer than {min_per_group} patients: {small}")
        tab = tab[~tab["_stage"].isin(small)]
    groups_present = sorted(tab["_stage"].unique().tolist())
    groups_txt = ";".join(f"{g}:{int((tab['_stage'] == g).sum())}" for g in groups_present)
    dropped_txt = f"; dropped {small}" if small else ""
    rows: List[Dict] = []

    # ---- PERMANOVA on the full vector --------------------------------------------------
    X = tab[cols].to_numpy(dtype=float)
    complete = np.isfinite(X).all(axis=1)
    if len(groups_present) < 2 or complete.sum() < 3:
        note = f"skipped: {len(groups_present)} group(s), {int(complete.sum())} patients with complete vectors{dropped_txt}"
        say(f"PERMANOVA {note}")
        rows.append(dict(test="permanova", program="all", statistic=np.nan, p_value=np.nan, q_value=np.nan,
                         r2_or_rho=np.nan, n_patients=int(complete.sum()), n_groups=len(groups_present),
                         groups=groups_txt, note=note))
    else:
        res = permanova(X[complete], tab["_stage"].to_numpy()[complete], n_perm=n_perm, seed=seed)
        if res["status"] == "ok":
            rows.append(dict(test="permanova", program="all", statistic=res["pseudo_F"], p_value=res["p_value"],
                             q_value=np.nan, r2_or_rho=res["r2"], n_patients=res["n_patients"], n_groups=res["n_groups"],
                             groups=groups_txt, note=f"pseudo-F on {len(cols)} z-scored programs, Euclidean, "
                                                     f"{res['n_perm']} permutations{dropped_txt}"))
        else:
            rows.append(dict(test="permanova", program="all", statistic=np.nan, p_value=np.nan, q_value=np.nan,
                             r2_or_rho=np.nan, n_patients=res["n_patients"], n_groups=res["n_groups"], groups=groups_txt,
                             note=f"skipped: {res['reason']}{dropped_txt}"))

    # ---- trend test per program ---------------------------------------------------------
    order = stage_order or {}
    ranks = tab["_stage"].map(order)
    has_rank = ranks.notna().to_numpy()
    if has_rank.sum() < len(tab):
        say(f"trend test: {int((~has_rank).sum())} patient(s) whose stage has no ordinal rank are left out of the trend test")
    trend_rows: List[Dict] = []
    for k_prog, c in enumerate(cols):
        v = tab[c].to_numpy(dtype=float)
        use = has_rank & np.isfinite(v)
        say(f"program {c}:")
        # one null per program, derived from the base seed, so the permutation nulls are independent across programs
        res = jonckheere_terpstra(v[use], ranks.to_numpy()[use].astype(int), n_perm=n_perm, seed=int(seed) * 1000 + k_prog + 1)
        if res["status"] == "ok":
            trend_rows.append(dict(test="jt_trend", program=c, statistic=res["JT"], p_value=res["p_value"], q_value=np.nan,
                                   r2_or_rho=res["spearman_rho"], n_patients=res["n_patients"], n_groups=res["n_groups"],
                                   groups=groups_txt, note=f"{res['direction']} along the stage order; expected JT {res['expected']:.1f}; "
                                                           f"{res['n_perm']} permutations{dropped_txt}"))
        else:
            trend_rows.append(dict(test="jt_trend", program=c, statistic=np.nan, p_value=np.nan, q_value=np.nan,
                                   r2_or_rho=np.nan, n_patients=res["n_patients"], n_groups=res["n_groups"],
                                   groups=groups_txt, note=f"skipped: {res['reason']}{dropped_txt}"))
    if trend_rows:
        q = bh_qvalues([r["p_value"] for r in trend_rows])
        for r, qv in zip(trend_rows, q):
            r["q_value"] = float(qv) if np.isfinite(qv) else np.nan
        say(f"Benjamini-Hochberg over {int(np.isfinite(q).sum())} trend tests: " +
            ", ".join(f"{r['program']} q={r['q_value']:.3g}" for r in trend_rows if np.isfinite(r['q_value'])))
    rows.extend(trend_rows)

    out = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    show_table(out.round(4), title="stage association (patient level)")
    say("caveat: few patients per group; a non-significant result is absence of evidence, not evidence of absence, "
        "and stage is a proxy for progression, not a measurement of drug response")
    return out
