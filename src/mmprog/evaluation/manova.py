"""Multivariate test of program-score vectors against disease stage, at the patient level.

Why patient level. Cells are nested in patients: hundreds of thousands of cells
from 21 trephines are not hundreds of thousands of independent observations, and
a cell-level MANOVA would report absurdly small p-values from pseudo-replication.
The unit of replication is the patient, so each patient contributes one vector
of mean program scores (measured) and one vector of mean predicted scores.

Why two tests.
- Parametric one-way MANOVA (statsmodels) with Pillai's trace, which is the
  statistic most robust to unequal group sizes and heterogeneous covariance
  (Olson 1974, Journal of the American Statistical Association 69:894-908; see also Olson 1976, Psychological Bulletin 83:579-586).
- A permutation MANOVA that uses the same Pillai statistic but obtains its null
  distribution by permuting stage labels across patients (the PERMANOVA idea,
  Anderson, 2001, Austral Ecology 26:32), so it needs no multivariate normality.

Caveat printed with every result: 21 patients in three collapsed groups (control 4, precursor 7, myeloma 10) is at the lower limit for a five-variable MANOVA (the composite wnt_camdr set is omitted because it is a function of its two subsets). The result is reported as evidence of a stage effect on the program
vector, never as a per-patient classifier.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say, section, show_table, warn

CAVEAT = ("caveat: patient-level test with few patients per group; Pillai's trace reported for robustness "
          "to unequal groups (Olson 1974); the permutation p-value does not assume multivariate normality")


def _nb_targets(targets: Sequence[str]) -> List[str]:
    """Neighbourhood-scale targets when present, otherwise everything."""
    nb = [t for t in targets if t.endswith("_nb_z")]
    return nb if nb else list(targets)


def patient_level_table(obs: pd.DataFrame, pred_full: np.ndarray, targets: Sequence[str],
                        stage_col: str = "stage") -> pd.DataFrame:
    """One row per patient: stage, n_cells, mean measured and mean predicted score per target."""
    use = _nb_targets(targets)
    df = pd.DataFrame({"patient": obs["patient"].values, stage_col: obs[stage_col].values})
    for t in use:
        j = list(targets).index(t)
        df[t] = obs[t].values
        df[f"{t}_pred"] = pred_full[:, j]
    grouped = df.groupby("patient", observed=True)
    table = grouped[[c for c in df.columns if c not in ("patient", stage_col)]].mean()
    table.insert(0, stage_col, grouped[stage_col].first())
    table.insert(1, "n_cells", grouped.size())
    table = table.reset_index()
    say(f"patient-level table: {len(table)} patients, {len(use)} targets ({', '.join(use)})")
    return table


def _pillai_from_sscp(X: np.ndarray, labels: np.ndarray) -> float:
    """Pillai's trace = tr(B (B + W)^-1) from between- and within-group SSCP matrices."""
    grand = X.mean(axis=0)
    B = np.zeros((X.shape[1], X.shape[1]))
    W = np.zeros_like(B)
    for g in np.unique(labels):
        Xg = X[labels == g]
        d = (Xg.mean(axis=0) - grand)[:, None]
        B += len(Xg) * (d @ d.T)
        R = Xg - Xg.mean(axis=0)
        W += R.T @ R
    T = B + W
    # pseudo-inverse keeps the statistic finite when a group has fewer rows than variables
    return float(np.trace(B @ np.linalg.pinv(T)))


def _drop_small_groups(table: pd.DataFrame, group_col: str, min_per_group: int) -> pd.DataFrame:
    counts = table[group_col].value_counts()
    small = counts[counts < min_per_group].index.tolist()
    if small:
        say(f"dropping groups with fewer than {min_per_group} patients: {small}", indent=1)
    return table[~table[group_col].isin(small)].copy()


def _degenerate(table: pd.DataFrame, columns: Sequence[str], group_col: str) -> Optional[str]:
    n_groups = table[group_col].nunique()
    if n_groups < 2:
        return f"only {n_groups} group(s) after filtering; a between-group test needs at least 2"
    if len(table) <= len(columns) + n_groups:
        return (f"n = {len(table)} patients is not above the number of variables ({len(columns)}) plus "
                f"groups ({n_groups}); the within-group covariance would be singular")
    return None


def manova_by_stage(table: pd.DataFrame, columns: Sequence[str], group_col: str = "stage",
                    min_per_group: int = 2) -> Dict:
    """Parametric one-way MANOVA on the patient table; returns Pillai's trace, F, df, p and partial eta squared.

    Partial eta squared is computed as Pillai / s with s = min(number of variables, groups - 1),
    the usual multivariate effect size for Pillai's trace (Tabachnick and Fidell).
    """
    section(f"MANOVA (parametric) of {list(columns)} on {group_col}")
    say(CAVEAT)
    tab = table.dropna(subset=list(columns)).copy()
    tab[group_col] = tab[group_col].astype(str)    # a categorical with an unused level gives statsmodels a zero column
    tab = _drop_small_groups(tab, group_col, min_per_group)
    reason = _degenerate(tab, columns, group_col)
    if reason:
        say(f"skipped: {reason}")
        return {"status": "skipped", "reason": reason}

    from statsmodels.multivariate.manova import MANOVA

    # statsmodels formulas need plain identifiers, so the DV columns are renamed v0..vk
    renamed = {c: f"v{i}" for i, c in enumerate(columns)}
    tab = tab.rename(columns=renamed)
    formula = " + ".join(renamed.values()) + f" ~ C({group_col})"
    res = MANOVA.from_formula(formula, data=tab).mv_test()
    stat = res.results[f"C({group_col})"]["stat"]
    row = stat.loc["Pillai's trace"]
    n_groups = tab[group_col].nunique()
    s = min(len(columns), n_groups - 1)
    out = {
        "status": "ok", "test": "manova_pillai", "n_patients": int(len(tab)), "n_groups": int(n_groups),
        "pillai": float(row["Value"]), "F": float(row["F Value"]),
        "df_num": float(row["Num DF"]), "df_den": float(row["Den DF"]), "p_value": float(row["Pr > F"]),
        "partial_eta2": float(row["Value"]) / s,
    }
    say(f"Pillai's trace {out['pillai']:.3f}, F({out['df_num']:.0f}, {out['df_den']:.0f}) = {out['F']:.2f}, "
        f"p = {out['p_value']:.3g}, partial eta2 = {out['partial_eta2']:.3f}")
    # homogeneity-of-covariance diagnostic: Box's M is not in statsmodels, so the group covariance
    # determinants are printed; very different magnitudes signal heterogeneity (Pillai tolerates it best)
    diag = []
    for g, d in tab.groupby(group_col, observed=True):
        X = d[list(renamed.values())].to_numpy(dtype=float)
        det = float(np.linalg.det(np.cov(X, rowvar=False))) if len(X) > len(columns) else float("nan")
        diag.append({group_col: g, "n": len(X), "log10_cov_det": np.log10(det) if det > 0 else float("nan")})
    show_table(pd.DataFrame(diag), title="per-group covariance determinants (homogeneity diagnostic; nan = too few patients)")
    say("assumption: MANOVA assumes multivariate normality and equal covariance across groups; see the permutation test")
    return out


def permutation_manova(table: pd.DataFrame, columns: Sequence[str], group_col: str = "stage",
                       n_perm: int = 999, seed: int = 0, min_per_group: int = 2) -> Dict:
    """Permutation MANOVA: Pillai's trace with stage labels permuted across patients."""
    section(f"MANOVA (permutation, {n_perm} permutations) of {list(columns)} on {group_col}")
    tab = table.dropna(subset=list(columns)).copy()
    tab[group_col] = tab[group_col].astype(str)
    tab = _drop_small_groups(tab, group_col, min_per_group)
    reason = _degenerate(tab, columns, group_col)
    if reason:
        say(f"skipped: {reason}")
        return {"status": "skipped", "reason": reason}
    X = tab[list(columns)].to_numpy(dtype=float)
    labels = tab[group_col].to_numpy()
    observed = _pillai_from_sscp(X, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _pillai_from_sscp(X, rng.permutation(labels))
    # +1 corrections keep the p-value away from exactly zero (Phipson and Smyth 2010)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    out = {"status": "ok", "test": "permutation_pillai", "n_patients": int(len(tab)),
           "n_groups": int(tab[group_col].nunique()), "pillai": observed, "p_value": float(p), "n_perm": n_perm,
           "null_median": float(np.median(null)), "null_95": float(np.percentile(null, 95))}
    say(f"observed Pillai {observed:.3f} vs null median {out['null_median']:.3f} (95th pct {out['null_95']:.3f}); "
        f"permutation p = {p:.3g}")
    return out


def run_manova_stage(obs: pd.DataFrame, pred_full: np.ndarray, targets: Sequence[str], run, cfg,
                     stage_order: Dict[str, int]) -> pd.DataFrame:
    """Run both tests on measured and on predicted program vectors and write ``manova_results.csv``."""
    section("S5b MANOVA of program vectors by disease stage (patient level)")
    table = patient_level_table(obs, pred_full, targets)
    use = _nb_targets(targets)
    table["stage_ord"] = table["stage"].map(stage_order)
    show_table(table.round(3), title="patient-level means (measured and predicted)")

    rows = []
    for label, cols in (("measured", use), ("predicted", [f"{t}_pred" for t in use])):
        par = manova_by_stage(table, cols)
        perm = permutation_manova(table, cols, seed=cfg.seed)
        rows.append({"vectors": label, **{f"param_{k}": v for k, v in par.items()},
                     **{f"perm_{k}": v for k, v in perm.items()}})
        if par.get("status") == "ok":
            say(f"{label.capitalize()} program vectors by stage: Pillai {par['pillai']:.2f}, "
                f"F({par['df_num']:.0f},{par['df_den']:.0f}) = {par['F']:.2f}, p = {par['p_value']:.3g}, "
                f"partial eta2 = {par['partial_eta2']:.2f}; permutation p = {perm.get('p_value', float('nan')):.3g}")
        else:
            warn(f"{label} program vectors: test skipped ({par.get('reason')})")

    # between-patient validity: does the predicted patient mean track the measured patient mean?
    say("between-patient agreement of patient means (Pearson, measured vs predicted):")
    for t in use:
        ok = table[[t, f"{t}_pred"]].dropna()
        r = stats.pearsonr(ok[t], ok[f"{t}_pred"])[0] if len(ok) >= 3 and ok[t].std() > 0 else float("nan")
        say(f"{t}: r = {r:.3f} over {len(ok)} patients", indent=1)
        # r goes in its own column; the Pillai column stays empty for these rows
        rows.append({"vectors": f"between_patient_r::{t}", "param_status": "ok", "value": r,
                     "param_n_patients": len(ok)})

    results = pd.DataFrame(rows)
    results.to_csv(run.art("manova"), index=False)
    table.to_csv(run.art("manova_patient_table.csv"), index=False)
    say(f"manova results -> {run.art('manova')}")
    return results
