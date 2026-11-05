"""Evaluation component and calibration on planted out-of-fold predictions."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from mmprog.config import Config  # noqa: E402
from mmprog.core.context import Context, Fold  # noqa: E402
from mmprog.evaluation.calibration import (calibration_from_folds, conformal_coverage, conformal_quantile,  # noqa: E402
                                           coverage_table, expected_calibration_error)


def _planted_predictions(n_pat=6, cells=200, T=4, seed=0):
    rng = np.random.default_rng(seed)
    N = n_pat * cells
    Y = rng.normal(size=(N, T)).astype(np.float32)
    groups = np.repeat([f"P{i}" for i in range(n_pat)], cells)
    std = np.full((N, T), 0.5, dtype=np.float32)
    mean = (Y + 0.5 * rng.normal(size=(N, T))).astype(np.float32)      # calibrated: residual sd equals the stated std
    return mean, std, Y, groups


def test_coverage_table_is_calibrated_on_planted_data():
    mean, std, Y, groups = _planted_predictions()
    targets = [f"t{j}" for j in range(Y.shape[1])]
    tab = coverage_table(mean, std, Y, groups, targets, levels=(0.5, 0.9))
    for level in (0.5, 0.9):
        d = tab[tab["level"] == level]
        assert abs(d["coverage_pooled"].mean() - level) < 0.05
        assert d["n_patients"].min() == 6
    assert expected_calibration_error(mean[:, 0], std[:, 0], Y[:, 0]) < 0.05
    # an over-confident model (std halved) under-covers
    tab2 = coverage_table(mean, std * 0.5, Y, groups, targets, levels=(0.9,))
    assert tab2["coverage_pooled"].mean() < 0.8


def test_conformal_quantile_and_coverage():
    mean, std, Y, groups = _planted_predictions()
    r = np.abs(Y[:, 0] - mean[:, 0]) / std[:, 0]
    q = conformal_quantile(r[:200], alpha=0.1)
    assert 1.4 < q < 1.9                       # about the 90% quantile of |N(0,1)|, 1.645
    tab = conformal_coverage(mean[200:], std[200:], Y[200:], groups[200:], ["t0", "t1", "t2", "t3"], {"t0": q}, alpha=0.1)
    assert len(tab) == 1 and abs(tab["coverage_pooled"].iloc[0] - 0.9) < 0.05
    assert conformal_quantile(np.array([]), 0.1) != conformal_quantile(np.array([]), 0.1)   # NaN for empty


def test_calibration_from_folds_has_both_kinds():
    mean, std, Y, groups = _planted_predictions()
    folds = [Fold(0, ["P0", "P1", "P2", "P3"], "P4", ["P5"]), Fold(1, ["P0", "P1", "P2", "P5"], "P3", ["P4"])]
    tab = calibration_from_folds(mean, std, Y, groups, ["t0", "t1", "t2", "t3"], folds, alpha=0.1, levels=(0.5, 0.9))
    assert set(tab["kind"]) == {"gaussian", "conformal"}
    assert "ece" in tab.columns and tab["ece"].max() < 0.1


def test_conformal_quantile_comes_from_in_fold_validation_predictions():
    """Item 15: with ``val_preds`` the per-fold quantile is computed from the fold's own validation
    predictions, not from the out-of-fold rows of the validation patient. Planting very different
    residual scales on the two paths makes the difference visible in the reported quantile."""
    mean, std, Y, groups = _planted_predictions()
    targets = ["t0", "t1", "t2", "t3"]
    folds = [Fold(0, ["P0", "P1", "P2", "P3"], "P4", ["P5"]), Fold(1, ["P0", "P1", "P2", "P5"], "P3", ["P4"])]
    fallback = calibration_from_folds(mean, std, Y, groups, targets, folds, alpha=0.1, levels=(0.9,))
    q_fallback = fallback[fallback["kind"] == "conformal"].set_index("target")["q"]
    # in-fold validation predictions whose residuals are exactly three stated standard deviations
    val_preds = {}
    for f in folds:
        rows = np.where(groups == f.val_patient)[0]
        val_preds[f.fold_id] = dict(rows=rows, mean=(Y[rows] + 3.0 * std[rows]).astype(np.float32), std=std[rows])
    infold = calibration_from_folds(mean, std, Y, groups, targets, folds, alpha=0.1, levels=(0.9,), val_preds=val_preds)
    q_infold = infold[infold["kind"] == "conformal"].set_index("target")["q"]
    assert np.allclose(q_infold.to_numpy(), 3.0, atol=1e-5), q_infold
    assert (q_fallback < 2.2).all(), q_fallback            # the out-of-fold residuals are about N(0,1): q near 1.645
    # the wider in-fold quantile is applied to the test patients and therefore covers more of them
    cov_in = infold[infold["kind"] == "conformal"]["coverage_pooled"].mean()
    cov_fb = fallback[fallback["kind"] == "conformal"]["coverage_pooled"].mean()
    assert cov_in > cov_fb


def _context_with_predictions(tmp_path):
    rng = np.random.default_rng(1)
    n_pat, cells = 6, 150
    N = n_pat * cells
    targets = ["prog_wnt_camdr_nb_z", "prog_wnt_core_nb_z", "prog_adhesion_survival_nb_z", "prog_t_exhaustion_nb_z",
               "anchor_wnt_camdr_nb_z"]
    family = {t: ("anchor_only" if t.startswith("anchor_") else "learned") for t in targets}
    Y = rng.normal(size=(N, len(targets))).astype(np.float32)
    groups = np.repeat([f"P{i}" for i in range(n_pat)], cells)
    stage = np.repeat(["control", "control", "SMM", "SMM", "MM", "MM"], cells)
    lin = rng.integers(0, 3, size=N)
    obs = pd.DataFrame({"x": rng.uniform(0, 500, N), "y": rng.uniform(0, 500, N), "sample": groups, "patient": groups,
                        "stage": stage, "lineage_name": np.array(["plasma", "T", "B"])[lin]})
    good = (Y + 0.4 * rng.normal(size=Y.shape)).astype(np.float32)
    bad = rng.normal(size=Y.shape).astype(np.float32)
    nan_anchor = good.copy()
    nan_anchor[:, 4] = np.nan
    anchor_only = np.full_like(Y, np.nan)
    anchor_only[:, 4] = good[:, 4]
    cv_preds = {"full": nan_anchor, "shuffled": bad, "no_neighbours": nan_anchor * 0.9, "oracle": nan_anchor,
                "anchor_only_targets": anchor_only}
    unc = np.full_like(Y, 0.4, dtype=np.float32)
    cfg = Config(root=str(tmp_path), run_name="e", local_data=str(tmp_path / "raw"), demo_mode=True, device="cpu",
                 n_folds=3, n_boot=20, manova_n_perm=99, stage_n_perm=99, coverage_levels=(0.5, 0.9))
    ctx = Context.new(cfg)
    rel = {(t, p): 0.8 for t in targets for p in np.unique(groups)}
    ctx.replace(obs=obs, Y=Y, targets=targets, target_family=family, GROUPS=groups, SID=np.repeat(np.arange(n_pat), cells),
                lineage=lin, LIN=np.eye(3)[lin].astype(np.float32), lin_names=["plasma", "T", "B"], rel_lookup=rel,
                reliability_summary=pd.DataFrame({"median": [0.8] * len(targets)}, index=targets),
                cv_preds=cv_preds, cv_unc={"full": unc},
                folds=[Fold(0, ["P0", "P1", "P2", "P3"], "P4", ["P5"]), Fold(1, ["P0", "P1", "P2", "P5"], "P3", ["P4"]),
                       Fold(2, ["P0", "P1", "P4", "P5"], "P2", ["P3"])])
    return ctx


def test_evaluation_component_end_to_end(tmp_path):
    from mmprog.components.evaluation import Evaluation
    ctx = _context_with_predictions(tmp_path)
    comp = Evaluation()
    ctx = comp.run(ctx)
    res = ctx.results
    assert list(res.columns[:3]) == ["shuffled", "composition_only", "anchor_only_targets"]
    assert res.loc["prog_wnt_camdr_nb_z", "full"] > res.loc["prog_wnt_camdr_nb_z", "shuffled"] + 0.3
    assert np.isfinite(res.loc["anchor_wnt_camdr_nb_z", "anchor_only_targets"])
    assert ctx.verdict["above_chance"] and any(s.startswith("LEARNED VS FIXED") for s in ctx.decisions)
    assert len(ctx.calibration) and "gaussian" in set(ctx.calibration["kind"])
    assert len(ctx.burden)
    # PERMANOVA + trend replace MANOVA by default: one PERMANOVA row and one trend row per program, for predicted and measured
    sa = ctx.stage_assoc
    assert set(sa["test"]) == {"permanova", "jt_trend"} and set(sa["vectors"]) == {"predicted", "measured"}
    assert ctx.manova is None and not os.path.exists(comp.path(ctx, "manova_results.csv"))
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    # reload path restores the same tables
    ctx2 = Context.new(ctx.cfg)
    for f in ("obs", "Y", "targets", "target_family", "GROUPS", "SID", "lineage", "LIN", "lin_names", "rel_lookup",
              "reliability_summary", "cv_preds", "cv_unc", "folds"):
        setattr(ctx2, f, getattr(ctx, f))
    ctx2.component_manifests = dict(ctx.component_manifests)
    comp.load(ctx2)
    pd.testing.assert_frame_equal(ctx2.results, res)
    assert ctx2.verdict == ctx.verdict
    assert list(ctx2.stage_assoc.columns) == list(ctx.stage_assoc.columns) and len(ctx2.stage_assoc) == len(ctx.stage_assoc)


def test_evaluation_component_runs_manova_when_asked(tmp_path):
    from mmprog.components.evaluation import Evaluation
    ctx = _context_with_predictions(tmp_path)
    ctx.cfg.stage_test = "both"
    ctx = Evaluation().run(ctx)
    assert ctx.manova is not None and len(ctx.manova) and len(ctx.stage_assoc)


# ----------------------------------------------------------------------------- stage association statistics
def test_permanova_detects_planted_group_difference_and_not_exchangeable_data():
    from mmprog.evaluation.stage_association import permanova
    rng = np.random.default_rng(0)
    labels = np.repeat(["a", "b", "c"], 8)
    X = rng.normal(size=(24, 5))
    shifted = X.copy()
    shifted[labels == "b"] += 2.0
    shifted[labels == "c"] -= 2.0
    res = permanova(shifted, labels, n_perm=499, seed=1)
    assert res["status"] == "ok" and res["p_value"] < 0.05 and res["r2"] > 0.3
    # under exchangeability the p-value is roughly uniform: check several draws, not one
    null_p = [permanova(rng.normal(size=(24, 5)), labels, n_perm=199, seed=k)["p_value"] for k in range(6)]
    assert np.median(null_p) > 0.2 and sum(v < 0.05 for v in null_p) <= 1


def test_permanova_handles_more_variables_than_patients():
    from mmprog.evaluation.stage_association import permanova
    rng = np.random.default_rng(3)
    labels = np.repeat(["a", "b"], 4)
    X = rng.normal(size=(8, 12))
    X[labels == "b"] += 3.0
    res = permanova(X, labels, n_perm=199, seed=0)
    assert res["status"] == "ok" and np.isfinite(res["pseudo_F"]) and res["p_value"] < 0.1


def test_jonckheere_terpstra_detects_monotone_trend_and_not_flat():
    from mmprog.evaluation.stage_association import jonckheere_terpstra
    rng = np.random.default_rng(0)
    ranks = np.repeat([0, 1, 2, 3], 6)
    trend = ranks + 0.3 * rng.normal(size=24)
    res = jonckheere_terpstra(trend, ranks, n_perm=499, seed=0)
    assert res["status"] == "ok" and res["p_value"] < 0.01 and res["direction"] == "increasing" and res["spearman_rho"] > 0.8
    flat_p = [jonckheere_terpstra(rng.normal(size=24), ranks, n_perm=199, seed=k)["p_value"] for k in range(6)]
    assert np.median(flat_p) > 0.2 and sum(v < 0.05 for v in flat_p) <= 1
    # exact statistic on a hand-checked case: groups [1,2] < [3,4]: every cross pair counts, JT = 4
    assert jonckheere_terpstra([1, 2, 3, 4], [0, 0, 1, 1], n_perm=19, seed=0)["JT"] == 4.0


def test_bh_qvalues_monotone_and_bounded():
    from mmprog.evaluation.stage_association import bh_qvalues
    p = np.array([0.001, 0.01, 0.04, 0.2, np.nan, 0.9])
    q = bh_qvalues(p)
    assert np.isnan(q[4])
    finite = q[np.isfinite(q)]
    order = np.argsort(p[np.isfinite(p)])
    assert np.all(np.diff(finite[order]) >= -1e-12) and finite.max() <= 1.0
    assert abs(q[0] - 0.005) < 1e-9       # 0.001 * 5 / 1


def test_stage_association_table_degenerate_input_writes_skipped_rows():
    from mmprog.evaluation.stage_association import stage_association_table
    burden = pd.DataFrame({"patient": ["P0", "P1", "P2"], "stage": ["MM", "MM", "MM"], "prog_a": [0.1, 0.2, 0.3]})
    out = stage_association_table(burden, ["prog_a"], stage_order={"MM": 3}, n_perm=19, seed=0)
    assert len(out) == 2 and out["note"].str.startswith("skipped:").all()
    assert list(out.columns) == ["test", "program", "statistic", "p_value", "q_value", "r2_or_rho", "n_patients",
                                 "n_groups", "groups", "note"]


def test_stage_association_table_on_planted_burdens():
    from mmprog.evaluation.stage_association import stage_association_table
    rng = np.random.default_rng(5)
    stages = np.repeat(["control", "MGUS", "SMM", "MM"], 5)
    order = {"control": 0, "MGUS": 1, "SMM": 2, "MM": 3}
    rank = pd.Series(stages).map(order).to_numpy()
    burden = pd.DataFrame({"patient": [f"P{i}" for i in range(20)], "stage": stages,
                           "prog_up": rank + 0.3 * rng.normal(size=20), "prog_flat": rng.normal(size=20)})
    out = stage_association_table(burden, ["prog_up", "prog_flat"], stage_order=order, n_perm=199, seed=0)
    perm = out[out["test"] == "permanova"].iloc[0]
    up = out[(out["test"] == "jt_trend") & (out["program"] == "prog_up")].iloc[0]
    flat = out[(out["test"] == "jt_trend") & (out["program"] == "prog_flat")].iloc[0]
    assert perm["p_value"] < 0.05 and perm["n_groups"] == 4
    assert up["p_value"] < 0.01 and up["q_value"] < 0.05 and up["r2_or_rho"] > 0.8
    assert flat["q_value"] > up["q_value"]

