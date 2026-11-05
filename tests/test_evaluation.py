"""Tests for mmprog.evaluation with stand-in feature space and selector."""
import numpy as np
import pandas as pd
import pytest

from mmprog.config import Config
from mmprog.paths import RunDir
from mmprog.evaluation import (
    build_cv_data, run_cv, composition_only, run_all_variants, run_multiseed,
    score, bootstrap_ci, apply_decision_rules, program_burden, stage_trend, cox_pfs, run_transfer,
)


class FakeFeatureSpace:
    """Standardise embeddings (optionally sliced), append morphometrics."""

    def __init__(self, pca_dim, use_morpho):
        self.pca_dim, self.use_morpho = pca_dim, use_morpho

    def fit(self, emb, mor, idx=None, seed=0, keep_emb_dims=None, keep_mor_dims=None):
        idx = np.arange(len(emb)) if idx is None else idx
        self.keep_emb = np.arange(emb.shape[1]) if keep_emb_dims is None else np.asarray(keep_emb_dims)
        self.keep_mor = np.arange(mor.shape[1]) if keep_mor_dims is None else np.asarray(keep_mor_dims)
        e = np.asarray(emb[idx], dtype=np.float32)[:, self.keep_emb]
        self.mu, self.sd = e.mean(0), e.std(0) + 1e-6
        return self

    def transform(self, emb, mor):
        z = (np.asarray(emb, dtype=np.float32)[:, self.keep_emb] - self.mu) / self.sd
        if self.use_morpho:
            z = np.hstack([z, np.asarray(mor, dtype=np.float32)[:, self.keep_mor]])
        return z.astype(np.float32)

    @property
    def dim(self):
        return len(self.keep_emb) + (len(self.keep_mor) if self.use_morpho else 0)


class FakeSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, emb, mor, Y, obs, train_idx):
        self.keep_emb_dims = np.arange(emb.shape[1])
        self.keep_mor_dims = np.arange(mor.shape[1])
        return self


def _fixture(n_pat=4, cells=120, d=6, t=2, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pat * cells
    EMB = rng.normal(size=(n, d)).astype(np.float32)
    MOR = rng.normal(size=(n, 3)).astype(np.float32)
    W = rng.normal(size=(d, t)).astype(np.float32)
    Y = EMB @ W + 0.2 * rng.normal(size=(n, t)).astype(np.float32)
    pats = np.repeat([f"P{i}" for i in range(n_pat)], cells)
    stages = np.repeat(["control", "MGUS", "SMM", "MM"][:n_pat], cells)
    lin = rng.integers(0, 3, n)
    obs = pd.DataFrame({"x": rng.uniform(0, 500, n), "y": rng.uniform(0, 500, n), "sample": pats, "patient": pats,
                        "stage": stages, "cohort": np.where(pats < "P2", "A", "B"), "lineage": lin,
                        "lineage_name": np.array(["plasma", "T", "B"])[lin],
                        "t0_z": Y[:, 0], "t1_z": Y[:, 1]})
    return obs, EMB, MOR


def _cfg(tmp_path, **kw):
    base = dict(run_name="t", root=str(tmp_path), local_data=str(tmp_path / "raw"), n_folds=4, max_epochs=4,
                patience=2, batch_cells=128, hidden=16, dropout=0.0, pca_dim=4, n_boot=20, seeds=(1, 2),
                cv_variants=("full", "shuffled", "oracle", "no_spatial"), knn_k=3)
    base.update(kw)
    return Config(**base)


def test_run_cv_shapes_and_fold_log(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    preds, Y, log, hist = run_cv("full", data, EMB, MOR, cfg, "cpu", seed=0,
                                 feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    assert preds.shape == data.Y.shape
    assert len(log) == 4
    assert set(hist) == {f"full_s0_f{k}" for k in range(4)}
    assert "val_mse" in hist["full_s0_f0"].columns
    assert set(log.columns) == {"fold", "val_patient", "test_patients", "val_mse", "epochs", "seconds"}
    assert np.allclose(Y, data.Y)
    assert np.abs(preds).sum() > 0


def test_run_cv_shuffled_destroys_link(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path, max_epochs=8)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    _, Ys, _, _ = run_cv("shuffled", data, EMB, MOR, cfg, "cpu", seed=0,
                      feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    assert not np.allclose(Ys, data.Y)
    # marginals preserved within slide
    for s in np.unique(data.SID):
        m = data.SID == s
        assert np.isclose(np.sort(Ys[m, 0]), np.sort(data.Y[m, 0])).all()


def test_composition_only_shape(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    p = composition_only(data, cfg)
    assert p.shape == data.Y.shape


def test_run_all_variants_writes_artefacts(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path, seeds=(7, 2))      # 7 is cfg.seed: must be reused by run_multiseed, 2 trained
    run = RunDir(cfg)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    RES, RES_ADJ, per, logs, pred_full, histories, full_scores = run_all_variants(
        data, EMB, MOR, cfg, "cpu", run, rel_lookup={("t0_z", "P0"): 0.8},
        feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    assert set(histories) == {f"full_s{cfg.seed}_f{k}" for k in range(4)}
    assert full_scores is not None and set(full_scores["target"]) == {"t0_z", "t1_z"}
    assert list(RES.columns) == ["shuffled", "composition_only", "no_spatial", "full", "oracle", "full_95ci"]
    assert set(RES.index) == {"t0_z", "t1_z"}
    assert pred_full.shape == data.Y.shape
    for name in ("cv_results", "cv_results_adj", "cv_per_patient", "cv_fold_log", "pred_full"):
        assert run.done(name)
    assert RES.loc["t0_z", "full"] > RES.loc["t0_z", "shuffled"]
    calls = []
    import mmprog.evaluation.cross_validation as cv_mod
    original_run_cv = cv_mod.run_cv

    def counting_run_cv(*a, **kw):
        calls.append(kw.get("seed"))
        return original_run_cv(*a, **kw)

    cv_mod.run_cv = counting_run_cv
    try:
        tab = run_multiseed(data, EMB, MOR, cfg, "cpu", run, feature_space_factory=FakeFeatureSpace,
                            selector_factory=FakeSelector, precomputed={cfg.seed: full_scores})
    finally:
        cv_mod.run_cv = original_run_cv
    assert list(tab.columns) == ["mean", "std"] and run.done("cv_multiseed")
    # the seed already trained by run_all_variants is reused, not retrained
    assert cfg.seed not in calls and set(calls) == set(cfg.seeds) - {cfg.seed}


def test_score_and_bootstrap():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c"], 50)
    Y = rng.normal(size=(150, 1))
    preds = Y + 0.1 * rng.normal(size=Y.shape)
    tab = score(preds, Y, ["t"], groups, rel_lookup={("t", "a"): 0.5}, min_cells=10)
    assert tab.loc[0, "pearson_patient_median"] > 0.95
    assert tab.loc[0, "n_patients_scored"] == 3
    assert tab.loc[0, "per_patient_adj"]["a"] > tab.loc[0, "per_patient"]["a"]
    assert np.isnan(tab.loc[0, "per_patient_adj"]["b"])          # no reliability for b -> NaN, not a silent 1.0
    # reliability is keyed by sample: patient "a" has two slides, adjusted r uses their median r_sb
    sids = np.where(np.arange(150) < 25, "a_s1", np.where(np.arange(150) < 50, "a_s2", groups))
    tab2 = score(preds, Y, ["t"], groups, rel_lookup={("t", "a_s1"): 0.25, ("t", "a_s2"): 0.64}, min_cells=10, sids=sids)
    expected = tab2.loc[0, "per_patient"]["a"] / np.sqrt(np.median([0.25, 0.64]))
    assert np.isclose(tab2.loc[0, "per_patient_adj"]["a"], expected)
    assert np.isnan(tab.loc[0, "per_patient_adj"]["b"])
    lo, hi = bootstrap_ci(tab.loc[0, "per_patient"], 50, seed=0)
    assert lo <= tab.loc[0, "pearson_patient_median"] <= hi
    assert bootstrap_ci({"a": 0.5, "b": 0.6}, 10) == (np.nan, np.nan) or np.isnan(bootstrap_ci({"a": 0.5}, 10)[0])


def test_decision_rules_verdicts():
    RES = pd.DataFrame({"shuffled": [0.01, 0.08], "composition_only": [0.20, 0.3], "no_spatial": [0.33, 0.1],
                        "full": [0.35, 0.1], "oracle": [0.50, 0.1]}, index=["main_z", "other_z"])
    RES_ADJ = RES * 1.2
    statements, verdict = apply_decision_rules(RES, RES_ADJ, "main_z", margin=0.05)
    assert verdict == {"above_chance": True, "morphology_beyond_identity": True,
                       "identity_bottleneck": True, "spatial_prior_useful": False}
    assert any(s.startswith("MORPHOLOGY BEYOND IDENTITY") for s in statements)
    assert any("1/2 exceed" in s for s in statements)
    _, v2 = apply_decision_rules(RES, None, "other_z")
    assert v2 == {"above_chance": False, "morphology_beyond_identity": False,
                  "identity_bottleneck": False, "spatial_prior_useful": False}


def test_burden_and_stage_trend(tmp_path):
    obs, EMB, MOR = _fixture(n_pat=4, cells=200)
    cfg = _cfg(tmp_path, min_plasma_for_burden=10)
    pred = np.zeros((len(obs), 2), dtype=np.float32)
    stage_ord = {"control": 0, "MGUS": 1, "SMM": 2, "MM": 3}
    # plant burden rising with stage
    pred[:, 0] = obs["stage"].map(stage_ord).values + np.random.default_rng(0).normal(0, 0.3, len(obs))
    burden = program_burden(obs, pred, 0, cfg, stage_ord)
    assert set(burden.columns) >= {"patient", "stage", "n_plasma", "burden", "stage_ord"}
    rho, p, n = stage_trend(burden)
    assert n == 4 and rho > 0.9
    assert cox_pfs(burden, None, str(tmp_path / "cox.csv")) is None


def test_run_transfer(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    out = run_transfer("A", "B", data, EMB, MOR, cfg, "cpu", seed=0,
                       feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    assert list(out["target"]) == ["t0_z", "t1_z"]
    assert (out["n_test_patients"] == 2).all()
    with pytest.raises(ValueError):
        run_transfer("A", "nope", data, EMB, MOR, cfg, "cpu",
                     feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
