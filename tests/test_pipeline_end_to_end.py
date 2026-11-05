"""End-to-end run of every stage on the synthetic fixture with a tiny encoder.

This is the test that proves the stages fit together. It runs on CPU in a few
minutes and checks that the artefacts exist and that the planted structure is
recovered above the shuffled-label bound.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.config import load_config  # noqa: E402
from mmprog.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    cfg = load_config(overrides=[
        "demo_mode=true", "run_name=e2e", f"root={root}", f"local_data={root}/raw",
        "encoder=tiny_test", "pca_dim=16", "n_folds=3", "seeds=[7]",
        "cv_variants=[full,shuffled,no_spatial]", "max_epochs=8", "patience=3", "n_boot=20",
        "max_cells_per_sample=1500", "srb_rounds=40", "srb_lambdas=[0.0,0.5]",
    ])
    pipe = Pipeline(cfg).run_all()
    return pipe


def test_artefacts_exist(finished_run):
    run = finished_run.run
    for name in ["config", "manifest", "adata_raw", "adata_proc", "targets", "reliability", "cv_results",
                 "cv_results_adj", "cv_per_patient", "cv_fold_log", "pred_full", "burden", "final_model",
                 "tier2", "tier3", "overlays", "model_card", "phase_status", "contributions", "run_log",
                 "srb_results", "srb_maps", "srb_rings", "manova", "motifs", "motif_vocab"]:
        assert run.done(name), f"missing artefact {name}"


def test_new_stages_report_honestly(finished_run):
    """Boosting has a lambda = 0 control row; MANOVA records its skip reason at n = 6; motifs find plasma|plasma."""
    srb = pd.read_csv(finished_run.run.art("srb_results"))
    assert (srb["lam"] == 0.0).sum() == 2 and set(srb["loss"]) == {"exponential", "squared"}
    manova = pd.read_csv(finished_run.run.art("manova"))
    assert len(manova) > 0
    motifs = pd.read_csv(finished_run.run.art("motifs"))
    assert any("plasma|plasma" in str(m) for m in motifs["motif"])
    text = open(finished_run.run.art("contributions")).read()
    assert "spatially regularized boosting" in text and "Multivariate test" in text


def test_results_table_shape_and_order(finished_run):
    res = pd.read_csv(finished_run.run.art("cv_results"), index_col=0)
    assert list(res.columns[:4]) == ["shuffled", "composition_only", "no_spatial", "full"]
    assert "full_95ci" in res.columns
    assert set(res.index) == set(finished_run.targets)


def test_planted_structure_recovered(finished_run):
    """The synthetic fixture plants the wnt_camdr program in plasma-cell blobs; the full model must beat chance."""
    res = pd.read_csv(finished_run.run.art("cv_results"), index_col=0)
    t = finished_run.main_target
    assert res.loc[t, "full"] > res.loc[t, "shuffled"] + 0.05, res


def test_reliability_ceiling_is_high_for_neighbourhood_target(finished_run):
    """The planted program is reproducible at the neighbourhood scale and noisier at the single-cell scale."""
    rel = finished_run.reliability_summary
    assert rel.loc["prog_wnt_camdr_nb_z", "median"] > 0.8
    assert rel.loc["prog_wnt_camdr_nb_z", "median"] > rel.loc["prog_wnt_camdr_cell_z", "median"]


def test_model_card_and_contributions(finished_run):
    card = json.load(open(finished_run.run.art("model_card")))
    assert card["verdict"]["above_chance"] in (True, False)
    text = open(finished_run.run.art("contributions")).read()
    assert "Decision rules" in text and "ceiling" in text


def test_rerun_is_idempotent(finished_run, monkeypatch):
    """A second run_all must load every artefact instead of recomputing.

    Three independent checks: the heavy artefacts are not rewritten (mtime), the log says the
    stages were loaded, and no training call happens at all.
    """
    import mmprog.models.train as train_mod
    import mmprog.evaluation.cross_validation as cv_mod
    import mmprog.inference.final_fit as ff_mod

    run = finished_run.run
    heavy = {name: os.path.getmtime(run.art(name)) for name in ("cv_results", "pred_full", "final_model", "adata_proc")}
    before = pd.read_csv(run.art("cv_results"), index_col=0)

    def no_training(*a, **kw):
        raise AssertionError("train_model was called on a resumed run")

    monkeypatch.setattr(train_mod, "train_model", no_training)
    monkeypatch.setattr(cv_mod, "train_model", no_training)
    monkeypatch.setattr(ff_mod, "train_model", no_training)
    Pipeline(finished_run.cfg).run_all()

    after = pd.read_csv(run.art("cv_results"), index_col=0)
    pd.testing.assert_frame_equal(before, after)
    for name, mtime in heavy.items():
        assert os.path.getmtime(run.art(name)) == mtime, f"{name} was rewritten on resume"
    log = open(run.art("run_log")).read()
    assert "S4 loaded" in log and "final model loaded" in log and "S2 loaded" in log
