"""Reporting component on hand-built tables: artefacts, provides, new CONTRIBUTIONS sections, skip rule."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402
from mmprog.components.ingest import Ingest  # noqa: E402
from mmprog.components.preprocess import Preprocess  # noqa: E402
from mmprog.components.reporting import Reporting  # noqa: E402


def _fill_evaluation_fields(ctx):
    targets = ["prog_wnt_camdr_nb_z", "prog_cytotoxicity_nb_z"]
    n = len(ctx.obs)
    rng = np.random.default_rng(0)
    ctx.set("targets", targets)
    for t in targets:
        ctx.obs[t] = rng.normal(size=n)
    ctx.set("programs", {"wnt_camdr": ["WNT5A", "ROR2"], "cytotoxicity": ["GZMB"]})
    ctx.set("morpho_names", ["p_mean"])
    ctx.set("results", pd.DataFrame({"shuffled": [0.01, 0.0], "composition_only": [0.1, 0.2], "full": [0.5, 0.3],
                                     "full_95ci": ["(0.3, 0.6)", "(0.1, 0.4)"]}, index=targets))
    ctx.set("results_adj", pd.DataFrame({"shuffled": [0.01, 0.0], "full": [0.55, 0.35]}, index=targets))
    ctx.set("reliability_summary", pd.DataFrame({"median": [0.9, 0.8], "min": [0.8, 0.7], "max": [0.95, 0.9],
                                                 "ceiling_sqrt": [0.95, 0.89]}, index=targets))
    ctx.set("decisions", ["ABOVE CHANCE: full (0.5) exceeds shuffled (0.01)"])
    ctx.set("verdict", {"above_chance": True, "morphology_beyond_identity": True,
                        "identity_bottleneck": False, "spatial_prior_useful": True})
    ctx.set("cv_preds", {"full": rng.normal(size=(n, len(targets))).astype(np.float32)})
    ctx.set("calibration", pd.DataFrame({"target": targets, "level": [0.9, 0.9], "coverage": [0.88, 0.91]}))
    ctx.set("gene_backbone", pd.DataFrame({"program": ["wnt_camdr", "wnt_camdr"], "gene": ["WNT5A", "ROR2"],
                                           "stability": [1.0, 0.9], "weight": [0.4, 0.3],
                                           "evidence_grade": ["single-study", "single-study"], "class": ["wnt_ligand", "wnt_receptor"]}))
    ctx.set("enrichment", pd.DataFrame({"program": ["wnt_camdr"], "gene_set": ["wnt_core"], "overlap": [2], "p_value": [0.01]}))
    ctx.set("pathway_attr", pd.DataFrame({"program": ["wnt_camdr"], "centre": [0.4], "neighbour_blocks": [0.3], "neighbour_cells": [0.3]}))
    ctx.component_manifests.setdefault("evaluation", {"hash": "e"})
    ctx.component_manifests.setdefault("interpretation", {"hash": "i"})
    ctx.component_manifests.setdefault("external_validation", {"hash": "v"})
    ctx.component_manifests.setdefault("deployment", {"hash": "d"})


def test_reporting_writes_everything(tmp_path, monkeypatch):
    ctx = small_ctx(tmp_path)
    Ingest().run(ctx)
    Preprocess().run(ctx)
    _fill_evaluation_fields(ctx)
    comp = Reporting()
    comp.run(ctx)
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    assert os.path.exists(comp.path(ctx, os.path.join("figures", "overlays.png")))
    assert ctx.model_card["name"].startswith("mmprog") and "sources" in ctx.model_card
    text = open(ctx.contributions_path).read()
    for heading in ("Calibration of the predictive uncertainty", "Gene backbone of each program",
                    "Pathway annotation", "Where the model looks", "What this run does not establish"):
        assert heading in text, heading
    assert "WNT5A" in text and "—" not in text
    assert "Release gates" in text and os.path.exists(os.path.join(ctx.run_dir, "RESULTS.json"))
    assert isinstance(ctx.gates, dict) and "above_chance" in ctx.gates

    calls = {"n": 0}

    def boom(self, c):
        calls["n"] += 1
        raise AssertionError("recomputed")

    monkeypatch.setattr(Reporting, "_run", boom)
    assert comp.is_done(ctx)
    comp.run(ctx)
    assert calls["n"] == 0
    assert ctx.contributions_path.endswith("CONTRIBUTIONS.md")


def _plant_run_dir(root: str, full=0.5, shuffled=0.01, comp=0.1, ci="(0.2, 0.6)", coverage=0.9, n_backbone=3,
                   parity=None, stage=None, validation=None):
    """A minimal run directory with just the artefacts the gates read."""
    os.makedirs(os.path.join(root, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(root, "interpretation"), exist_ok=True)
    pd.DataFrame({"target": ["prog_wnt_camdr_nb_z"], "full": [full], "shuffled": [shuffled],
                  "composition_only": [comp], "full_95ci": [ci]}).to_csv(os.path.join(root, "evaluation", "cv_results.csv"), index=False)
    pd.DataFrame({"target": ["prog_wnt_camdr_nb_z"] * 2, "level": [0.5, 0.9], "coverage_pooled": [0.5, coverage],
                  "kind": ["gaussian", "gaussian"]}).to_csv(os.path.join(root, "evaluation", "calibration.csv"), index=False)
    pd.DataFrame({"program": ["wnt_camdr"] * n_backbone + ["cytotoxicity"], "gene": [f"G{i}" for i in range(n_backbone)] + ["GZMB"],
                  "selection_fraction": [1.0] * n_backbone + [1.0]}).to_csv(os.path.join(root, "interpretation", "gene_backbone.csv"), index=False)
    if parity is not None:
        os.makedirs(os.path.join(root, "deployment"), exist_ok=True)
        pd.DataFrame({"slide": ["S"], "pearson": [0.3], "parity_pearson": [parity]}).to_csv(
            os.path.join(root, "deployment", "sanity_check.csv"), index=False)
    if stage is not None:
        pd.DataFrame(stage).to_csv(os.path.join(root, "evaluation", "stage_association.csv"), index=False)
    if validation is not None:
        import json
        os.makedirs(os.path.join(root, "external_validation"), exist_ok=True)
        with open(os.path.join(root, "external_validation", "validation_summary.json"), "w") as fh:
            json.dump(validation, fh)


def test_release_gates_pass_fail_and_not_evaluated(tmp_path):
    from mmprog.reporting.gates import evaluate_gates, write_gates
    root = str(tmp_path / "run")
    _plant_run_dir(root, parity=0.99,
                   stage=[{"vectors": "predicted", "test": "permanova", "program": "all", "p_value": 0.02, "q_value": None, "note": ""},
                          {"vectors": "predicted", "test": "jt_trend", "program": "wnt_camdr", "p_value": 0.3, "q_value": 0.4, "note": ""},
                          {"vectors": "measured", "test": "permanova", "program": "all", "p_value": 0.9, "q_value": None, "note": ""}],
                   validation={"specificity": {"status": "ran", "passed": True}, "cohort": {"status": "skipped", "reason": "no manifest"}})
    gates = {g.name: g for g in evaluate_gates(root, "prog_wnt_camdr_nb_z")}
    assert gates["above_chance"].passed is True and abs(gates["above_chance"].value - 0.49) < 1e-9
    assert gates["beyond_identity"].passed is True
    assert gates["ci_excludes_zero"].passed is True and gates["ci_excludes_zero"].value == 0.2
    assert gates["shuffled_at_chance"].passed is True
    assert gates["calibrated"].passed is True
    assert gates["backbone_stable"].passed is True and gates["backbone_stable"].value == 3
    assert gates["inference_parity"].passed is True
    assert gates["stage_association"].passed is True and gates["stage_association"].value == 0.02
    assert gates["external_validation"].passed is True and gates["external_validation"].value == "specificity:pass; cohort:skipped"
    results = write_gates(root, "prog_wnt_camdr_nb_z", list(gates.values()))
    assert results["summary"] == {"passed": 9, "failed": 0, "not_evaluated": 0, "total": 9}
    assert os.path.exists(os.path.join(root, "RESULTS.json")) and os.path.exists(os.path.join(root, "reporting", "release_gates.csv"))

    # failing values
    root2 = str(tmp_path / "run2")
    _plant_run_dir(root2, full=0.12, shuffled=0.1, comp=0.2, ci="(-0.3, 0.5)", coverage=0.7, n_backbone=1, parity=0.5,
                   stage=[{"vectors": "predicted", "test": "permanova", "program": "all", "p_value": 0.4, "q_value": None, "note": ""},
                          {"vectors": "predicted", "test": "jt_trend", "program": "wnt_camdr", "p_value": 0.3, "q_value": 0.5, "note": ""},
                          {"vectors": "measured", "test": "jt_trend", "program": "wnt_camdr", "p_value": 0.001, "q_value": 0.001, "note": ""}],
                   validation={"specificity": {"status": "ran", "passed": False}})
    gates2 = {g.name: g for g in evaluate_gates(root2, "prog_wnt_camdr_nb_z")}
    assert all(gates2[k].passed is False for k in ("above_chance", "beyond_identity", "ci_excludes_zero", "shuffled_at_chance",
                                                     "calibrated", "backbone_stable", "inference_parity", "stage_association",
                                                     "external_validation")), {k: g.passed for k, g in gates2.items()}

    # absent artefacts are reported as not evaluated, never passed
    root3 = str(tmp_path / "run3")
    os.makedirs(root3)
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert all(g.passed == "not_evaluated" for g in gates3.values())
    assert all(g.status == "NOT EVALUATED" for g in gates3.values())
    # a parity file without the parity column is not evaluated either
    _plant_run_dir(root3, validation={"cohort": {"status": "skipped", "reason": "no manifest"}})
    os.makedirs(os.path.join(root3, "deployment"), exist_ok=True)
    pd.DataFrame({"slide": ["S"], "pearson": [0.3]}).to_csv(os.path.join(root3, "deployment", "sanity_check.csv"), index=False)
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["inference_parity"].passed == "not_evaluated"
    assert gates3["external_validation"].passed == "not_evaluated"
    assert gates3["stage_association"].passed == "not_evaluated"
    # a skipped stage test and a broken validator
    pd.DataFrame([{"vectors": "predicted", "test": "permanova", "program": "all", "p_value": 0.01, "q_value": None,
                   "note": "skipped: fewer than 2 groups"}]).to_csv(os.path.join(root3, "evaluation", "stage_association.csv"), index=False)
    _plant_run_dir(root3, validation={"specificity": {"status": "ran", "passed": True}, "robustness": {"status": "failed", "reason": "x"}})
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["stage_association"].passed == "not_evaluated"
    assert gates3["external_validation"].passed is False
    # a validator that ran without judging itself is not a pass
    _plant_run_dir(root3, validation={"specificity": {"status": "ran"}, "cohort": {"status": "skipped", "reason": "no manifest"}})
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["external_validation"].passed == "not_evaluated"
    assert "specificity:unjudged" in gates3["external_validation"].value
    # one validator that ran and failed its bar fails the gate even when another passed
    _plant_run_dir(root3, validation={"specificity": {"status": "ran", "passed": True}, "robustness": {"status": "ran", "passed": False}})
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["external_validation"].passed is False and "robustness:fail" in gates3["external_validation"].value
    # an inference error recorded by deployment fails the parity gate even without a number
    pd.DataFrame({"slide": ["S"], "pearson": [0.3], "parity_pearson": [float("nan")], "parity_error": ["RuntimeError: boom"]}).to_csv(
        os.path.join(root3, "deployment", "sanity_check.csv"), index=False)
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["inference_parity"].passed is False and "boom" in gates3["inference_parity"].meaning
    # an empty error string with a good number passes
    pd.DataFrame({"slide": ["S"], "pearson": [0.3], "parity_pearson": [0.99], "parity_error": [""]}).to_csv(
        os.path.join(root3, "deployment", "sanity_check.csv"), index=False)
    gates3 = {g.name: g for g in evaluate_gates(root3, "prog_wnt_camdr_nb_z")}
    assert gates3["inference_parity"].passed is True
