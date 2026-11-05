"""End-to-end: main.py on the synthetic fixture, resume, and forced recomputation.

Runs the real entry point as a subprocess three times:
1. a fresh run must produce the reporting artefacts and recover the planted program above chance;
2. a second identical run must load every component (run_manifest.json: recomputed false everywhere);
3. ``--force training`` must recompute training and everything after it, nothing before.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = os.path.join(ROOT, "main.py")

SMALL = [
    "--set", "ingest.demo_n_patients=4", "--set", "ingest.demo_cells_per=1200", "--set", "ingest.demo_img_size=512",
    # four folds on four patients: each fold trains on two patients after the inner validation patient is
    # removed; with two folds a single training patient remained and the planted signal was not recoverable
    "--set", "training.n_folds=4", "--set", "training.seeds=[7]",
    "--set", "training.model_variants=[full,shuffled,no_program_tokens]",
    "--set", "training.max_epochs=12", "--set", "training.batch_cells=256", "--set", "training.warmup_steps=5",
    "--set", "training.patience=4", "--set", "interpretation.srb_rounds=15", "--set", "programs.programs_max_iter=40",
]


def _run(workspace: str, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, MAIN, "--config", os.path.join(ROOT, "configs", "synthetic.yaml"),
           "--set", f"run.root={workspace}", "--set", f"run.local_data={workspace}/raw", *SMALL, *extra]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-4000:])
    assert proc.returncode == 0, "main.py failed"
    return proc


def _manifest(workspace: str) -> dict:
    with open(os.path.join(workspace, "runs", "synthetic_v3", "run_manifest.json")) as fh:
        return json.load(fh)["components"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = str(tmp_path_factory.mktemp("v3e2e"))
    _run(ws)
    return ws


def test_artefacts_and_planted_signal(workspace):
    run_dir = os.path.join(workspace, "runs", "synthetic_v3")
    for rel in ("reporting/CONTRIBUTIONS.md", "reporting/model_card.json", "reporting/release_gates.csv",
                "evaluation/cv_results.csv", "deployment/vmap_check.csv", "training/final_model.pt",
                "interpretation/gene_backbone.csv", "external_validation/validation_summary.json",
                "run_manifest.json", "RESULTS.json", "environment.json"):
        assert os.path.exists(os.path.join(run_dir, rel)), rel
    with open(os.path.join(run_dir, "RESULTS.json")) as fh:
        results = json.load(fh)
    for gate in ("above_chance", "beyond_identity", "ci_excludes_zero", "shuffled_at_chance", "calibrated",
                 "backbone_stable", "inference_parity", "stage_association", "external_validation"):
        assert gate in results["gates"], gate
        assert results["gates"][gate]["status"] in ("PASS", "FAIL", "NOT EVALUATED")
    assert results["summary"]["total"] == 9
    res = pd.read_csv(os.path.join(run_dir, "evaluation", "cv_results.csv"), index_col=0)
    t = "prog_wnt_camdr_nb_z"
    assert t in res.index
    best = max(res.loc[t, "full"], res.loc[t, "no_program_tokens"])
    # the planted program must be recovered above the shuffled-label bound by at least one model path;
    # on four synthetic patients the small transformer is capacity-limited, so the MLP path may be the one
    assert best > res.loc[t, "shuffled"] + 0.15, res.loc[t]
    assert pd.notna(res.loc[t, "full"])
    # item 18: the shuffled baseline permutes targets, expression embedding and lineage together within slide
    # and is scored against the permuted targets, so its per-patient median must sit at chance
    assert abs(res.loc[t, "shuffled"]) < 0.08, res.loc[t, "shuffled"]
    text = open(os.path.join(run_dir, "reporting", "CONTRIBUTIONS.md")).read()
    assert "Decision rules" in text and "ceiling" in text


def test_second_run_loads_everything(workspace):
    _run(workspace)
    recomputed = {k: v["recomputed"] for k, v in _manifest(workspace).items()}
    assert recomputed and not any(recomputed.values()), recomputed


def test_force_training_recomputes_downstream_only(workspace):
    _run(workspace, "--force", "training")
    recomputed = {k: v["recomputed"] for k, v in _manifest(workspace).items()}
    before = ["ingest", "preprocess", "program_inference", "feature_extraction", "feature_selection", "unified_model"]
    after = ["training", "evaluation", "interpretation", "external_validation", "deployment", "reporting"]
    assert not any(recomputed[k] for k in before), recomputed
    assert all(recomputed[k] for k in after), recomputed
