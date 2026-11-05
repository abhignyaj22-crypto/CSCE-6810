"""ExternalValidation component and its validators on a hand-built context."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_cfg  # noqa: E402
from mmprog.core.context import Context  # noqa: E402

GRID = 4
K = 3
LIN = ["plasma", "T", "B"]


class FakeUnifiedModel(torch.nn.Module):
    """Mean depends linearly on the centre features; dims are inferred from the first batch."""
    accepts_batch_dict = True

    def __init__(self, n_targets, seed=0):
        super().__init__()
        self.seed, self.n_targets = seed, n_targets
        self.W = None
        self.C = None
        self.temperature = 1.0

    def forward(self, batch):
        centre = batch["centre"].float()
        if self.W is None:
            g = torch.Generator().manual_seed(self.seed)
            self.W = torch.nn.Parameter(torch.randn(centre.shape[1], self.n_targets, generator=g))
            self.C = torch.nn.Parameter(torch.randn(centre.shape[1], len(LIN), generator=g))
        mean = centre @ self.W
        n_tok = 1 + GRID * GRID + batch["nb"].shape[1]
        return dict(mean=mean, log_var=torch.zeros_like(mean) - 1.0, comp_logits=centre @ self.C, morph_pooled=centre,
                    attn=torch.full((centre.shape[0], 2, n_tok), 1.0 / n_tok))


def _build_ctx(tmp_path, planted: bool = True, with_checkpoint: bool = False, seed: int = 5) -> Context:
    from mmprog.data.images import ImageStore
    from mmprog.data.synthetic import make_demo
    from mmprog.models.neighbours import knn_table
    from mmprog.preprocessing.qc import apply_cell_qc, calculate_qc_metrics, normalize_log1p
    from mmprog.programs.targets_builder import neighbourhood_mean
    from mmprog.resources import load_lineages, load_programs
    from mmprog.targets.programs import standardise
    from mmprog.validation.program_specificity import weighted_mean_scores
    from scipy.sparse import csr_matrix

    cfg = small_cfg(tmp_path, validation_n_random_sets=40, validation_perturbations="[gamma_1.0,gamma_0.8,rot90]")
    programs, lineages = load_programs(), load_lineages()
    adata, images, manifest = make_demo(programs, lineages, cfg, n_patients=3, cells_per=300, img_size=256, seed=seed)
    calculate_qc_metrics(adata)
    adata = apply_cell_qc(adata, cfg)
    adata = normalize_log1p(adata)
    ctx = Context.new(cfg)
    obs = adata.obs.reset_index(drop=True)
    var = pd.Index(adata.var_names)
    names = ["wnt_camdr", "cytotoxicity"]
    H = pd.DataFrame(0.0, index=names, columns=var)
    for p in names:
        for g in programs[p]:
            if g in var:
                H.loc[p, g] = 1.0
    targets = ["prog_wnt_camdr_cell_z", "prog_wnt_camdr_nb_z", "prog_cytotoxicity_nb_z"]
    X = csr_matrix(adata.X)
    S = weighted_mean_scores(X, csr_matrix(H.to_numpy(dtype=np.float32).T))
    um = {str(r["sample"]): cfg.um_per_px for r in manifest}
    S_nb = neighbourhood_mean(S, obs, cfg, um)
    groups = obs["patient"].astype(str).to_numpy()
    Y = np.column_stack([standardise(S[:, 0], groups, "cohort"), standardise(S_nb[:, 0], groups, "cohort"),
                         standardise(S_nb[:, 1], groups, "cohort")]).astype(np.float32)
    rng = np.random.default_rng(0)
    preds = (Y + 0.7 * rng.normal(size=Y.shape)) if planted else rng.normal(size=Y.shape)
    lin_true = obs["lineage_true"].astype(str).to_numpy()
    lineage = np.array([LIN.index(x) if x in LIN else 1 for x in lin_true], dtype=np.int64)
    xy_by_sample = {s: obs.loc[obs["sample"] == s, ["x", "y"]].to_numpy(np.float32) for s in obs["sample"].unique()}
    ctx.set("manifest", manifest)
    ctx.set("images", ImageStore(cfg, manifest, xy_by_sample, demo_images=images))
    ctx.set("adata", adata)
    ctx.set("obs", obs)
    ctx.set("gene_mask", np.ones(adata.n_vars, dtype=bool))
    ctx.set("programs", programs)
    ctx.set("program_weights", H)
    ctx.set("targets", targets)
    ctx.set("Y", Y)
    ctx.set("GROUPS", groups)
    ctx.set("NB", knn_table(obs, K))
    ctx.set("lineage", lineage)
    ctx.set("lin_names", LIN)
    ctx.set("cv_preds", {"full": preds.astype(np.float32)})
    if with_checkpoint:
        ck = dict(model=FakeUnifiedModel(len(targets)), spec=dict(grid=GRID, k=K, n_samples=3), targets=targets, lin_names=LIN,
                  temperature=1.0, fs=None, selector=None, flags={})
        path = str(tmp_path / "final_model.pt")
        torch.save(ck, path)
        ctx.set("final_model_path", path)
    ctx.component_manifests["training"] = {"hash": "t"}
    return ctx


def test_perturbation_parsing_covers_every_form_and_rejects_unknown():
    from mmprog.validation.perturbations import apply_perturbation, parse_perturbation

    assert parse_perturbation("gamma_0.8") == ("gamma", 0.8)
    assert parse_perturbation("blur_1.5") == ("blur", 1.5)
    assert parse_perturbation("rot90") == ("rot90", 90.0)
    assert parse_perturbation("noise_0.02") == ("noise", 0.02)
    assert parse_perturbation("scale_0.9") == ("scale", 0.9)
    for bad in ("flip", "gamma", "gamma_x", "rot45", "gamma_-1"):
        with pytest.raises(ValueError, match="perturbation"):
            parse_perturbation(bad)
    img = (np.arange(20 * 30).reshape(20, 30) % 251).astype(np.uint8)
    xy = np.array([[29.0, 0.0], [0.0, 19.0]], dtype=np.float32)
    rot, xy2 = apply_perturbation(img, xy, "rot90")
    assert rot.shape == (30, 20)
    # a pixel's value travels with its rotated coordinate
    for (x, y), (x2, y2) in zip(xy, xy2):
        assert rot[int(y2), int(x2)] == img[int(y), int(x)]
    same, _ = apply_perturbation(img, xy, "gamma_1.0")
    assert np.array_equal(same, img)


def test_program_specificity_planted_target_beats_matched_null(tmp_path):
    from mmprog.components.external_validation import ExternalValidation

    ctx = _build_ctx(tmp_path, planted=True)
    comp = ExternalValidation()
    comp.strict = True
    ctx = comp.run(ctx)
    assert os.path.exists(comp.path(ctx, "validation_summary.json"))
    v = ctx.validation["program_specificity"]
    assert v["status"] == "ran", v
    assert v["median_z"] > 2, v
    assert v["median_r_half_a"] > 0 and v["median_r_half_b"] > 0, v
    tab = ctx.validation_tables["program_specificity"]
    assert {"target", "patient", "z", "p_empirical", "r_half_a", "r_half_b", "r_target", "r_proxy"} <= set(tab.columns)
    sf = ctx.validation["spatial_fidelity"]
    assert sf["status"] == "ran" and np.isfinite(sf["median_moran_measured"])
    assert ctx.validation["perturbation_robustness"]["status"] == "skipped"       # no checkpoint
    assert ctx.validation["independent_cohort"]["status"] == "skipped"
    assert ctx.validation["protein_bridge"]["status"] == "skipped" and "codex_table" in ctx.validation["protein_bridge"]["reason"]
    assert ctx.codex["status"] == "skipped"
    # resume: a second run loads
    calls = {"n": 0}
    orig = comp._run
    comp._run = lambda c: (calls.__setitem__("n", calls["n"] + 1), orig(c))
    comp.run(ctx)
    assert calls["n"] == 0 and "program_specificity" in ctx.validation_tables


def test_program_specificity_random_prediction_sits_in_the_null(tmp_path):
    from mmprog.components.external_validation import ExternalValidation

    ctx = _build_ctx(tmp_path, planted=False)
    comp = ExternalValidation()
    comp.strict = True
    ctx = comp.run(ctx)
    v = ctx.validation["program_specificity"]
    assert v["status"] == "ran"
    assert abs(v["median_z"]) < 2, v


def test_failing_validator_is_recorded_and_others_still_run(tmp_path, monkeypatch):
    from mmprog.components.external_validation import ExternalValidation

    ctx = _build_ctx(tmp_path, planted=True)
    comp = ExternalValidation()
    monkeypatch.setattr(comp, "_spatial_fidelity", lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    ctx = comp.run(ctx)
    assert ctx.validation["spatial_fidelity"]["status"] == "failed" and "boom" in ctx.validation["spatial_fidelity"]["reason"]
    assert ctx.validation["program_specificity"]["status"] == "ran"
    assert ctx.validation["_overall"]["failed"] == 1
    with open(comp.path(ctx, "validation_summary.json")) as fh:
        assert json.load(fh)["spatial_fidelity"]["status"] == "failed"


def test_robustness_and_demo_cohort_run_with_a_checkpoint(tmp_path):
    from mmprog.components.external_validation import ExternalValidation

    ctx = _build_ctx(tmp_path, planted=True, with_checkpoint=True)
    manifest = tmp_path / "cohort.json"
    manifest.write_text(json.dumps({"demo": {"n_patients": 2, "cells_per": 200, "img_size": 256, "seed": 11}}))
    ctx.cfg.validation_cohort_manifest = str(manifest)
    comp = ExternalValidation()
    comp.strict = True
    ctx = comp.run(ctx)
    rob = ctx.validation["perturbation_robustness"]
    assert rob["status"] == "ran", rob
    # every validator that ran judges itself against its stated bar; the release gate reads the verdict
    for name, v in ctx.validation.items():
        if isinstance(v, dict) and v.get("status") == "ran":
            assert v.get("passed") in (True, False), (name, v)
            assert v.get("pass_bar"), name
    assert rob["passed"] == (rob["min_spearman_main"] > 0.9)
    tab = ctx.validation_tables["perturbation_robustness"]
    assert {"perturbation", "target", "spearman_vs_clean", "mean_abs_shift"} <= set(tab.columns)
    ident = tab[tab["perturbation"] == "gamma_1.0"]
    assert len(ident) and np.allclose(ident["spearman_vs_clean"], 1.0)
    assert set(tab["perturbation"]) == {"gamma_1.0", "gamma_0.8", "rot90"}
    coh = ctx.validation["independent_cohort"]
    assert coh["status"] in ("ran", "skipped"), coh
    if coh["status"] == "ran":
        t = ctx.validation_tables["independent_cohort"]
        assert {"target", "patient", "pearson", "pearson_shuffled", "measured_kind"} <= set(t.columns)
        assert "above_chance" in coh


def test_protein_bridge_runs_on_the_synthetic_codex_fixture(tmp_path):
    from mmprog.components.external_validation import ExternalValidation
    from mmprog.data.synthetic_codex import write_demo_codex

    ctx = _build_ctx(tmp_path, planted=True, with_checkpoint=True)
    ctx.cfg.validation_perturbations = ()
    s0 = ctx.obs["sample"].iloc[0]
    obs_s = ctx.obs[ctx.obs["sample"] == s0]
    csv_path, tif_path = write_demo_codex(str(tmp_path / "codex"), obs_s, ctx.images.get(s0),
                                          obs_s["lineage_true"].astype(str).to_numpy(), ctx.cfg)
    ctx.cfg.codex_table, ctx.cfg.codex_dapi = csv_path, tif_path
    ctx.cfg.codex_dapi_um_per_px = ctx.cfg.um_per_px
    comp = ExternalValidation()
    comp.strict = True
    ctx = comp.run(ctx)
    pb = ctx.validation["protein_bridge"]
    assert pb["status"] in ("ran", "insufficient_matches"), pb
    assert ctx.codex["status"] == pb["status"]
    if pb["status"] == "ran":
        assert "niche_contrast" in pb and pb["n_matched"] >= 10
        assert os.path.exists(comp.path(ctx, "protein_bridge.json"))
        assert "protein_bridge" in ctx.validation_tables


def test_moran_neighbour_remap_keeps_indices_inside_the_kept_subset():
    """Rows dropped because a neighbour lies outside the patient must not leave stale indices behind."""
    from mmprog.validation.spatial_fidelity import morans_i, remap_neighbours

    rng = np.random.default_rng(0)
    n, k = 12, 3
    nb = np.array([[(i + d) % n for d in (1, 2, 3)] for i in range(n)])
    keep = np.ones(n, dtype=bool)
    keep[[4, 9]] = False
    out = remap_neighbours(nb, keep)
    assert out.shape == (10, k) and out.min() >= 0 and out.max() < 10
    # a kept cell whose neighbour was dropped points at itself
    kept_rows = np.where(keep)[0]
    for r, orig in enumerate(kept_rows):
        for c in range(k):
            if not keep[nb[orig, c]]:
                assert out[r, c] == r
            else:
                assert kept_rows[out[r, c]] == nb[orig, c]
    x = rng.normal(size=n)
    val = morans_i(x[keep], out)
    assert np.isfinite(val)
    # with nothing dropped the remap is the identity
    assert np.array_equal(remap_neighbours(nb, np.ones(n, dtype=bool)), nb)


def test_random_sets_exclude_every_program_gene():
    from scipy.sparse import csr_matrix  # noqa: F401
    from mmprog.validation.program_specificity import matched_random_sets

    rng = np.random.default_rng(1)
    G = 60
    bins = np.repeat(np.arange(6), 10)
    usable = np.ones(G, dtype=bool)
    w_full = np.zeros(G)
    w_full[[0, 1, 2, 11, 12, 21]] = [1.0, 0.9, 0.8, 0.5, 0.4, 0.3]
    w_full[[3, 13, 22]] = 0.02                        # long tail: above the 1 % presence threshold, outside the core
    w_full[w_full == 0] = 1e-6                        # the NMF floor: numerically positive on every gene
    core = np.array([0, 1, 2, 11, 12, 21])
    W = matched_random_sets(core, w_full[core], bins, usable, 30, rng, exclude=w_full > 0.01 * w_full.max())
    assert W.nnz > 0                                  # the floor genes stay available as spare genes
    used = set(W.nonzero()[0].tolist())
    assert not (used & set(np.where(w_full > 0.01 * w_full.max())[0].tolist()))   # no material program gene, tail included
    # without the mask the long-tail genes can be drawn (the behaviour the fix removes from the pipeline)
    W2 = matched_random_sets(core, w_full[core], bins, usable, 200, np.random.default_rng(1))
    assert set(W2.nonzero()[0].tolist()) & {3, 13, 22}
