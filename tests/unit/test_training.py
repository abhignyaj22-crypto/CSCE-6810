"""Training: batcher shapes and padding, loop convergence, temperature scaling, CV plumbing, component skip rule."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from mmprog.config import Config  # noqa: E402
from mmprog.core.context import Context  # noqa: E402
from mmprog.training.data import ALL_VARIANTS, CellBatcher, variant_flags  # noqa: E402
from mmprog.training.loop import batch_loss, fit_temperature, predict, train_one  # noqa: E402
from mmprog.training import cv as cvmod  # noqa: E402


class FakeUnifiedModel(nn.Module):
    """Same forward signature as models.unified.UnifiedProgramModel; a linear map from the centre features."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.lin = nn.Linear(spec["d_centre"], spec["n_targets"])
        self.logv = nn.Linear(spec["d_centre"], spec["n_targets"])
        self.comp = nn.Linear(spec["d_centre"], spec["n_lin"])
        self.proj = nn.Linear(spec["d_centre"], int(spec.get("d_align", 3)))
        self.nb_mix = nn.Linear(spec["d_centre"], spec["n_targets"])
        self.temperature = 1.0

    def forward(self, centre, grid, nb, nb_offsets_um, sid, flags):
        y = self.lin(centre)
        if flags.get("use_neighbours", True) and nb is not None and nb.shape[1] > 0:
            y = y + 0.1 * self.nb_mix(nb.mean(1))
        return dict(mean=y, log_var=self.logv(centre).clamp(-6, 6), comp_logits=self.comp(centre),
                    morph_pooled=self.proj(centre), attn=None)


def fake_builder(spec, device):
    torch.manual_seed(0)
    return FakeUnifiedModel(spec).to(device)


def _planted(n_pat=4, cells=120, d=6, T=2, k=3, seed=0):
    rng = np.random.default_rng(seed)
    N = n_pat * cells
    X = rng.normal(size=(N, d)).astype(np.float32)
    W = rng.normal(size=(d, T)).astype(np.float32)
    Y = (X @ W + 0.3 * rng.normal(size=(N, T))).astype(np.float32)
    groups = np.repeat([f"P{i}" for i in range(n_pat)], cells)
    sid = np.repeat(np.arange(n_pat), cells).astype(np.int64)
    NB = np.zeros((N, k), dtype=np.int64)
    for s in range(n_pat):
        idx = np.where(sid == s)[0]
        for i in idx:
            NB[i] = rng.choice(idx[idx != i], k, replace=False)
    xy = rng.uniform(0, 500, size=(N, 2)).astype(np.float32)
    lin = rng.integers(0, 3, size=N)
    grid = rng.normal(size=(N, 4, 5)).astype(np.float16)
    return X, Y, groups, sid, NB, xy, lin, grid


def _cfg(**kw):
    base = dict(max_epochs=6, patience=3, lr=1e-2, weight_decay=0.0, warmup_steps=2, batch_cells=64,
                lambda_comp=0.1, lambda_smooth=0.05, lambda_align=0.05, uncertainty="gaussian", temperature_scaling=True,
                n_folds=2, seeds=(7,), model_variants=("full", "shuffled"), pca_dim=4, use_morpho=False,
                selection_enabled=False, amp_enabled=False, tf32=False, compile_model=False)
    base.update(kw)
    return Config(**base)


def test_batcher_pads_last_batch_and_masks_it():
    X, Y, groups, sid, NB, xy, lin, grid = _planted(cells=50)
    idx = np.arange(70)
    b = CellBatcher(X, grid, NB, None, xy, sid, Y, lin, X[:, :3], idx, batch_cells=32, name="t")
    batches = list(b)
    assert len(batches) == 3
    last = batches[-1]
    assert last["centre"].shape == (32, X.shape[1]) and last["grid"].shape == (32, 4, 5)
    assert int(last["pad_mask"].sum()) == 70 - 64
    assert last["nb"].shape == (32, 3, X.shape[1]) and last["nb_offsets_um"].shape == (32, 3, 2)
    assert set(np.concatenate([bt["idx"][bt["pad_mask"]].numpy() for bt in batches]).tolist()) == set(idx.tolist())


def test_batch_loss_has_all_parts_and_respects_flags():
    X, Y, groups, sid, NB, xy, lin, grid = _planted(cells=40)
    b = CellBatcher(X, grid, NB, None, xy, sid, Y, lin, X[:, :3], np.arange(40), batch_cells=40, name="t")
    batch = next(iter(b))
    model = FakeUnifiedModel(dict(d_centre=X.shape[1], n_targets=Y.shape[1], n_lin=3))
    out = model(batch["centre"], batch["grid"], batch["nb"], batch["nb_offsets_um"], batch["sid"], variant_flags("full"))
    loss, parts = batch_loss(out, batch, _cfg(), variant_flags("full"))
    # alignment is off in 'full' (D31); the 'align' variant switches it on with lambda_align_on
    assert {"nll", "comp", "smooth", "total"} <= set(parts) and "align" not in parts
    _, parts_al = batch_loss(out, batch, _cfg(lambda_align_on=0.1), variant_flags("align"))
    assert "align" in parts_al
    _, parts2 = batch_loss(out, batch, _cfg(), variant_flags("no_comp"))
    assert "comp" not in parts2
    _, parts3 = batch_loss(out, batch, _cfg(uncertainty="none"), variant_flags("no_smooth"))
    assert "mse" in parts3 and "smooth" not in parts3


def test_train_one_reduces_validation_loss_and_sets_temperature():
    X, Y, groups, sid, NB, xy, lin, grid = _planted()
    tr = np.where(np.isin(groups, ["P0", "P1", "P2"]))[0]
    va = np.where(groups == "P3")[0]
    mask = np.zeros(len(X), bool)
    mask[tr] = True
    common = dict(X_centre=X, GRID=grid, NB=NB, NB_DIST=None, XY_um=xy, SID=sid, Y=Y, LIN=lin, expression_embedding=X[:, :3],
                  batch_cells=64, train_mask=mask)
    b_tr = CellBatcher(cell_idx=tr, shuffle=True, name="tr", **common)
    b_va = CellBatcher(cell_idx=va, name="va", **common)
    model = fake_builder(dict(d_centre=X.shape[1], n_targets=Y.shape[1], n_lin=3), "cpu")
    cfg = _cfg(max_epochs=15, patience=15)
    model, best, epochs, hist = train_one(model, b_tr, b_va, cfg, "cpu", variant_flags("full"), seed=0, verbose=False)
    assert hist["val_mse"].iloc[-1] < hist["val_mse"].iloc[0]
    assert 0.25 <= model.temperature <= 4.0
    mean, std, comp, _ = predict(model, b_va, cfg, "cpu", variant_flags("full"))
    assert mean.shape == (len(va), Y.shape[1]) and np.isfinite(std).all() and comp.shape == (len(va), 3)
    r = np.corrcoef(mean[:, 0], Y[va, 0])[0, 1]
    assert r > 0.8


def test_temperature_scaling_leaves_mean_unchanged():
    X, Y, groups, sid, NB, xy, lin, grid = _planted(cells=60)
    b = CellBatcher(X, grid, NB, None, xy, sid, Y, lin, None, np.arange(60), batch_cells=60, name="t")
    model = fake_builder(dict(d_centre=X.shape[1], n_targets=Y.shape[1], n_lin=3), "cpu")
    cfg = _cfg()
    m0, s0, _, _ = predict(model, b, cfg, "cpu", variant_flags("full"))
    model.temperature = fit_temperature(model, b, cfg, "cpu", variant_flags("full"))
    m1, s1, _, _ = predict(model, b, cfg, "cpu", variant_flags("full"))
    assert np.allclose(m0, m1)
    assert np.allclose(s1, s0 * model.temperature)


def _context(tmp_path):
    X, Y, groups, sid, NB, xy, lin, grid = _planted(n_pat=4, cells=80, d=8)
    cfg = _cfg(root=str(tmp_path), run_name="t", local_data=str(tmp_path / "raw"), demo_mode=True, device="cpu",
               pca_dim=8, max_epochs=15, patience=15)
    ctx = Context.new(cfg)
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sample": [f"S{s}" for s in sid], "patient": groups,
                        "stage": np.repeat(["control", "SMM", "MM", "MM"], 80), "lineage_name": np.array(["plasma", "T", "B"])[lin]})
    targets = ["prog_wnt_camdr_nb_z", "anchor_wnt_camdr_nb_z"]
    ctx.replace(obs=obs, EMB=X, GRID=grid, MOR=np.zeros((len(X), 2), np.float32), NB=NB, NB_DIST=np.ones_like(NB, dtype=np.float32),
                XY=xy / 500, SID=sid, GROUPS=groups, Y=Y, LIN=np.eye(3)[lin].astype(np.float32), lineage=lin,
                lin_names=["plasma", "T", "B"], targets=targets, target_family={"prog_wnt_camdr_nb_z": "learned", "anchor_wnt_camdr_nb_z": "anchor_only"},
                model_spec={"d_model": 16}, expression_embedding=X[:, :3], n_samples=4)
    return ctx


def test_run_variant_shapes_and_all_variants(tmp_path):
    ctx = _context(tmp_path)
    for v in ALL_VARIANTS:
        preds, unc, comp, log, hist, tg = cvmod.run_variant(v, ctx, ctx.cfg, seed=0, builder=fake_builder)
        # full width (N, T) with NaN in the columns the variant does not predict
        assert preds.shape == (len(ctx.obs), 2) and len(log) == 2
        if v == "anchor_only_targets":
            assert tg == ["anchor_wnt_camdr_nb_z"] and np.isfinite(preds[:, 1]).all() and np.isnan(preds[:, 0]).all()
        else:
            assert tg == ["prog_wnt_camdr_nb_z"] and np.isfinite(preds[:, 0]).all() and np.isnan(preds[:, 1]).all()
        if v != "no_comp":
            assert comp is not None and comp.shape == (len(ctx.obs), 3)
    p_full, *_ = cvmod.run_variant("full", ctx, ctx.cfg, seed=0, builder=fake_builder)
    p_shuf, *_ = cvmod.run_variant("shuffled", ctx, ctx.cfg, seed=0, builder=fake_builder)
    r_full = np.corrcoef(p_full[:, 0], ctx.Y[:, 0])[0, 1]
    r_shuf = np.corrcoef(p_shuf[:, 0], ctx.Y[:, 0])[0, 1]
    assert r_full > r_shuf + 0.3


def test_training_component_writes_artefacts_and_skips(tmp_path, monkeypatch):
    from mmprog.components.training import Training
    ctx = _context(tmp_path)
    monkeypatch.setattr(cvmod, "_model_builder", lambda: fake_builder)
    comp = Training()
    calls = {"n": 0}
    orig = comp._run

    def counted(c):
        calls["n"] += 1
        orig(c)

    comp._run = counted
    ctx = comp.run(ctx)
    assert calls["n"] == 1
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    assert set(ctx.cv_preds) == {"full", "shuffled"} and ctx.hardware["device"] == "cpu"
    assert os.path.exists(os.path.join(comp.dir(ctx), "multiseed", "seed7.npy"))
    ctx2 = Context.new(ctx.cfg)
    for f in ("obs", "EMB", "GRID", "MOR", "NB", "NB_DIST", "XY", "SID", "GROUPS", "Y", "LIN", "lineage", "lin_names",
              "targets", "target_family", "model_spec", "expression_embedding", "n_samples"):
        setattr(ctx2, f, getattr(ctx, f))
    ctx2 = comp.run(ctx2)
    assert calls["n"] == 1 and set(ctx2.cv_preds) == {"full", "shuffled"} and ctx2.folds[0].fold_id == 0
    model, ckpt = cvmod.load_final(ctx.final_model_path, "cpu", builder=fake_builder)
    assert ckpt["targets"] == ["prog_wnt_camdr_nb_z"] and ckpt["all_targets"] == ctx.targets
    assert model.temperature == ckpt["temperature"]


@pytest.mark.skipif(__import__("importlib").util.find_spec("mmprog.models.unified") is None, reason="unified model not available yet")
def test_real_model_forward_matches_contract(tmp_path):
    from mmprog.components.unified_model import build_model
    ctx = _context(tmp_path)
    ctx.model_spec = dict(d_model=16, n_layers=1, n_heads=2, head_dim=8, dropout=0.0, n_fourier=4, fourier_scale=1.0,
                          slide_emb_dim=4, grid=2, n_programs=2, n_scales=1, n_genes=5, uncertainty="gaussian",
                          program_init_weights=np.random.default_rng(0).random((2, 5)).tolist(), mlp_hidden=16)
    spec = cvmod._spec_for(ctx, 8, 2, variant_flags("full"))
    model = build_model(spec, "cpu")
    b = CellBatcher(ctx.EMB, ctx.GRID, ctx.NB, None, ctx.obs[["x", "y"]].to_numpy(np.float32), ctx.SID, ctx.Y, ctx.lineage,
                    ctx.expression_embedding, np.arange(16), batch_cells=16, name="t")
    batch = next(iter(b))
    out = model(batch["centre"], batch["grid"], batch["nb"], batch["nb_offsets_um"], batch["sid"], variant_flags("full"))
    assert out["mean"].shape == (16, 2)


# ----------------------------------------------------------------------------- review-driven tests
def test_shuffled_permutes_targets_expression_and_lineage_together(tmp_path):
    """Chance bound: Y, expression embedding and lineage are permuted with ONE within-slide permutation, the
    batcher's neighbour predictions come from the cache (never from targets), and the permuted targets are
    returned so Evaluation can score against them."""
    ctx = _context(tmp_path)
    seen = {}

    def spy_builder(spec, device):
        return fake_builder(spec, device)

    orig = cvmod.CellBatcher

    class SpyBatcher(orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.setdefault("batchers", []).append(self)

    cvmod.CellBatcher = SpyBatcher
    try:
        preds, *_ = cvmod.run_variant("shuffled", ctx, ctx.cfg, seed=3, builder=spy_builder)
    finally:
        cvmod.CellBatcher = orig
    Y_used = cvmod.run_variant.last_targets_used
    assert Y_used.shape == ctx.Y.shape and np.isfinite(Y_used[:, 0]).all()
    # permuted within slide: same multiset per slide, different order overall
    for s in np.unique(ctx.SID):
        m = ctx.SID == s
        assert np.allclose(np.sort(Y_used[m, 0]), np.sort(ctx.Y[m, 0]))
    assert not np.allclose(Y_used[:, 0], ctx.Y[:, 0])
    b = seen["batchers"][0]
    # the batcher was built from the permuted targets, lineage and expression, with the same permutation
    rows = np.where(ctx.SID == ctx.SID[0])[0]
    perm_pos = np.array([np.where(np.isclose(ctx.Y[rows, 0], y))[0][0] for y in b.Y[rows, 0]])
    assert np.array_equal(b.LIN[rows], np.asarray(ctx.lineage)[rows][perm_pos])
    assert np.allclose(b.EXPR[rows], np.asarray(ctx.expression_embedding)[rows][perm_pos])
    # neighbour predictions come from the cache, not from the target array
    batch = next(iter(orig(cell_idx=np.arange(len(ctx.obs)), X_centre=ctx.EMB, GRID=ctx.GRID, NB=ctx.NB, NB_DIST=None,
                           XY_um=ctx.obs[["x", "y"]].to_numpy(np.float32), SID=ctx.SID, Y=b.Y, LIN=b.LIN,
                           expression_embedding=b.EXPR, batch_cells=len(ctx.obs), name="chk")))
    assert "nb_y" not in batch and "nb_pred" in batch


def test_prediction_cache_feeds_the_smoothness_term():
    X, Y, groups, sid, NB, xy, lin, grid = _planted(cells=40)
    b = CellBatcher(X, grid, NB, None, xy, sid, Y, lin, X[:, :3], np.arange(40), batch_cells=40, name="t")
    batch = next(iter(b))
    # before any update the cache holds the warm start (targets); after an update it holds the predictions
    rows = batch["idx"].numpy()
    assert np.allclose(batch["nb_pred"].numpy(), Y[NB[rows]].astype(np.float32))
    b.update_cache(np.arange(len(X)), np.full_like(Y, 7.0))
    batch2 = next(iter(b))
    assert np.allclose(batch2["nb_pred"].numpy(), 7.0) and bool(b.cache_visited.all())


def test_real_builder_cv_round_trip_with_permuted_target_index(tmp_path):
    """Signal planted in ONE ctx.Y column must land in that column of the out-of-fold predictions even when
    the head order is a non-identity permutation of the context order; anchor_only_targets fills only anchor
    columns and slices the program tokens."""
    from mmprog.components.unified_model import build_model
    rng = np.random.default_rng(0)
    X, Y0, groups, sid, NB, xy, lin, grid = _planted(n_pat=4, cells=120, d=8, T=1)
    # context targets in scale-major order (cell then nb) for two programs plus two anchor columns
    targets = ["prog_a_cell_z", "prog_b_cell_z", "prog_a_nb_z", "prog_b_nb_z", "anchor_a_cell_z", "anchor_a_nb_z"]
    noise = rng.normal(size=(len(X), 6)).astype(np.float32)
    Y = noise.copy()
    Y[:, 3] = Y0[:, 0]                      # planted signal only in prog_b_nb_z (ctx column 3)
    Y[:, 5] = Y0[:, 0]                      # and in anchor_a_nb_z
    cfg = _cfg(root=str(tmp_path), run_name="t", local_data=str(tmp_path / "raw"), demo_mode=True, device="cpu",
               pca_dim=8, max_epochs=40, patience=40, lr=1e-2, warmup_steps=2, batch_cells=120, lambda_smooth=0.0)
    ctx = Context.new(cfg)
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sample": [f"S{s}" for s in sid], "patient": groups,
                        "stage": np.repeat(["control", "SMM", "MM", "MM"], 120), "lineage_name": np.array(["plasma", "T", "B"])[lin]})
    # head order is program-major: a_cell, a_nb, b_cell, b_nb -> ctx columns [0, 2, 1, 3]
    spec = dict(d_model=16, n_layers=1, n_heads=2, head_dim=8, dropout=0.0, n_fourier=4, fourier_scale=1.0, slide_emb_dim=4,
                grid=2, programs=["a", "b"], scales=["cell", "nb"], n_programs=2, n_scales=2, n_genes=5,
                target_index=[0, 2, 1, 3], uncertainty="gaussian", mlp_hidden=16,
                program_init_weights=rng.random((2, 5)).tolist())
    ctx.replace(obs=obs, EMB=X, GRID=grid, MOR=np.zeros((len(X), 2), np.float32), NB=NB, NB_DIST=np.ones_like(NB, dtype=np.float32),
                XY=xy / 500, SID=sid, GROUPS=groups, Y=Y, LIN=np.eye(3)[lin].astype(np.float32), lineage=lin,
                lin_names=["plasma", "T", "B"], targets=targets,
                target_family={t: ("anchor_only" if t.startswith("anchor_") else "learned") for t in targets},
                model_spec=spec, expression_embedding=X[:, :3], n_samples=4)
    preds, unc, comp, log, hist, tg = cvmod.run_variant("full", ctx, ctx.cfg, seed=0, builder=build_model)
    assert tg == ["prog_a_cell_z", "prog_a_nb_z", "prog_b_cell_z", "prog_b_nb_z"]
    assert np.isnan(preds[:, 4]).all() and np.isnan(preds[:, 5]).all()
    r = {j: np.corrcoef(preds[:, j], Y[:, j])[0, 1] for j in range(4)}
    assert r[3] > 0.5, r                      # the planted column is recovered in its own column
    assert r[3] > max(r[0], r[1], r[2]) + 0.3
    # anchor_only_targets: one program (a) with two scales; tokens sliced; only anchor columns filled
    p_a, _, _, _, _, tg_a = cvmod.run_variant("anchor_only_targets", ctx, ctx.cfg, seed=0, builder=build_model)
    assert tg_a == ["anchor_a_cell_z", "anchor_a_nb_z"]
    assert np.isfinite(p_a[:, 4]).all() and np.isfinite(p_a[:, 5]).all() and np.isnan(p_a[:, :4]).all()
    assert np.corrcoef(p_a[:, 5], Y[:, 5])[0, 1] > 0.4


def test_run_variant_is_reproducible_per_seed(tmp_path):
    from mmprog.components.unified_model import build_model
    ctx = _context(tmp_path)
    ctx.model_spec = dict(d_model=16, n_layers=1, n_heads=2, head_dim=8, dropout=0.0, n_fourier=4, fourier_scale=1.0,
                          slide_emb_dim=4, grid=2, n_programs=1, n_scales=1, n_genes=5, uncertainty="gaussian",
                          program_init_weights=np.random.default_rng(0).random((1, 5)).tolist(), mlp_hidden=16)
    ctx.cfg.max_epochs = 3
    a, *_ = cvmod.run_variant("full", ctx, ctx.cfg, seed=7, builder=build_model)
    b, *_ = cvmod.run_variant("full", ctx, ctx.cfg, seed=7, builder=build_model)
    c, *_ = cvmod.run_variant("full", ctx, ctx.cfg, seed=8, builder=build_model)
    assert np.array_equal(a[:, 0], b[:, 0]), "same seed must give bit-identical out-of-fold predictions"
    assert not np.allclose(a[:, 0], c[:, 0]), "a different seed must change the predictions"


def test_make_folds_refuses_too_few_patients():
    groups = np.repeat(["P0", "P1", "P2"], 10)
    with pytest.raises(ValueError, match="no training patient"):
        cvmod.make_folds(groups, n_folds=2)
