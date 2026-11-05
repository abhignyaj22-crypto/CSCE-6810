"""Deployment component and the v3 morphology-only inference path."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from mmprog.config import load_config  # noqa: E402
from mmprog.core.context import Context  # noqa: E402

GRID = 4
K = 3
T = 4
L = 3


class FakeUnifiedModel(torch.nn.Module):
    """Mean depends linearly on the centre features; dims are inferred lazily from the first batch."""
    accepts_batch_dict = True

    def __init__(self, d_in=None, seed=0):
        super().__init__()
        self.seed = seed
        self.W = None if d_in is None else torch.nn.Parameter(torch.randn(d_in, T, generator=torch.Generator().manual_seed(seed)))
        self.C = None if d_in is None else torch.nn.Parameter(torch.randn(d_in, L, generator=torch.Generator().manual_seed(seed + 1)))
        self.temperature = 1.5

    def forward(self, batch):
        centre = batch["centre"].float()
        if self.W is None:
            g = torch.Generator().manual_seed(self.seed)
            self.W = torch.nn.Parameter(torch.randn(centre.shape[1], T, generator=g))
            self.C = torch.nn.Parameter(torch.randn(centre.shape[1], L, generator=g))
        mean = centre @ self.W
        n_tok = 1 + GRID * GRID + batch["nb"].shape[1]
        return dict(mean=mean, log_var=torch.zeros_like(mean) - 1.0, comp_logits=centre @ self.C, morph_pooled=centre,
                    attn=torch.full((centre.shape[0], 2, n_tok), 1.0 / n_tok))


def _synthetic_image(seed=0, size=256, n_nuclei=60):
    rng = np.random.default_rng(seed)
    img = rng.normal(150, 8, size=(size, size)).astype(np.float32)
    pts = rng.uniform(20, size - 20, size=(n_nuclei, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy in pts:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 16] -= 90
    return np.clip(img, 0, 255).astype(np.uint8), pts


TARGETS = ["prog_wnt_camdr_cell_z", "prog_wnt_camdr_nb_z", "prog_cytotoxicity_cell_z", "prog_cytotoxicity_nb_z"]
LIN = ["plasma", "T", "B"]


def _cfg(tmp_path, **extra):
    over = [f"root={tmp_path}", f"local_data={tmp_path}/raw", "run_name=deploy", "demo_mode=true", "encoder=tiny_test",
            "patch_px=32", "grid=4", "knn_k=3", "num_workers=0"] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


def test_virtual_program_map_v3_columns_with_tiny_encoder(tmp_path):
    from mmprog.inference.virtual_map_v3 import virtual_program_map_v3

    cfg = _cfg(tmp_path)
    img, _ = _synthetic_image()
    ck = dict(model=FakeUnifiedModel(), spec=dict(grid=GRID, k=K, n_samples=2), targets=TARGETS, lin_names=LIN,
              temperature=1.5, fs=None, selector=None, flags={})
    vmap = virtual_program_map_v3(img, ck, cfg, "cpu", batch_size=16)
    assert len(vmap) > 10
    for t in TARGETS:
        assert t.replace("_z", "_pred") in vmap and t.replace("_z", "_std") in vmap
    for name in LIN:
        assert f"lineage_prob_{name}" in vmap
    probs = vmap[[f"lineage_prob_{n}" for n in LIN]].to_numpy().sum(1)
    assert np.allclose(probs, 1.0, atol=1e-4)
    # std = exp(0.5 * log_var) * temperature with log_var = -1
    assert np.allclose(vmap["prog_wnt_camdr_cell_std"], np.exp(-0.5) * 1.5, atol=1e-4)


def test_padded_last_batch_matches_unpadded(tmp_path):
    from mmprog.inference.virtual_map_v3 import predict_unified

    rng = np.random.default_rng(0)
    arrays = dict(centre=rng.normal(size=(37, 5)).astype(np.float32), grid=rng.normal(size=(37, GRID * GRID, 2)).astype(np.float32),
                  nb=rng.normal(size=(37, K, 5)).astype(np.float32), nb_offset=rng.normal(size=(37, K, 2)).astype(np.float32),
                  nb_mask=np.ones((37, K), dtype=bool), sid=np.zeros(37, dtype=np.int64), morpho=rng.normal(size=(37, 3)).astype(np.float32))
    model = FakeUnifiedModel(d_in=5)
    a = predict_unified(model, arrays, "cpu", temperature=1.0, batch_size=16)["mean"]
    b = predict_unified(model, arrays, "cpu", temperature=1.0, batch_size=37)["mean"]
    assert a.shape == (37, T) and np.allclose(a, b, atol=1e-6)


@pytest.fixture
def deploy_ctx(tmp_path):
    from mmprog.data.images import ImageStore
    from mmprog.data.synthetic import make_demo
    from mmprog.resources import load_lineages, load_programs

    cfg = _cfg(tmp_path)
    adata, images, manifest = make_demo(load_programs(), load_lineages(), cfg, n_patients=2, cells_per=80, img_size=256, seed=5)
    ctx = Context.new(cfg)
    obs = adata.obs.reset_index(drop=True)
    xy_by_sample = {s: obs.loc[obs["sample"] == s, ["x", "y"]].to_numpy(np.float32) for s in obs["sample"].unique()}
    ctx.set("images", ImageStore(cfg, manifest, xy_by_sample, demo_images=images))
    ctx.set("obs", obs)
    rng = np.random.default_rng(1)
    ctx.set("Y", rng.normal(size=(len(obs), len(TARGETS))).astype(np.float32))
    ctx.set("targets", TARGETS)
    ctx.set("lin_names", LIN)
    ck = dict(model=FakeUnifiedModel(), spec=dict(grid=GRID, k=K, n_samples=2), targets=TARGETS, lin_names=LIN,
              temperature=1.0, fs=None, selector=None, flags={})
    path = str(tmp_path / "final_model.pt")
    torch.save(ck, path)
    ctx.set("final_model_path", path)
    ctx.component_manifests["training"] = {"hash": "t"}
    return ctx


def test_deployment_component_writes_artefacts(deploy_ctx):
    from mmprog.components.deployment import Deployment

    comp = Deployment()
    ctx = comp.run(deploy_ctx)
    for name in comp.provides:
        assert getattr(ctx, name) is not None, name
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    assert "codex" not in comp.provides          # the protein bridge now lives in external_validation
    assert len(ctx.exports) == 2 and all(os.path.exists(p) for p in ctx.exports)
    sanity = pd.read_csv(comp.path(ctx, "sanity_check.csv"))
    assert sanity["n_detected"].iloc[0] > 0 and "pearson" in sanity
    # the fake checkpoint has no feature space, so the parity number is recorded as NaN, never as a pass
    assert "parity_pearson" in sanity and not np.isfinite(sanity["parity_pearson"].iloc[0])
    qupath = pd.read_csv(ctx.exports[0], sep="\t")
    assert "x_um" in qupath and any(c.startswith("Measurement:") for c in qupath.columns)
    # resume: second run loads
    calls = {"n": 0}
    orig = comp._run
    comp._run = lambda c: (calls.__setitem__("n", calls["n"] + 1), orig(c))
    comp.run(ctx)
    assert calls["n"] == 0


def test_inference_uses_the_slide_pixel_size_for_neighbour_offsets(tmp_path):
    """Training scales neighbour offsets by each slide's OME pixel size; the inference path must do the same.
    With um_per_px doubled, the offsets handed to the model are exactly twice as large."""
    from mmprog.inference.virtual_map_v3 import build_inference_batch_arrays

    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(3)
    obs = pd.DataFrame({"x": rng.uniform(0, 200, 40), "y": rng.uniform(0, 200, 40), "sample": "S"})
    feats = dict(EMB=rng.normal(size=(40, 5)).astype(np.float32), GRID=rng.normal(size=(40, GRID * GRID, 2)).astype(np.float32),
                 MOR=rng.normal(size=(40, 3)).astype(np.float32))
    spec = dict(grid=GRID, k=K, n_samples=2)
    base = build_inference_batch_arrays(obs, feats, cfg, spec, um_per_px=cfg.um_per_px)
    doubled = build_inference_batch_arrays(obs, feats, cfg, spec, um_per_px=2 * cfg.um_per_px)
    default = build_inference_batch_arrays(obs, feats, cfg, spec)
    assert np.allclose(doubled["nb_offset"], 2 * base["nb_offset"])
    assert np.allclose(default["nb_offset"], base["nb_offset"])      # the configured value is the fallback


def test_parity_error_is_recorded_and_not_hidden(deploy_ctx, monkeypatch):
    """A crash inside the inference path on the parity check lands in sanity_check.csv as parity_error (the
    inference_parity gate fails on it) instead of a silent NaN."""
    import mmprog.inference.virtual_map_v3 as vm
    from mmprog.components.deployment import Deployment

    calls = {"n": 0}
    real = vm.virtual_program_map_v3

    def flaky(img, ck, cfg, device, **kw):
        calls["n"] += 1
        if "detect_fn" in kw:                      # the parity call feeds the annotated centroids
            raise RuntimeError("boom in the inference path")
        return real(img, ck, cfg, device, **kw)

    monkeypatch.setattr(vm, "virtual_program_map_v3", flaky)
    comp = Deployment()
    ctx = comp.run(deploy_ctx, force=True)
    sanity = pd.read_csv(comp.path(ctx, "sanity_check.csv"))
    assert "boom" in str(sanity["parity_error"].iloc[0])
    assert not np.isfinite(sanity["parity_pearson"].iloc[0])
    assert calls["n"] >= 2
