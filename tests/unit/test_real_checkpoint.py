"""Interpretation and deployment against a REAL final_model.pt produced by the v3 components.

The fake-model tests prove the plumbing; these prove that the checkpoint written by Training is
what Interpretation and Deployment read: integrated-gradient rows are labelled with the head
targets, attention mass is non-empty, the virtual map on a training slide's own centroids
reproduces the training-time predictions, and a mismatched grid is refused.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402


@pytest.fixture(scope="module")
def trained_ctx(tmp_path_factory):
    pytest.importorskip("torch")
    from mmprog.components.ingest import Ingest
    from mmprog.components.preprocess import Preprocess
    from mmprog.components.program_inference import ProgramInference
    from mmprog.components.feature_extraction import FeatureExtraction
    from mmprog.components.feature_selection import FeatureSelection
    from mmprog.components.unified_model import UnifiedModel
    from mmprog.components.training import Training

    tmp = tmp_path_factory.mktemp("real_ckpt")
    ctx = small_ctx(tmp, n_patients=4, cells_per=400, max_epochs=3, batch_cells=256, warmup_steps=2,
                    programs_max_iter=20, grid=4, model_variants="[full,shuffled]", attribution_n_cells=120, ig_steps=4,
                    srb_rounds=8, max_cells_per_sample=400)
    for comp in (Ingest(), Preprocess(), ProgramInference(), FeatureExtraction(), FeatureSelection(), UnifiedModel(), Training()):
        ctx = comp.run(ctx)
    return ctx


@pytest.mark.slow
def test_interpretation_labels_ig_rows_with_head_targets(trained_ctx):
    import torch
    from mmprog.components.interpretation import Interpretation

    ctx = Interpretation().run(trained_ctx)
    ck = torch.load(ctx.final_model_path, map_location="cpu", weights_only=False)
    attr = pd.read_csv(os.path.join(ctx.component_dir("interpretation"), "pathway_attribution.csv"))
    kinds = set(attr["kind"])
    assert "attention_mass" in kinds and "integrated_gradients" in kinds, kinds
    ig = attr[attr["kind"] == "integrated_gradients"]
    assert set(ig["target"]) == set(ck["targets"]), (sorted(set(ig["target"]))[:3], ck["targets"][:3])
    att = attr[attr["kind"] == "attention_mass"]
    assert len(att) > 0 and set(att["program"]) <= set(ck["spec"]["programs"])


@pytest.mark.slow
def test_virtual_map_reproduces_training_predictions_on_annotated_centroids(trained_ctx):
    """Feeding the training slide's own cell centroids as 'detected nuclei' must give the same predictions
    the training-time predict() gives for those cells (same encoder, grid, feature space, selector, temperature)."""
    from mmprog.inference.virtual_map_v3 import load_checkpoint, virtual_program_map_v3
    from mmprog.training.cv import head_layout, load_final
    from mmprog.training.data import CellBatcher, variant_flags
    from mmprog.training.loop import predict

    ctx = trained_ctx
    s0 = ctx.obs["sample"].iloc[0]
    rows = np.where(ctx.obs["sample"].to_numpy() == s0)[0]
    ox, oy = ctx.images.offset(s0)
    xy_crop = ctx.obs[["x", "y"]].to_numpy(np.float32)[rows] - np.array([ox, oy], dtype=np.float32)
    ck = load_checkpoint(ctx.final_model_path, "cpu")
    vmap = virtual_program_map_v3(ctx.images.get(s0), ck, ctx.cfg, "cpu", detect_fn=lambda img: xy_crop.copy(),
                                  sample_name=str(s0), batch_size=256)
    assert len(vmap) == len(rows)
    main = "prog_wnt_camdr_nb_z"
    col = main.replace("_z", "")                      # vmap columns drop the _z suffix: <target>_pred, <target>_std
    assert f"{col}_pred" in vmap.columns and f"{col}_std" in vmap.columns
    assert any(c.startswith("lineage_prob_") for c in vmap.columns)

    # training-time predictions for the same cells with the saved feature space and selector
    model, raw = load_final(ctx.final_model_path, "cpu")
    keep, targets = head_layout(ctx, "full")
    X = raw["feature_space"].transform(ctx.EMB, ctx.MOR)
    Y = np.asarray(ctx.Y, dtype=np.float32)[:, keep]
    xy_um = ctx.obs[["x", "y"]].to_numpy(np.float32) * ctx.um_per_px_per_cell()[:, None]
    b = CellBatcher(cell_idx=rows, X_centre=X, GRID=ctx.GRID, NB=ctx.NB, NB_DIST=ctx.NB_DIST, XY_um=xy_um, SID=ctx.SID, Y=Y,
                    LIN=ctx.lineage, expression_embedding=ctx.expression_embedding, batch_cells=256, name="chk")
    mean, std, comp, _ = predict(model, b, ctx.cfg, "cpu", variant_flags("full"))
    j = targets.index(main)
    r = np.corrcoef(vmap[f"{col}_pred"].to_numpy(), mean[:, j])[0, 1]
    # the only differences are the unseen-slide embedding slot and the neighbour graph rebuilt from the same
    # centroids, so the two must agree closely
    assert r > 0.95, r


@pytest.mark.slow
def test_virtual_map_refuses_mismatched_grid(trained_ctx):
    from mmprog.inference.virtual_map_v3 import extract_inference_features, load_checkpoint

    ctx = trained_ctx
    ck = load_checkpoint(ctx.final_model_path, "cpu")
    spec = dict(ck["spec"])
    s0 = ctx.obs["sample"].iloc[0]
    obs = pd.DataFrame({"x": [40.0, 80.0], "y": [40.0, 80.0], "sample": "NEW"})

    class Store:
        def get(self, s):
            return ctx.images.get(s0)

        def offset(self, s):
            return (0, 0)

    # a grid_fn that returns the wrong number of tokens must be refused by the shape check
    bad = lambda obs_, store, cfg_, grid: np.zeros((len(obs_), 9, int(spec["d_grid"])), np.float32)  # noqa: E731
    emb_fn = lambda obs_, store, enc, path, cfg_, device: np.zeros((len(obs_), int(spec["d_emb_raw"])), np.float32)  # noqa: E731
    with pytest.raises(ValueError, match="tokens per cell"):
        extract_inference_features(obs, Store(), ctx.cfg, "cpu", spec, embed_fn=emb_fn, grid_fn=bad,
                                   encoder_fn=lambda name, device: None)
    # a wrong raw embedding dimension is refused as well
    ok_grid = lambda obs_, store, cfg_, grid: np.zeros((len(obs_), grid * grid, int(spec["d_grid"])), np.float32)  # noqa: E731
    bad_emb = lambda obs_, store, enc, path, cfg_, device: np.zeros((len(obs_), int(spec["d_emb_raw"]) + 1), np.float32)  # noqa: E731
    with pytest.raises(ValueError, match="mismatched features"):
        extract_inference_features(obs, Store(), ctx.cfg, "cpu", spec, embed_fn=bad_emb, grid_fn=ok_grid,
                                   encoder_fn=lambda name, device: None)


@pytest.mark.slow
def test_deployment_records_inference_parity_on_real_checkpoint(trained_ctx):
    """The deployment component writes parity_pearson from the real checkpoint, and it is close to 1."""
    from mmprog.components.deployment import Deployment

    ctx = trained_ctx
    ctx.component_manifests.setdefault("training", {"hash": "t"})
    comp = Deployment()
    comp.run(ctx, force=True)
    sanity = pd.read_csv(comp.path(ctx, "sanity_check.csv"))
    r = float(sanity["parity_pearson"].iloc[0])
    assert np.isfinite(r) and r > 0.95, r
