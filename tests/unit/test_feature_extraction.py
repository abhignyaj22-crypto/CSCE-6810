"""FeatureExtraction and FeatureSelection components on the tiny synthetic fixture."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_cfg, small_ctx  # noqa: E402
from mmprog.features.encoders import ENCODERS, check_licence, get_encoder  # noqa: E402


def _upstream(tmp_path):
    from mmprog.components.ingest import Ingest
    from mmprog.components.preprocess import Preprocess
    from mmprog.components.program_inference import ProgramInference
    ctx = small_ctx(tmp_path)
    for comp in (Ingest(), Preprocess(), ProgramInference()):
        ctx = comp.run(ctx)
    return ctx


def test_registry_refuses_non_commercial_when_commercial_use_true(tmp_path):
    cfg = small_cfg(tmp_path, commercial_use=True)
    with pytest.raises(PermissionError):
        check_licence("uni", cfg)
    with pytest.raises(PermissionError):
        check_licence("phikon_v2", cfg)
    assert check_licence("hoptimus0", cfg)["commercial"] is True
    assert check_licence("resnet50", cfg)["commercial"] is True
    with pytest.raises(ValueError):
        check_licence("not_an_encoder", cfg)


def test_registry_cards_carry_provenance_fields():
    for name, card in ENCODERS.items():
        for key in ("licence", "dim", "input_px", "source", "paper", "gated", "commercial", "mean", "std"):
            assert key in card, f"{name} lacks {key}"


def test_tiny_encoder_exposes_feature_map(tmp_path):
    import torch
    enc, card = get_encoder("tiny_test", "cpu", small_cfg(tmp_path))
    x = torch.zeros(2, 3, 224, 224)
    assert enc.forward_features(x).shape == (2, 64, 8, 8)
    assert enc(x).shape == (2, 64)
    assert card["weight_hash"] and card["name"] == "tiny_test"
    assert all(not p.requires_grad for p in enc.parameters())


def test_feature_extraction_component_runs_and_reloads(tmp_path):
    from mmprog.components.feature_extraction import FeatureExtraction
    ctx = _upstream(tmp_path)
    n = len(ctx.obs)
    comp = FeatureExtraction()
    ctx = comp.run(ctx)
    assert ctx.EMB.shape == (n, 64) and ctx.EMB.dtype == np.float16
    assert ctx.GRID.shape == (n, 16, 64) and ctx.GRID.dtype == np.float16       # grid=4 in the fixture
    assert ctx.MOR.shape[0] == n and len(ctx.morpho_names) == ctx.MOR.shape[1]
    assert ctx.NB.shape == (n, ctx.cfg.knn_k) and ctx.NB_DIST.shape == ctx.NB.shape
    assert ctx.NB_DIST.min() >= 0 and ctx.XY.min() >= -1.0 and ctx.XY.max() <= 1.0
    assert ctx.n_samples == ctx.obs["sample"].nunique() and len(ctx.GROUPS) == n
    assert np.abs(np.asarray(ctx.EMB[:50], dtype=np.float32)).sum() > 0
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a))
    assert ctx.encoder_card["grid_tokens"] == 16

    # second run must load, not recompute
    calls = {"n": 0}
    original = comp._run

    def counting(c):
        calls["n"] += 1
        return original(c)

    comp._run = counting
    ctx2 = comp.run(ctx)
    assert calls["n"] == 0 and ctx2.EMB.shape == (n, 64)


def test_feature_selection_component(tmp_path):
    from mmprog.components.feature_extraction import FeatureExtraction
    from mmprog.components.feature_selection import FeatureSelection
    ctx = _upstream(tmp_path)
    ctx = FeatureExtraction().run(ctx)
    comp = FeatureSelection()
    ctx = comp.run(ctx)
    assert ctx.selection["keep_emb_dims"] is not None and len(ctx.selection["keep_emb_dims"]) <= 64
    assert ctx.selection["deterministic"] is True
    assert os.path.exists(comp.path(ctx, "feature_selection.json")) and os.path.exists(comp.path(ctx, "selection_report.csv"))
    ctx = comp.run(ctx)      # reload path
    assert "report" in ctx.selection
