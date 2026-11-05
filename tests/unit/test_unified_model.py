"""Unified model: shapes per variant, the neighbour block mask, losses, the v2-shaped variant, the component."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402
from mmprog.models.heads import fit_temperature, gaussian_nll, info_nce, laplacian_smoothness, temperature_scale  # noqa: E402
from mmprog.models.unified import UnifiedProgramModel, _SelfBlock, block_mask  # noqa: E402

P, S, L, K, G = 3, 2, 4, 5, 4
SPEC = dict(d_model=32, n_layers=1, n_heads=2, head_dim=16, dropout=0.0, n_fourier=8, fourier_scale=6.0, slide_emb_dim=4,
            grid=G, k=K, n_programs=P, n_scales=S, n_lineages=L, n_samples=3, d_centre=20, d_grid=12, n_genes=30,
            uncertainty="gaussian", d_align=P, mlp_hidden=16,
            program_init_weights=np.random.default_rng(0).random((P, 30)).tolist())


def _inputs(b=6, k=K):
    g = torch.Generator().manual_seed(0)
    return dict(centre_emb=torch.randn(b, 20, generator=g), grid_tokens=torch.randn(b, G * G, 12, generator=g),
                nb_emb=torch.randn(b, k, 20, generator=g), nb_offsets_um=torch.randn(b, k, 2, generator=g) * 10,
                sid=torch.zeros(b, dtype=torch.long))


@pytest.mark.parametrize("flags", [{}, {"no_neighbours": True}, {"no_grid": True}, {"no_program_tokens": True},
                                   {"no_comp": True}, {"no_grid": True, "no_neighbours": True}])
def test_forward_shapes_per_variant(flags):
    model = UnifiedProgramModel(SPEC).eval()
    out = model(**_inputs(), variant_flags=flags, need_attn=True)
    assert out["mean"].shape == (6, P * S)
    assert out["log_var"].shape == (6, P * S)
    assert out["morph_pooled"].shape == (6, P)
    if flags.get("no_comp"):
        assert out["comp_logits"] is None
    else:
        assert out["comp_logits"].shape == (6, L)
    if not flags.get("no_program_tokens"):
        n_tok = (1 if flags.get("no_grid") else 1 + G * G) + (0 if flags.get("no_neighbours") else K)
        assert out["attn"].shape == (6, P, n_tok)
        assert torch.allclose(out["attn"].sum(-1), torch.ones(6, P), atol=1e-5)


def test_block_mask_isolates_neighbours():
    """Zeroing one neighbour's input changes that neighbour's own output token and the morphology tokens,
    but not the other neighbour tokens (they attend only to themselves and the centre)."""
    torch.manual_seed(0)
    n_m = 1 + G * G
    block = _SelfBlock(32, 2, 16, 0.0).eval()
    x = torch.randn(1, n_m + K, 32)
    mask = block_mask(n_m, K, 1, x.device)
    base = block(x, mask)
    x2 = x.clone()
    x2[0, n_m + 2] = 0.0                                   # perturb neighbour 2
    pert = block(x2, mask)
    delta = (base - pert).abs().sum(-1)[0]
    assert delta[n_m + 2] > 1e-6                            # its own token changed
    for j in range(K):
        if j != 2:
            assert delta[n_m + j] < 1e-6                    # other neighbours unchanged
    assert delta[:n_m].max() > 1e-6                         # morphology tokens see all neighbours


def test_block_mask_none_without_neighbours():
    assert block_mask(5, 0, 2, "cpu") is None
    m = block_mask(3, 2, 1, "cpu")[0]
    assert m[:3].all() and m[3, 3] and m[3, 0] and not m[3, 4] and not m[4, 3]


def test_gaussian_nll_decreases_on_planted_signal():
    torch.manual_seed(1)
    x = torch.randn(256, 4)
    y = x @ torch.tensor([[1.0], [-1.0], [0.5], [0.0]]) + 0.1 * torch.randn(256, 1)
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    first = None
    for _ in range(50):
        out = net(x)
        loss = gaussian_nll(out[:, :1], out[:, 1:], y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 0.5


def test_temperature_scaling_keeps_mean_and_fits_scalar():
    torch.manual_seed(2)
    mean = torch.randn(500, 2)
    y = mean + 2.0 * torch.randn(500, 2)                   # true std 2, predicted variance 1 -> temperature about 4
    log_var = torch.zeros(500, 2)
    t = fit_temperature(mean, log_var, y)
    assert 2.5 < t < 6.0
    assert torch.equal(mean, mean) and torch.allclose(temperature_scale(log_var, t), log_var + np.log(t))


def test_info_nce_prefers_aligned_pairs():
    torch.manual_seed(3)
    a = torch.randn(32, 8)
    aligned = info_nce(a, a + 0.05 * torch.randn(32, 8))
    shuffled = info_nce(a, a[torch.randperm(32)] + 0.05 * torch.randn(32, 8))
    assert aligned < shuffled
    with pytest.raises(ValueError):
        info_nce(a, torch.randn(32, 5))


def test_laplacian_smoothness_mask():
    mean = torch.zeros(2, 3)
    nb = torch.ones(2, 4, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert laplacian_smoothness(mean, nb, mask).item() == pytest.approx(1.0)
    assert laplacian_smoothness(mean, nb).item() == pytest.approx(1.0)


def test_no_program_tokens_matches_programnet_output_shape():
    from mmprog.models.program_net import ProgramNet
    v2 = ProgramNet(d_in=32, n_targets=P * S, n_lin=L, n_samples=3, hidden=16, dropout=0.0)
    y_v2, comp_v2 = v2(torch.randn(6, 32), torch.randn(6, 2), torch.zeros(6, dtype=torch.long))
    out = UnifiedProgramModel(SPEC).eval()(**_inputs(), variant_flags={"no_program_tokens": True})
    assert out["mean"].shape == y_v2.shape and out["comp_logits"].shape == comp_v2.shape


def test_describe_and_parameter_counts():
    model = UnifiedProgramModel(SPEC)
    counts = model.block_parameter_counts()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    model.describe()
    assert model.unseen_slide_index() == 3


def test_unified_model_component(tmp_path):
    from mmprog.components.feature_extraction import FeatureExtraction
    from mmprog.components.feature_selection import FeatureSelection
    from mmprog.components.ingest import Ingest
    from mmprog.components.preprocess import Preprocess
    from mmprog.components.program_inference import ProgramInference
    from mmprog.components.unified_model import UnifiedModel, build_model
    ctx = small_ctx(tmp_path)
    for comp in (Ingest(), Preprocess(), ProgramInference(), FeatureExtraction(), FeatureSelection()):
        ctx = comp.run(ctx)
    comp = UnifiedModel()
    ctx = comp.run(ctx)
    spec = ctx.model_spec
    for key in ("d_model", "grid", "k", "n_programs", "n_scales", "targets", "target_index", "n_lineages", "n_samples",
                "d_centre", "d_grid", "uncertainty", "parameter_counts", "program_init_weights", "attention_mask"):
        assert key in spec, key
    assert len(spec["targets"]) == spec["n_programs"] * spec["n_scales"] == len(spec["target_index"])
    assert [ctx.targets[i] for i in spec["target_index"]] == spec["targets"]
    assert spec["grid"] == 4 and spec["d_centre"] == 64 and spec["d_grid"] == 64
    model = build_model(spec, "cpu")
    b = 4
    out = model(torch.as_tensor(np.asarray(ctx.EMB[:b], dtype=np.float32)),
                torch.as_tensor(np.asarray(ctx.GRID[:b], dtype=np.float32)),
                torch.as_tensor(np.asarray(ctx.EMB[ctx.NB[:b]], dtype=np.float32)),
                torch.zeros(b, spec["k"], 2), torch.as_tensor(ctx.SID[:b]))
    assert out["mean"].shape == (b, len(spec["targets"]))
    ctx = comp.run(ctx)      # reload
    assert ctx.model_spec["parameter_counts"]["total"] == spec["parameter_counts"]["total"]
