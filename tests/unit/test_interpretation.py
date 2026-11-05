"""Interpretation component and attribution helpers, with a fake unified model."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from mmprog.config import load_config  # noqa: E402
from mmprog.core.context import Context  # noqa: E402
from mmprog.interpretation.attribution import (attention_attribution, default_token_groups,  # noqa: E402
                                               integrated_gradients, summarise_attributions)

GRID = 4
K = 3
D = 6
T = 4
L = 3
N_TOKENS = 1 + GRID * GRID + K


class FakeUnifiedModel(torch.nn.Module):
    """Linear in the centre features; attention is a fixed per-program distribution over tokens."""
    accepts_batch_dict = True

    def __init__(self, n_programs=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(D, T, generator=g))
        self.C = torch.nn.Parameter(torch.randn(D, L, generator=g))
        self.attn_logits = torch.nn.Parameter(torch.randn(n_programs, N_TOKENS, generator=g))
        self.temperature = 1.0

    def forward(self, batch):
        centre = batch["centre"].float()
        b = centre.shape[0]
        mean = centre @ self.W
        return dict(mean=mean, log_var=torch.zeros_like(mean), comp_logits=centre @ self.C,
                    morph_pooled=centre, attn=torch.softmax(self.attn_logits, -1).expand(b, -1, -1))


def _arrays(n=40, seed=1):
    rng = np.random.default_rng(seed)
    return dict(centre=rng.normal(size=(n, D)).astype(np.float32), grid=rng.normal(size=(n, GRID * GRID, 2)).astype(np.float32),
                nb=rng.normal(size=(n, K, D)).astype(np.float32), nb_offset=rng.normal(size=(n, K, 2)).astype(np.float32),
                nb_mask=np.ones((n, K), dtype=bool), sid=np.zeros(n, dtype=np.int64), morpho=rng.normal(size=(n, 3)).astype(np.float32))


def test_default_token_groups_partition_tokens():
    groups = default_token_groups(GRID, block_um=3.4, k=K)
    all_idx = np.concatenate(list(groups.values()))
    assert sorted(all_idx.tolist()) == list(range(N_TOKENS))


def test_attention_attribution_mass_sums_to_one_per_program():
    model = FakeUnifiedModel()
    groups = default_token_groups(GRID, 3.4, K)
    lin = np.array(["plasma", "T"] * 20)
    df = attention_attribution(model, _arrays(), [f"t{i}" for i in range(T)], ["p0", "p1"], groups, lineage_names=lin)
    for key in ("all", "plasma", "T"):
        for p in ("p0", "p1"):
            mass = df[(df.lineage == key) & (df.program == p)]["mass"].sum()
            assert mass == pytest.approx(1.0, abs=1e-5)
    expected = torch.softmax(model.attn_logits, -1)[0, groups["centre_token"]].sum().item()
    got = df[(df.lineage == "all") & (df.program == "p0") & (df.group == "centre_token")]["mass"].iloc[0]
    assert got == pytest.approx(expected, abs=1e-5)


def test_integrated_gradients_recovers_linear_weights():
    model = FakeUnifiedModel()
    arr = _arrays(n=8)
    attr = integrated_gradients(model, arr, target_idx=2, attr_keys=("centre",), steps=16)
    expected = arr["centre"] * model.W[:, 2].detach().numpy()[None, :]
    assert np.allclose(attr["centre"], expected, atol=1e-5)
    summary = summarise_attributions(attr, top=3)
    assert set(summary.columns) >= {"input", "dim", "mean_abs_attribution", "rank"} and len(summary) == 3


@pytest.fixture
def small_ctx(tmp_path):
    """Context with everything Interpretation requires, from the synthetic fixture (no adata: single-fit backbone)."""
    from mmprog.data.images import ImageStore
    from mmprog.data.synthetic import make_demo
    from mmprog.models.neighbours import knn_table
    from mmprog.resources import load_lineages, load_programs

    cfg = load_config(overrides=[f"root={tmp_path}", f"local_data={tmp_path}/raw", "run_name=interp", "demo_mode=true",
                                 "srb_grid=4", "srb_rounds=10", "srb_lambdas=[0.0,0.3]", "n_folds=2", "knn_k=3",
                                 "attribution_n_cells=60", "ig_steps=4", "motifs_enabled=true", "patch_px=32"])
    programs, lineages = load_programs(), load_lineages()
    adata, images, manifest = make_demo(programs, lineages, cfg, n_patients=3, cells_per=120, img_size=256, seed=3)
    ctx = Context.new(cfg)
    obs = adata.obs.reset_index(drop=True)
    obs["lineage_name"] = obs["lineage_true"].astype(str)
    n = len(obs)
    prog_names = ["wnt_camdr", "cytotoxicity"]
    targets = [f"prog_{p}_{s}_z" for p in prog_names for s in ("cell", "nb")]
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(n, len(targets))).astype(np.float32)
    Y[:, 1] += (obs["lineage_name"] == "plasma").to_numpy() * 2.0
    xy_by_sample = {s: obs.loc[obs["sample"] == s, ["x", "y"]].to_numpy(np.float32) for s in obs["sample"].unique()}
    ctx.set("images", ImageStore(cfg, manifest, xy_by_sample, demo_images=images))
    ctx.set("obs", obs)
    ctx.set("Y", Y)
    ctx.set("targets", targets)
    ctx.set("SID", pd.Categorical(obs["sample"]).codes.astype(np.int64))
    ctx.set("GROUPS", obs["patient"].astype(str).to_numpy())
    ctx.set("EMB", rng.normal(size=(n, D)).astype(np.float32))
    ctx.set("GRID", rng.normal(size=(n, GRID * GRID, 2)).astype(np.float32))
    ctx.set("MOR", rng.normal(size=(n, 3)).astype(np.float32))
    ctx.set("NB", knn_table(obs, K))
    ctx.set("lin_names", ["plasma", "T", "B"])
    ctx.set("programs", {p: programs[p] for p in prog_names})
    genes = sorted(set(g for p in prog_names for g in programs[p]))
    H = pd.DataFrame(rng.uniform(size=(len(prog_names), len(genes))), index=prog_names, columns=genes)
    for p in prog_names:
        H.loc[p, programs[p][:5]] = 1.0
    ctx.set("program_weights", H)
    ck = dict(model=FakeUnifiedModel(), spec=dict(grid=GRID, k=K, n_samples=3, program_names=prog_names),
              targets=targets, lin_names=ctx.lin_names, temperature=1.0, fs=None, selector=None, flags={})
    path = str(tmp_path / "final_model.pt")
    torch.save(ck, path)
    ctx.set("final_model_path", path)
    ctx.component_manifests["training"] = {"hash": "t"}
    ctx.component_manifests["program_inference"] = {"hash": "p"}
    return ctx


def test_interpretation_component_runs_and_skips(small_ctx):
    from mmprog.components.interpretation import Interpretation

    comp = Interpretation()
    ctx = comp.run(small_ctx)
    for name in comp.provides:
        assert getattr(ctx, name) is not None, name
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    assert (ctx.srb_results["lam"] == 0.0).any()
    assert set(ctx.gene_backbone["program"]) == {"wnt_camdr", "cytotoxicity"}
    assert "wnt_camdr" in ctx.enrichment["gene_set"].values
    assert ctx.pathway_attr[ctx.pathway_attr.kind == "attention_mass"]["mass"].gt(0).any()
    # second run loads instead of recomputing
    calls = {"n": 0}
    orig = comp._run

    def counting(ctx_):
        calls["n"] += 1
        return orig(ctx_)

    comp._run = counting
    comp.run(ctx)
    assert calls["n"] == 0 and comp.is_done(ctx)
