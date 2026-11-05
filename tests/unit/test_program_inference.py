"""ProgramInference: anchored NMF recovers the planted program, both target families and their ceilings exist,
and the component reloads instead of recomputing on a second run."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402
from mmprog.components.program_inference import ProgramInference  # noqa: E402
from mmprog.data.synthetic import make_demo  # noqa: E402
from mmprog.preprocessing.qc import apply_cell_qc, calculate_qc_metrics, normalize_log1p  # noqa: E402
from mmprog.programs.anchored_nmf import AnchoredNMF  # noqa: E402
from mmprog.resources import load_lineages, load_programs  # noqa: E402


class _Images:
    """Stand-in image store: ProgramInference only checks that images are present."""

    def __init__(self, images):
        self._im = images

    def get(self, s):
        return self._im[s]

    def offset(self, s):
        return (0, 0)


def preprocessed_ctx(tmp_path, n_patients=4, cells_per=400):
    """Fill the context the way Preprocess would, without depending on that component."""
    ctx = small_ctx(tmp_path, n_patients=n_patients, cells_per=cells_per)
    programs, lineages = load_programs(), load_lineages()
    adata, images, manifest = make_demo(programs, lineages, ctx.cfg, n_patients=n_patients, cells_per=cells_per,
                                        img_size=256, seed=ctx.cfg.seed)
    calculate_qc_metrics(adata)
    adata = apply_cell_qc(adata, ctx.cfg)
    adata = normalize_log1p(adata)
    ctx.set("manifest", manifest)
    ctx.set("adata", adata)
    ctx.set("obs", adata.obs.reset_index(drop=True))
    ctx.set("gene_mask", np.ones(adata.n_vars, dtype=bool))
    ctx.set("images", _Images(images))
    ctx.component_manifests["preprocess"] = {"hash": "test-preprocess"}
    return ctx


@pytest.fixture(scope="module")
def run_ctx(tmp_path_factory):
    ctx = preprocessed_ctx(tmp_path_factory.mktemp("pi"))
    ProgramInference().run(ctx)
    return ctx


def test_anchored_nmf_puts_weight_on_anchors(run_ctx):
    H = run_ctx.program_weights
    anchors = run_ctx.programs["wnt_camdr"]
    w = H.loc["wnt_camdr"]
    assert w[anchors].sum() / w.sum() >= 0.6
    assert set(H.index) >= set(run_ctx.programs) and any(p.startswith("extra_") for p in H.index)


def test_learned_target_tracks_anchor_target(run_ctx):
    obs = run_ctx.obs
    assert "prog_wnt_camdr_nb_z" in run_ctx.targets and "anchor_wnt_camdr_nb_z" in run_ctx.targets
    r = np.corrcoef(obs["prog_wnt_camdr_nb_z"], obs["anchor_wnt_camdr_nb_z"])[0, 1]
    assert r > 0.8, r
    assert run_ctx.target_family["prog_wnt_camdr_nb_z"] == "learned"
    assert run_ctx.target_family["anchor_wnt_camdr_nb_z"] == "anchor_only"


def test_provides_shapes_and_artefacts(run_ctx):
    ctx = run_ctx
    N, T = ctx.Y.shape
    assert T == len(ctx.targets) and N == len(ctx.obs)
    assert ctx.program_scores.shape == (N, ctx.program_weights.shape[0], len(ctx.cfg.target_spaces))
    assert ctx.expression_embedding.shape == (N, ctx.program_weights.shape[0])
    assert ctx.LIN.shape == (N, len(ctx.lin_names)) and ctx.LIN.sum() == N
    for a in ProgramInference.artefacts:
        assert os.path.exists(os.path.join(ctx.run_dir, "program_inference", a)), a


def test_reliability_has_both_families(run_ctx):
    rel = run_ctx.reliability
    fams = set(run_ctx.target_family[t] for t in rel["target"].unique())
    assert fams == {"learned", "anchor_only"}
    summ = run_ctx.reliability_summary
    assert summ.loc["prog_wnt_camdr_nb_z", "median"] > 0.5
    assert all(k[0] in run_ctx.targets for k in run_ctx.rel_lookup)


def test_second_run_loads(tmp_path, monkeypatch):
    ctx = preprocessed_ctx(tmp_path, n_patients=3, cells_per=300)
    comp = ProgramInference()
    comp.run(ctx)
    calls = {"n": 0}
    original = ProgramInference._run

    def counting(self, c):
        calls["n"] += 1
        return original(self, c)

    monkeypatch.setattr(ProgramInference, "_run", counting)
    ctx2 = preprocessed_ctx(tmp_path, n_patients=3, cells_per=300)
    ProgramInference().run(ctx2)
    assert calls["n"] == 0
    assert ctx2.targets == ctx.targets and np.allclose(ctx2.Y, ctx.Y)


def test_transform_is_nonnegative_projection():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(40)]
    H_true = np.zeros((2, 40))
    H_true[0, :10] = 1
    H_true[1, 20:30] = 1
    W_true = rng.uniform(0, 2, size=(500, 2))
    X = W_true @ H_true + rng.uniform(0, 0.05, size=(500, 40))
    nmf = AnchoredNMF(n_extra=0, max_iter=100, seed=0, verbose_every=50).fit(X, genes, {"a": genes[:10], "b": genes[20:30]})
    W = nmf.transform(X)
    assert (W >= 0).all() and W.shape == (500, 2)
    assert np.corrcoef(W[:, 0], W_true[:, 0])[0, 1] > 0.95
    assert nmf.H_.loc["a", genes[:10]].sum() / nmf.H_.loc["a"].sum() > 0.9
