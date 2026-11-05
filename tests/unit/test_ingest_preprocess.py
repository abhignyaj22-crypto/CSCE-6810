"""Ingest and Preprocess components: provides, artefacts, manifest, skip rule, invalidation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402
from mmprog.components.ingest import Ingest  # noqa: E402
from mmprog.components.preprocess import Preprocess  # noqa: E402


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    c = small_ctx(tmp_path_factory.mktemp("ws"))
    Ingest().run(c)
    Preprocess().run(c)
    return c


def test_ingest_provides_and_artefacts(ctx):
    comp = Ingest()
    for field in comp.provides:
        assert getattr(ctx, field) is not None, field
    for a in comp.artefacts + ("images.npz",):
        assert os.path.exists(comp.path(ctx, a)), a
    assert os.path.exists(comp.manifest_path(ctx))
    assert len(ctx.manifest) == 3 and ctx.adata_raw.n_obs == 3 * 300   # the demo fixture is not subsampled
    # count summaries were computed before subsampling, on all cells of the slide
    assert "cnt_total" in ctx.adata_raw.obs and "nb_n" in ctx.adata_raw.obs
    assert all(len(ctx.full_xy[s]) == 300 for s in ctx.full_xy)
    img = ctx.images.get(ctx.manifest[0]["sample"])
    assert img.dtype == np.uint8 and ctx.images.offset(ctx.manifest[0]["sample"]) == (0, 0)


def test_preprocess_provides_and_artefacts(ctx):
    comp = Preprocess()
    for field in comp.provides:
        assert getattr(ctx, field) is not None, field
    for a in comp.artefacts:
        assert os.path.exists(comp.path(ctx, a)), a
    assert "counts" in ctx.adata.layers
    assert ctx.gene_mask.dtype == bool and ctx.gene_mask.shape == (ctx.adata.n_vars,)
    assert set(ctx.nuclei) == {m["sample"] for m in ctx.manifest}
    assert all(v.shape[1] == 4 for v in ctx.nuclei.values())    # x, y, offset_x, offset_y
    assert len(ctx.obs) == ctx.adata.n_obs


def test_second_run_loads_instead_of_recomputing(ctx, monkeypatch):
    for comp in (Ingest(), Preprocess()):
        calls = {"n": 0}

        def boom(self, c, _calls=calls):
            _calls["n"] += 1
            raise AssertionError("_run called on a valid artefact set")

        monkeypatch.setattr(type(comp), "_run", boom)
        assert comp.is_done(ctx)
        comp.run(ctx)
        assert calls["n"] == 0


def test_config_change_invalidates(ctx):
    comp = Preprocess()
    assert comp.is_done(ctx)
    old = ctx.cfg.min_transcripts
    ctx.cfg.min_transcripts = old + 1
    try:
        assert not comp.is_done(ctx)
    finally:
        ctx.cfg.min_transcripts = old
    assert comp.is_done(ctx)


def test_upstream_change_invalidates_downstream(ctx):
    pre = Preprocess()
    assert pre.is_done(ctx)
    saved = ctx.component_manifests["ingest"]["hash"]
    ctx.component_manifests["ingest"]["hash"] = "different"
    try:
        assert not pre.is_done(ctx)
    finally:
        ctx.component_manifests["ingest"]["hash"] = saved


def test_gene_mask_masks_genes_absent_in_one_sample(ctx):
    from scipy.sparse import csr_matrix
    adata = ctx.adata.copy()
    X = csr_matrix(adata.layers["counts"]).tolil()
    s0 = adata.obs["sample"].to_numpy() == ctx.manifest[0]["sample"]
    X[np.where(s0)[0], 0] = 0          # gene 0 disappears from sample 0 only
    adata.layers["counts"] = X.tocsr()
    mask = Preprocess._gene_mask(adata)
    assert not mask[0] and mask[1:].sum() >= adata.n_vars - 2
