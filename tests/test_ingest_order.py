"""The real-data ingest helper runs QC -> count summaries on all cells -> image crop -> subsample -> attach.

Uses a stub image store and monkeypatched image readers, so no TIFF is needed.
"""
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.config import Config  # noqa: E402
from mmprog import pipeline as pl  # noqa: E402
from mmprog.preprocessing.spatial_summaries import attach_summaries, count_summaries, subsample_cells  # noqa: E402


class StubStore:
    def __init__(self):
        self.calls = []

    def prepare(self, sample, xy):
        self.calls.append((sample, np.asarray(xy).copy()))


def _toy_adata(n=400, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["WNT5A", "WNT5B", "FRZB", "CD38", "XBP1", "OTHER"]
    X = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    X[:50] = 0                                                   # 50 cells fail QC (no transcripts)
    obs = pd.DataFrame({"x": rng.uniform(0, 500, n).astype(np.float32),
                        "y": rng.uniform(0, 300, n).astype(np.float32)},
                       index=[f"c{i}" for i in range(n)])
    return ad.AnnData(csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))


def test_ingest_one_sample_order(tmp_path, monkeypatch):
    cfg = Config(run_name="t", root=str(tmp_path), local_data=str(tmp_path), img_level=0,
                 um_per_px_level0=1.0, min_transcripts=1, min_genes=1, max_cells_per_sample=100,
                 min_panel_genes=3, nb_radius_um=40)
    a = _toy_adata()
    # stand-ins for the image readers and loader (the test is about order, not TIFF decoding)
    monkeypatch.setattr(pl, "ingest_one_sample", pl.ingest_one_sample)
    import mmprog.data.images as images
    import mmprog.data.tenx as tenx
    monkeypatch.setattr(images, "um_per_px_level0", lambda path, default: 1.0)
    monkeypatch.setattr(images, "level_shape", lambda path, level: (300, 500))
    monkeypatch.setattr(tenx, "load_row", lambda row, cfg: a.copy())
    store = StubStore()
    programs = {"wnt_camdr": ["WNT5A", "WNT5B", "FRZB"], "missing": ["ZZZ"]}
    row = dict(sample="S", patient="S", stage="MM", image="fake.tif", h5="x", cells="y")

    a_sub, a_full = pl.ingest_one_sample(row, programs, cfg, store, count_summaries, attach_summaries, subsample_cells)

    assert a_full.n_obs == 350, "QC ran before summaries (50 empty cells dropped)"
    assert a_sub.n_obs == 100, "subsample happened after summaries"
    assert store.calls and store.calls[0][0] == "S" and len(store.calls[0][1]) == 350, \
        "image crop was requested with every QC-passing cell, not the subsample"
    assert "cnt_wnt_camdr" in a_sub.obs and "nb_cnt_total" in a_sub.obs and "density_r25" in a_sub.obs
    # summaries on the subsample equal those computed on the full slide for the same cells
    full_summ = count_summaries(a_full, programs, cfg, px_per_um=1.0)
    joined = full_summ.loc[a_sub.obs_names]
    assert np.allclose(joined["nb_cnt_wnt_camdr"].values, a_sub.obs["nb_cnt_wnt_camdr"].values)
    assert row["axes_swapped"] is False


def test_ingest_one_sample_swaps_transposed_axes(tmp_path, monkeypatch):
    cfg = Config(run_name="t", root=str(tmp_path), local_data=str(tmp_path), img_level=0,
                 um_per_px_level0=1.0, min_transcripts=1, min_genes=1, max_cells_per_sample=None,
                 min_panel_genes=3)
    a = _toy_adata()
    import mmprog.data.images as images
    import mmprog.data.tenx as tenx
    monkeypatch.setattr(images, "um_per_px_level0", lambda path, default: 1.0)
    monkeypatch.setattr(images, "level_shape", lambda path, level: (500, 300))   # H=500, W=300: fits only transposed
    monkeypatch.setattr(tenx, "load_row", lambda row, cfg: a.copy())
    store = StubStore()
    row = dict(sample="S", patient="S", stage="MM", image="fake.tif", h5="x", cells="y")
    a_sub, _ = pl.ingest_one_sample(row, {"p": ["WNT5A"]}, cfg, store, count_summaries, attach_summaries, subsample_cells)
    assert row["axes_swapped"] is True
    assert a_sub.obs["x"].max() <= 300 and a_sub.obs["y"].max() <= 500
    assert store.calls[0][1][:, 0].max() <= 300
