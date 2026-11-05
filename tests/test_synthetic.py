"""make_demo produces a complete, self-consistent fixture."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from mmprog.config import Config
from mmprog.data.synthetic import make_demo
from mmprog.resources import load_lineages, load_programs


def test_make_demo_shapes_and_columns():
    cfg = Config(demo_mode=True)
    programs, lineages = load_programs(), load_lineages()
    adata, images, manifest = make_demo(programs, lineages, cfg, n_patients=3, cells_per=400, img_size=256, seed=1)
    assert adata.n_obs == 3 * 400
    assert len(manifest) == 3 and len(images) == 3
    for col in ["x", "y", "sample", "patient", "stage", "lineage_true", "cnt_total", "nb_n", "nb_cnt_total"]:
        assert col in adata.obs, col
    for p in programs:
        assert f"cnt_{p}" in adata.obs and f"nb_cnt_{p}" in adata.obs
    for r in cfg.density_radii_px:
        assert f"density_r{r}" in adata.obs
    assert (adata.obs["cnt_total"] > 0).all()
    assert (adata.obs["nb_n"] >= 1).all()
    for img in images.values():
        assert img.dtype == np.uint8 and img.shape == (256, 256)
    assert set(adata.obs["lineage_true"]) <= set(lineages)
    assert adata.obs["stage"].isin(["control", "SMM", "MM"]).all()
    # planted signal: plasma cells carry more wnt_camdr counts than other cells
    pc = adata.obs["lineage_true"] == "plasma"
    assert adata.obs.loc[pc, "cnt_wnt_camdr"].mean() > adata.obs.loc[~pc, "cnt_wnt_camdr"].mean()
