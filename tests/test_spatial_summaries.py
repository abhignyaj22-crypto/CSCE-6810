"""Neighbourhood sums and density on five hand-placed cells."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mmprog.config import Config
from mmprog.preprocessing.spatial_summaries import attach_summaries, count_summaries, subsample_cells


def _toy():
    # cells on a line at x = 0, 10, 20, 100, 110 (pixels); y = 0
    X = np.array([[1, 0, 2], [2, 1, 0], [0, 3, 1], [4, 0, 0], [1, 1, 1]], dtype=np.float32)
    obs = pd.DataFrame({"x": [0, 10, 20, 100, 110], "y": [0] * 5, "sample": "s"},
                       index=[f"c{i}" for i in range(5)]).astype({"x": np.float32, "y": np.float32})
    return ad.AnnData(csr_matrix(X), obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))


def test_count_summaries_hand_computed():
    # um_per_px_level0=0.5 at img_level=1 -> 1 um per px -> px_per_um = 1; radius 15 um = 15 px.
    # The explicit px_per_um check below is what guards the micron-to-pixel direction.
    cfg = Config(um_per_px_level0=0.5, img_level=1, nb_radius_um=15, density_radii_px=(12, 200))
    assert np.isclose(cfg.um_per_px, 1.0) and np.isclose(cfg.px_per_um, 1.0)
    a = _toy()
    programs = {"p": ["g1", "g3"], "missing": ["zzz"]}
    s = count_summaries(a, programs, cfg)
    assert np.allclose(s["cnt_total"], [3, 3, 4, 4, 3])
    assert np.allclose(s["cnt_p"], [3, 2, 1, 4, 2])
    assert np.allclose(s["cnt_missing"], 0)
    # radius 15 px: cell0 sees {0,1}, cell1 sees {0,1,2}, cell2 sees {1,2}, cell3 sees {3,4}, cell4 sees {3,4}
    assert np.allclose(s["nb_n"], [2, 3, 2, 2, 2])
    assert np.allclose(s["nb_cnt_total"], [6, 10, 7, 7, 7])
    assert np.allclose(s["nb_cnt_p"], [5, 6, 3, 6, 6])
    assert np.allclose(s["density_r12"], [1, 2, 1, 1, 1])
    assert np.allclose(s["density_r200"], [4, 4, 4, 4, 4])


def test_radius_converts_microns_to_pixels():
    """With 0.25 um per px, a 5 um radius is 20 px: cells 10 px apart are neighbours, 100 px apart are not."""
    cfg = Config(um_per_px_level0=0.125, img_level=1, nb_radius_um=5.0, density_radii_px=(12,))
    assert np.isclose(cfg.px_per_um, 4.0)
    s = count_summaries(_toy(), {"p": ["g1"]}, cfg)
    # radius 20 px: cell0 sees {0,1,2}, cell3 sees {3,4}
    assert np.allclose(s["nb_n"], [3, 3, 3, 2, 2])
    # a slide-specific pixel size overrides the config: 1 px per um -> radius 5 px -> every cell alone
    s2 = count_summaries(_toy(), {"p": ["g1"]}, cfg, px_per_um=1.0)
    assert np.allclose(s2["nb_n"], [1, 1, 1, 1, 1])


def test_attach_and_subsample():
    cfg = Config(um_per_px_level0=0.5, img_level=1, nb_radius_um=15)
    a = _toy()
    s = count_summaries(a, {"p": ["g1"]}, cfg)
    sub = subsample_cells(a, 3, seed=0)
    assert sub.n_obs == 3
    sub = attach_summaries(sub, s, a)
    for c in s.columns:
        assert c in sub.obs
    # values must match the full-slide rows for the same coordinates
    for cell in sub.obs_names:
        assert sub.obs.loc[cell, "nb_cnt_total"] == s.loc[cell, "nb_cnt_total"]
