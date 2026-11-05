import json

import numpy as np
import pandas as pd

from mmprog.config import Config
from mmprog.features import FeatureSelector, save_selection_report


def test_selector_drops_dead_and_duplicate_dims(tmp_path):
    cfg = Config(density_radii_px=(25, 50, 100), pca_dim=16)
    rng = np.random.default_rng(0)
    n = 2000
    emb = rng.normal(size=(n, 50)).astype(np.float16)
    emb[:, [3, 7]] = 0.0                                   # dead channels
    mor = rng.normal(size=(n, 8)).astype(np.float32)
    mor[:, 6] = mor[:, 5] * 2.0 + 0.001                    # density_r50 duplicates density_r25
    Y = np.stack([mor[:, 0] * 0.5 + rng.normal(size=n) * 0.1, rng.normal(size=n)], axis=1).astype(np.float32)
    obs = pd.DataFrame(dict(sample="s1", prog_a_cell_z=Y[:, 0], prog_b_cell_z=Y[:, 1]))
    sel = FeatureSelector(cfg).fit(emb, mor, Y, obs, train_idx=np.arange(1500))
    assert 3 not in sel.keep_emb_dims and 7 not in sel.keep_emb_dims
    assert len(sel.keep_emb_dims) == 48
    assert 5 in sel.keep_mor_dims and 6 not in sel.keep_mor_dims
    for key in ("embedding_dims_dropped", "morphometrics_kept", "morphometrics_dropped",
                "univariate_spearman", "pca_variance", "permutation_importance_top", "ridge_in_sample_r2"):
        assert key in sel.report
    # the planted driver of target a should rank first in permutation importance
    assert sel.report["permutation_importance_top"][0]["feature"] == "p_mean"
    path = tmp_path / "sel.json"
    save_selection_report(sel, str(path))
    loaded = json.load(open(path))
    assert loaded["keep_mor_dims"] == sel.keep_mor_dims.tolist()
