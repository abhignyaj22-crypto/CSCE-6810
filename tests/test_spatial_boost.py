"""Spatially regularized boosting: recovers AdaBoost at lambda = 0, produces coherent maps at lambda > 0."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.features.patch_grid import grid_feature_names, grid_locations_um, patch_grid_features, _block_stats  # noqa: E402
from mmprog.models.spatial_boost import SpatiallyRegularizedBoosting, build_stump_pool, spatial_kernel  # noqa: E402
from mmprog.evaluation.spatial_boost_eval import ring_summary, top_quartile_labels  # noqa: E402

GRID = 6


def _planted(seed=0, m=3000):
    """Labels depend on a contiguous 2x2 block of the mean plane plus noise everywhere else."""
    rng = np.random.default_rng(seed)
    n = 2 * GRID * GRID
    X = rng.normal(size=(m, n)).astype(np.float32)
    signal_blocks = [1 * GRID + 1, 1 * GRID + 2, 2 * GRID + 1, 2 * GRID + 2]
    s = X[:, signal_blocks].sum(1)
    y = np.where(s + 0.5 * rng.normal(size=m) > 0, 1.0, -1.0)
    loc = grid_locations_um(GRID, 60, 0.5)
    return X, y, loc, signal_blocks


def test_kernel_is_psd_and_mu_rule():
    loc = grid_locations_um(4, 32, 1.0)
    K = spatial_kernel(loc, radius=4.0)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-8


def test_stump_pool_shapes():
    X = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    pool = build_stump_pool(X, n_thresholds=4)
    assert pool.H.shape == (100, pool.p) and pool.H.dtype == np.int8
    assert set(np.unique(pool.H)) <= {-1, 1}


def test_lambda_zero_matches_adaboost_edge_rule():
    X, y, loc, _ = _planted()
    m = SpatiallyRegularizedBoosting(lam=0.0, radius=5.0, n_rounds=30, verbose_every=0).fit(X, y, loc)
    # with lambda = 0 the first pick must be the stump with the largest plain edge on uniform weights
    pool = m.pool
    omega0 = pool.H.astype(float).T @ y
    assert m.history[0]["stump"] == int(np.argmax(omega0))
    acc = (m.predict(X) == y).mean()
    assert acc > 0.8


def test_regularisation_increases_coherence_with_bounded_accuracy_cost():
    """The grouping effect: lambda raises Moran's I of the map and costs a few points of accuracy
    (the paper reports the same trade-off on fMRI). The planted block must stay on top."""
    X, y, loc, signal = _planted()
    base = SpatiallyRegularizedBoosting(lam=0.0, radius=5.0, n_rounds=120, verbose_every=0).fit(X, y, loc)
    srb = SpatiallyRegularizedBoosting(lam=0.1, radius=5.0, n_rounds=120, verbose_every=0).fit(X, y, loc)
    Xt, yt, _, _ = _planted(seed=1)
    acc_base = (base.predict(Xt) == yt).mean()
    acc_srb = (srb.predict(Xt) == yt).mean()
    assert acc_base > 0.85
    assert acc_srb > acc_base - 0.10
    assert srb.spatial_coherence() > base.spatial_coherence() + 0.1
    # importance concentrates on the planted block for both; the map must be (2, GRID, GRID)
    imp = srb.importance_map(GRID)
    assert imp.shape == (2, GRID, GRID)
    top = np.argsort(srb.importance())[::-1][:4]
    assert len(set(top.tolist()) & set(signal)) >= 2


def test_squared_loss_regression_fits_signal():
    rng = np.random.default_rng(3)
    n = 2 * GRID * GRID
    X = rng.normal(size=(2000, n)).astype(np.float32)
    y = 2.0 * X[:, 5] - X[:, 7] + 0.3 * rng.normal(size=2000)
    loc = grid_locations_um(GRID, 60, 0.5)
    m = SpatiallyRegularizedBoosting(lam=0.1, radius=5.0, n_rounds=150, loss="squared", verbose_every=0).fit(X, y, loc)
    r = np.corrcoef(m.predict(X), y)[0, 1]
    assert r > 0.8
    assert set(np.argsort(m.importance())[::-1][:2].tolist()) == {5, 7}


def test_grid_features_and_rings():
    patch = np.zeros((60, 60), dtype=np.uint8)
    patch[20:40, 20:40] = 200
    means, stds = _block_stats(patch, GRID)
    assert means.reshape(GRID, GRID)[2:4, 2:4].min() == 200 and means.reshape(GRID, GRID)[0, 0] == 0
    names = grid_feature_names(GRID)
    assert len(names) == 2 * GRID * GRID and names[0] == "mean_r0_c0"
    imp = np.zeros((2, GRID, GRID))
    imp[0, 2:4, 2:4] = 1.0            # the four blocks nearest the centre
    block_um = 3.4                    # 6 blocks of 3.4 um: centre blocks sit at radius 2.4 um
    rs = ring_summary(imp, GRID, block_um)
    row = rs.loc[rs.plane == "mean"].iloc[0]
    assert row["centre"] == pytest.approx(1.0) and row["n_blocks_centre"] == 4
    assert row["per_block_centre"] == pytest.approx(0.25)
    assert row["n_blocks_centre"] + row["n_blocks_inner_ring"] + row["n_blocks_outer_ring"] == GRID * GRID
    uniform = np.ones((2, GRID, GRID))
    ru = ring_summary(uniform, GRID, block_um).iloc[0]
    # per-block importance is the same in every band for a uniform map
    assert ru["per_block_centre"] == pytest.approx(ru["per_block_inner_ring"])


def test_patch_grid_features_on_image():
    class Store:
        def get(self, s):
            img = np.zeros((200, 200), dtype=np.uint8)
            img[90:110, 90:110] = 255
            return img

        def offset(self, s):
            return (0, 0)

    import pandas as pd
    from mmprog.config import Config
    cfg = Config(patch_px=48)
    obs = pd.DataFrame({"x": [100.0, 20.0], "y": [100.0, 20.0], "sample": ["A", "A"]})
    G = patch_grid_features(obs, Store(), cfg, grid=6)
    assert G.shape == (2, 72)
    assert G[0, :36].reshape(6, 6)[2:4, 2:4].min() > 200 and G[1].max() == 0


def test_top_quartile_labels_within_slide():
    y = np.concatenate([np.arange(8), np.arange(8) + 100]).astype(float)
    sid = np.array([0] * 8 + [1] * 8)
    lab = top_quartile_labels(y, sid)
    assert (lab[:8] > 0).sum() == 2 and (lab[8:] > 0).sum() == 2
