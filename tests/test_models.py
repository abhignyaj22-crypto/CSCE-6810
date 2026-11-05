"""Tests for mmprog.models: network shapes, neighbour tables, training loop."""
import numpy as np
import pandas as pd
import torch

from mmprog.config import Config
from mmprog.models import ProgramNet, knn_table, norm_xy, train_model, predict


def _cfg(**kw):
    base = dict(max_epochs=30, patience=4, batch_cells=256, hidden=32, dropout=0.0, lr=1e-2, pca_dim=8)
    base.update(kw)
    return Config(**base)


def test_program_net_shapes_with_and_without_blocks():
    for use_spatial in (True, False):
        for use_comp in (True, False):
            net = ProgramNet(d_in=10, n_targets=3, n_lin=4, n_samples=2, hidden=16,
                             use_spatial=use_spatial, use_comp=use_comp)
            x = torch.randn(7, 10)
            xy = torch.rand(7, 2) * 2 - 1
            sid = torch.randint(0, 3, (7,))          # index 2 = unseen slide
            y, comp = net(x, xy, sid)
            assert y.shape == (7, 3)
            if use_comp:
                assert comp.shape == (7, 4)
            else:
                assert comp is None
            net.describe()


def test_spatial_residual_starts_at_zero():
    """g is zero-initialised so the first forward equals the feature branch alone."""
    net = ProgramNet(d_in=5, n_targets=2, n_lin=3, n_samples=1, hidden=8, dropout=0.0)
    net.eval()
    x = torch.randn(4, 5)
    xy = torch.rand(4, 2)
    sid = torch.zeros(4, dtype=torch.long)
    y_full, _ = net(x, xy, sid)
    y_feat = net.head(net.f(x))
    assert torch.allclose(y_full, y_feat)


def test_knn_table_stays_within_sample():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "x": np.concatenate([rng.uniform(0, 100, 50), rng.uniform(0, 100, 50)]),
        "y": np.concatenate([rng.uniform(0, 100, 50), rng.uniform(0, 100, 50)]),
        "sample": ["A"] * 50 + ["B"] * 50,
    })
    nb = knn_table(obs, k=4)
    assert nb.shape == (100, 4)
    assert (nb[:50] < 50).all() and (nb[50:] >= 50).all()
    assert not (nb == np.arange(100)[:, None]).any()   # never self


def test_knn_table_pads_tiny_slides():
    obs = pd.DataFrame({"x": [0.0, 1.0, 5.0], "y": [0.0, 0.0, 0.0], "sample": ["A", "A", "B"]})
    nb = knn_table(obs, k=3)
    assert nb.shape == (3, 3)
    assert (nb[0] == 1).all() and (nb[1] == 0).all()
    assert (nb[2] == 2).all()


def test_norm_xy_range_per_slide():
    obs = pd.DataFrame({"x": [0, 10, 100, 300], "y": [5, 5, 50, 70], "sample": ["A", "A", "B", "B"]})
    xy = norm_xy(obs)
    assert xy.shape == (4, 2)
    assert xy.min() >= -1 - 1e-5 and xy.max() <= 1 + 1e-5
    assert np.isclose(xy[0, 0], -1) and np.isclose(xy[1, 0], 1)


def _linear_problem(n=600, d=6, t=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, t)).astype(np.float32)
    Y = (X @ W + 0.1 * rng.normal(size=(n, t))).astype(np.float32)
    samples = np.repeat(np.arange(3), n // 3)
    obs = pd.DataFrame({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n), "sample": samples.astype(str)})
    XY = norm_xy(obs)
    NB = knn_table(obs, 4)
    LIN = rng.integers(0, 3, n)
    return X, Y, XY, samples.astype(np.int64), LIN, NB


def test_train_model_learns_and_early_stops():
    X, Y, XY, SID, LIN, NB = _linear_problem()
    cfg = _cfg()
    tr = np.where(SID != 2)[0]
    va = np.where(SID == 2)[0]
    model, best, epochs, hist = train_model(X, Y, XY, SID, LIN, NB, tr, va, n_samples=3, n_lin=3,
                                            cfg=cfg, device="cpu", seed=1, verbose=True)
    assert list(hist.columns) == ["epoch", "train_mse", "train_smooth", "train_comp", "val_mse"]
    assert len(hist) == epochs
    assert best < hist["val_mse"].iloc[0]            # improved over the first epoch
    assert best < 0.5                                # well below the target variance (about 6)
    assert epochs <= cfg.max_epochs
    p = predict(model, X[va], XY[va], SID[va], "cpu", batch=100)
    assert p.shape == (len(va), 2)
    r = np.corrcoef(p[:, 0], Y[va, 0])[0, 1]
    assert r > 0.9


def test_train_model_ablations_run():
    X, Y, XY, SID, LIN, NB = _linear_problem(n=300)
    cfg = _cfg(max_epochs=3)
    tr = np.where(SID != 2)[0]
    va = np.where(SID == 2)[0]
    for sp, cp in ((False, True), (True, False), (False, False)):
        model, _, _, hist = train_model(X, Y, XY, SID, LIN, NB, tr, va, 3, 3, cfg, "cpu",
                                        use_spatial=sp, use_comp=cp, verbose=False)
        assert len(hist) == 3
        if not sp:
            assert (hist["train_smooth"] == 0).all()
        if not cp:
            assert (hist["train_comp"] == 0).all()
