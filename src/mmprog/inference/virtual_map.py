"""Morphology-only inference on a new slide.

Steps: detect nuclei with classical image processing, embed a patch around each
nucleus, compute morphometrics from the detected centroids, apply the fitted
selector and feature space, and predict with the reserved unseen-slide
embedding. Nothing here reads a transcript.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KDTree

from ..logging_utils import say, timed
from ..models.neighbours import norm_xy
from ..models.train import predict


class _SingleImageStore:
    """Minimal ImageStore stand-in for one in-memory slide."""

    def __init__(self, img: np.ndarray, sample: str = "NEW"):
        self._img = img
        self._sample = sample

    def get(self, sample: str) -> np.ndarray:
        return self._img

    def offset(self, sample: str) -> Tuple[int, int]:
        return (0, 0)


def virtual_program_map(
    img: np.ndarray, model: Any, fs: Any, selector: Any, cfg: Any, device: str, targets: List[str],
    n_samples: int, encoder: Optional[Any] = None,
    detect_fn: Optional[Callable] = None, embed_fn: Optional[Callable] = None, morpho_fn: Optional[Callable] = None,
    encoder_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Return a DataFrame with x, y, sample and one ``<target>_pred`` column per target.

    The ``*_fn`` arguments default to the pipeline implementations and exist so the
    function can be tested without an image encoder.
    """
    if detect_fn is None:
        from ..preprocessing.nuclei import detect_nuclei as detect_fn
    if embed_fn is None:
        from ..features.encoder import embed_cells as embed_fn
    if morpho_fn is None:
        from ..features.morphometrics import morphometrics as morpho_fn
    if encoder_fn is None:
        from ..features.encoder import get_encoder as encoder_fn

    with timed("nucleus detection"):
        xy = detect_fn(img)
    say(f"detected {len(xy)} nuclei on a {img.shape[1]}x{img.shape[0]} image")
    if len(xy) == 0:
        return pd.DataFrame(columns=["x", "y", "sample"] + [t.replace("_z", "_pred") for t in targets])

    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sample": "NEW"})
    store = _SingleImageStore(img)
    enc = encoder if encoder is not None else encoder_fn(cfg.encoder, device)

    tmp = tempfile.NamedTemporaryFile(suffix="_new_emb.npy", delete=False)
    tmp.close()
    try:
        with timed("embedding detected nuclei"):
            emb = embed_fn(obs, store, enc, tmp.name, cfg, device)
        mor = morpho_fn(obs, store, cfg)
        X = fs.transform(np.asarray(emb), mor)
        XY = norm_xy(obs)
        SID = np.full(len(obs), n_samples, dtype=np.int64)   # unseen slide: zero embedding
        pred = predict(model, X, XY, SID, device)
    finally:
        try:
            del emb
        except NameError:
            pass
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    cols = [t.replace("_z", "_pred") for t in targets]
    out = pd.concat([obs, pd.DataFrame(pred, columns=cols)], axis=1)
    say(f"virtual program map: {len(out)} nuclei x {len(cols)} programs; "
        + ", ".join(f"{c} mean {out[c].mean():.3f}" for c in cols[:4]))
    return out


def sanity_check_on_training_slide(
    vmap: pd.DataFrame, obs_slide: pd.DataFrame, offset: Tuple[int, int], main_target: str, cfg: Any,
    match_um: float = 8.0,
) -> float:
    """Pearson between predictions at detected nuclei and measured scores at the nearest annotated cell.

    Only nuclei within ``match_um`` microns of an annotated centroid are compared.
    """
    ox, oy = offset
    ref = obs_slide[["x", "y"]].values.astype(np.float32) - np.array([ox, oy], dtype=np.float32)
    tree = KDTree(ref)
    d, nn = tree.query(vmap[["x", "y"]].values.astype(np.float32), k=1)
    ok = d[:, 0] < match_um / cfg.um_per_px
    col = main_target.replace("_z", "_pred")
    if ok.sum() < 3:
        say(f"sanity check: only {int(ok.sum())} nuclei matched within {match_um} um; no correlation computed")
        return np.nan
    pred = vmap.loc[ok, col].values
    meas = obs_slide[main_target].values[nn[ok, 0]]
    r = float(stats.pearsonr(pred, meas)[0])
    say(f"sanity check: matched {int(ok.sum())}/{len(vmap)} detected nuclei to annotated cells; "
        f"pearson(pred, measured) = {r:.3f}")
    # Calibration: the detector and the segmentation disagree on how many nuclei there are, so the
    # density features (and therefore the predicted level) can shift even when the ranking is right.
    say(f"sanity check: detected {len(vmap)} nuclei vs {len(obs_slide)} segmented cells "
        f"({len(vmap) / max(len(obs_slide), 1):.2f}x); predicted mean {pred.mean():+.3f} vs measured mean {meas.mean():+.3f} "
        f"(a large offset means the density features shifted between training and inference)")
    return r
