"""Feature space: standardised PCA of the embeddings plus standardised morphometrics.

Fit inside each cross-validation fold on training cells only, so no information
from a held-out patient reaches the model through the scaler or the PCA basis.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..logging_utils import say


class FeatureSpace:
    """Scaler + PCA on embeddings, scaler on morphometrics, concatenated.

    ``fit`` uses at most ``fit_max`` random training cells (a full-cohort float32
    copy of 2048-d embeddings does not fit in Colab RAM); ``transform`` works in
    chunks so no full-size float32 intermediate is ever allocated.
    """

    def __init__(self, pca_dim: int, use_morpho: bool, fit_max: int = 100000, chunk: int = 20000):
        self.pca_dim = pca_dim
        self.use_morpho = use_morpho
        self.fit_max = fit_max
        self.chunk = chunk
        self.keep_emb_dims: Optional[np.ndarray] = None
        self.keep_mor_dims: Optional[np.ndarray] = None
        self.n_mor = 0
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def _emb(self, emb, rows) -> np.ndarray:
        e = np.asarray(emb[rows], dtype=np.float32)
        return e[:, self.keep_emb_dims] if self.keep_emb_dims is not None else e

    def _mor(self, mor, rows) -> np.ndarray:
        m = np.asarray(mor[rows], dtype=np.float32)
        return m[:, self.keep_mor_dims] if self.keep_mor_dims is not None else m

    def fit(self, emb, mor, idx=None, seed: int = 0, keep_emb_dims=None, keep_mor_dims=None) -> "FeatureSpace":
        self.keep_emb_dims = None if keep_emb_dims is None else np.asarray(keep_emb_dims, dtype=np.int64)
        self.keep_mor_dims = None if keep_mor_dims is None else np.asarray(keep_mor_dims, dtype=np.int64)
        idx = np.arange(len(emb)) if idx is None else np.asarray(idx)
        if len(idx) > self.fit_max:
            idx = np.random.default_rng(seed).choice(idx, self.fit_max, replace=False)
        idx = np.sort(idx)
        e = self._emb(emb, idx)
        n_comp = min(self.pca_dim, e.shape[0], e.shape[1])
        self.sc_e = StandardScaler().fit(e)
        self.pca = PCA(n_comp, random_state=seed).fit(self.sc_e.transform(e))
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        cum = np.cumsum(self.explained_variance_ratio_)
        n90 = int(np.searchsorted(cum, 0.90) + 1) if cum[-1] >= 0.90 else None
        n95 = int(np.searchsorted(cum, 0.95) + 1) if cum[-1] >= 0.95 else None
        say(f"feature space fit on {len(idx)} cells: {e.shape[1]} embedding dims -> {n_comp} PCs "
            f"explaining {100 * cum[-1]:.1f}% of variance (90% at {n90 or '>' + str(n_comp)} PCs, 95% at {n95 or '>' + str(n_comp)} PCs)")
        del e
        if self.use_morpho:
            m = self._mor(mor, idx)
            self.n_mor = m.shape[1]
            self.sc_m = StandardScaler().fit(m)
            say(f"morphometrics: {self.n_mor} standardised columns appended; total feature dim {self.dim}")
        else:
            self.n_mor = 0
        return self

    def transform(self, emb, mor) -> np.ndarray:
        out = np.empty((len(emb), self.dim), dtype=np.float32)
        for start in range(0, len(emb), self.chunk):
            rows = slice(start, start + self.chunk)
            z = self.pca.transform(self.sc_e.transform(self._emb(emb, rows))).astype(np.float32)
            if self.use_morpho:
                z = np.hstack([z, self.sc_m.transform(self._mor(mor, rows)).astype(np.float32)])
            out[start:start + len(z)] = z
        return out

    @property
    def dim(self) -> int:
        return int(self.pca.n_components_) + (self.n_mor if self.use_morpho else 0)
