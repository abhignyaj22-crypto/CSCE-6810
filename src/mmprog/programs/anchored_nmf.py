"""Anchored non-negative matrix factorisation: gene programs seeded by literature gene sets.

Model
    X (cells x genes, log-normalised, non-negative)  ~  W H
    W >= 0 (cells x P)  per-cell program usage,  H >= 0 (P x genes)  gene weights.

Seeding
    The first programs are seeded, one per literature gene set: their rows of H start
    with ``anchor_prior`` spread over the anchor genes and a small random value
    elsewhere, and an L1 penalty of ``l1_nonanchor`` acts on their non-anchor entries,
    so a non-anchor gene enters a seeded program only when the data support it. The
    remaining ``n_extra`` programs are unseeded and absorb structure the literature sets
    do not describe, as in consensus NMF (Kotliar et al., eLife 2019).

    This is a seeded factorisation in the spirit of Spectra (Kunes et al., Nature
    Biotechnology 2024), written here from scratch with a simpler penalty; it is not an
    implementation of Spectra and must not be described as one.

Optimisation
    Multiplicative updates (Lee and Seung, NIPS 2000, the standard NMF algorithm; venue not verified in docs/research)
    for the objective  0.5 * ||X - W H||_F^2 + l1_nonanchor * sum(H[seeded, non-anchor]),
    with the penalty entering the denominator of the H update. ``l1_nonanchor`` is
    dimensionless: it multiplies the mean of the gradient numerator (W^T X) so that the
    same value means the same relative shrinkage on 2,000 or 600,000 cells. Rows of H are
    rescaled to unit L1 after every update (and W columns inversely) so programs stay
    comparable.

Transform
    ``transform(X)`` solves for W with H fixed by the same multiplicative W update, which
    is a non-negative least-squares projection; the only expensive product, X H^T, is
    computed once, so scoring 600k cells costs one sparse matmul.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from ..logging_utils import say, section

EPS = 1e-9


def _as_csr(X) -> csr_matrix:
    return X.tocsr() if issparse(X) else csr_matrix(np.asarray(X, dtype=np.float32))


class AnchoredNMF:
    """Seeded NMF with an L1 prior on non-anchor genes of seeded programs."""

    def __init__(self, n_extra: int = 2, anchor_prior: float = 1.0, l1_nonanchor: float = 0.05,
                 max_iter: int = 200, fit_cells: int = 100_000, seed: int = 0, verbose_every: int = 20):
        self.n_extra = int(n_extra)
        self.anchor_prior = float(anchor_prior)
        self.l1_nonanchor = float(l1_nonanchor)
        self.max_iter = int(max_iter)
        self.fit_cells = int(fit_cells)
        self.seed = int(seed)
        self.verbose_every = int(verbose_every)
        # fitted state
        self.H_: Optional[pd.DataFrame] = None
        self.program_names_: List[str] = []
        self.anchors_: Dict[str, List[str]] = {}
        self.var_names_: Optional[pd.Index] = None
        self.history_: List[dict] = []

    # ------------------------------------------------------------------ helpers
    def _init_H(self, var_names: pd.Index, anchors: Dict[str, List[str]], rng: np.random.Generator):
        G = len(var_names)
        names = list(anchors) + [f"extra_{i}" for i in range(self.n_extra)]
        P = len(names)
        H = rng.uniform(0.0, 1e-3, size=(P, G)).astype(np.float64)
        penalty = np.zeros((P, G), dtype=np.float64)
        gene_pos = {g: i for i, g in enumerate(var_names)}
        for p, (name, genes) in enumerate(anchors.items()):
            idx = [gene_pos[g] for g in genes if g in gene_pos]
            if idx:
                H[p, idx] = self.anchor_prior / len(idx)
            penalty[p, :] = self.l1_nonanchor
            penalty[p, idx] = 0.0                      # anchors are not shrunk
        for p in range(len(anchors), P):
            H[p] = rng.uniform(0.0, 1.0, size=G)      # unseeded programs start random
        H /= H.sum(axis=1, keepdims=True) + EPS
        return H, penalty, names

    @staticmethod
    def _normalise_rows(W: np.ndarray, H: np.ndarray):
        scale = H.sum(axis=1) + EPS
        return W * scale[None, :], H / scale[:, None]

    # ------------------------------------------------------------------ fit
    def fit(self, X, var_names, anchors: Dict[str, List[str]]) -> "AnchoredNMF":
        section("Anchored NMF: learning gene programs")
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        X = _as_csr(X)
        var_names = pd.Index(var_names)
        assert X.shape[1] == len(var_names), "var_names must match the columns of X"
        if X.min() < 0:
            raise ValueError("NMF needs non-negative input; pass log-normalised counts")
        n = X.shape[0]
        idx = np.sort(rng.choice(n, self.fit_cells, replace=False)) if n > self.fit_cells else np.arange(n)
        Xf = X[idx].toarray().astype(np.float64)   # dense subsample; fit_cells caps its size
        say(f"fitting on {len(idx)}/{n} cells x {Xf.shape[1]} genes; seeded programs {list(anchors)}, "
            f"extra programs {self.n_extra}, anchor_prior {self.anchor_prior}, l1_nonanchor {self.l1_nonanchor}")
        H, penalty, names = self._init_H(var_names, anchors, rng)
        P = H.shape[0]
        W = rng.uniform(0.0, 1.0, size=(len(idx), P))
        self.history_ = []
        for it in range(self.max_iter):
            # W update
            W *= (Xf @ H.T) / (W @ (H @ H.T) + EPS)
            # H update with the L1 prior in the denominator, scaled to the gradient numerator
            numer = W.T @ Xf
            pen = penalty * float(numer.mean())
            H *= numer / ((W.T @ W) @ H + pen + EPS)
            if it % self.verbose_every == 0 or it == self.max_iter - 1:
                resid = Xf - W @ H
                obj = 0.5 * float((resid ** 2).sum()) + float((pen * H).sum())
                self.history_.append(dict(iter=it, objective=obj))
                say(f"iter {it:4d}: objective {obj:.4g}", indent=1)
        # rows are normalised once at the end: rescaling W and H inside the loop would change the L1 term and
        # break the monotone descent of the multiplicative updates (the scale of W absorbs the row sums of H)
        W, H = self._normalise_rows(W, H)
        objs = [h["objective"] for h in self.history_]
        if len(objs) >= 2:
            rises = sum(1 for a, b in zip(objs[:-1], objs[1:]) if b > a * (1 + 1e-9))
            say(f"objective {objs[0]:.4g} -> {objs[-1]:.4g} over {self.max_iter} iterations "
                f"({'monotone' if rises == 0 else f'{rises} logged rises; the penalty scale pen is re-estimated each iteration'})")
        self.H_ = pd.DataFrame(H, index=names, columns=var_names)
        self.program_names_ = names
        self.anchors_ = {k: [g for g in v if g in var_names] for k, v in anchors.items()}
        self.var_names_ = var_names
        say(f"fit done in {time.time() - t0:.1f}s")
        self.describe()
        return self

    def describe(self, top: int = 15, recall_k: int = 30) -> pd.DataFrame:
        rows = []
        for name in self.program_names_:
            w = self.H_.loc[name]
            top_genes = w.sort_values(ascending=False)
            anchors = set(self.anchors_.get(name, []))
            anchor_mass = float(w[list(anchors)].sum() / (w.sum() + EPS)) if anchors else np.nan
            recall = float(len(anchors & set(top_genes.index[:recall_k])) / len(anchors)) if anchors else np.nan
            rows.append(dict(program=name, n_anchors=len(anchors), anchor_weight_fraction=anchor_mass,
                             anchor_recall_top30=recall, top_genes=", ".join(top_genes.index[:top])))
            say(f"program {name}: anchor weight fraction {anchor_mass:.3f}, anchor recall@{recall_k} {recall if recall == recall else float('nan'):.2f}")
            say("top genes: " + rows[-1]["top_genes"], indent=1)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ transform
    def transform(self, X, var_names=None, n_iter: int = 30) -> np.ndarray:
        """Non-negative projection of cells onto the fixed programs; returns (cells, P)."""
        assert self.H_ is not None, "fit first"
        X = _as_csr(X)
        H = self.H_.to_numpy() if var_names is None else self.H_.reindex(columns=pd.Index(var_names), fill_value=0.0).to_numpy()
        assert X.shape[1] == H.shape[1]
        XHt = np.asarray(X @ H.T, dtype=np.float64)   # the one sparse product
        HHt = H @ H.T
        rng = np.random.default_rng(self.seed)
        W = rng.uniform(0.5, 1.0, size=(X.shape[0], H.shape[0]))
        for _ in range(n_iter):
            W *= XHt / (W @ HHt + EPS)
        return W.astype(np.float32)
