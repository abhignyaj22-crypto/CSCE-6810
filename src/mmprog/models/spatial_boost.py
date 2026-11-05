"""Boosting with spatial regularization (Xiang, Xi, Hasson and Ramadge, NIPS 2009, Advances in Neural Information Processing Systems 22).

The algorithm, restated
    Instances x_i in R^n with labels y_i in {-1, +1}. A pool H of p decision stumps,
    each acting on one input component; s(j) is the component used by stump j.
    Every component k has a location v_k. Q is the n x p indicator with
    Q[k, j] = 1 when s(j) = k, so beta = Q alpha is the per-component importance.

    Regularised exponential loss
        L(alpha) = sum_i exp(-y_i sum_j alpha_j h_j(x_i)) + lambda * beta^T K beta
    with K = mu I - G, G_ab = exp(-||v_a - v_b||^2 / (2 r^2)), mu = max_b sum_a G_ab.
    K is positive semidefinite; the quadratic form rewards importance that is
    spread over nearby components and penalises isolated spikes.

    Round t:  beta = Q alpha;  gamma = -2 lambda K beta;
              j* = argmax_j  Omega(h_j, w) + gamma_{s(j)},  Omega = sum_i y_i h_j(x_i) w_i;
              step eps along h_{j*}  (greedy, clipped by the conservative bound);
              w_i <- w_i exp(-eps y_i h_{j*}(x_i)).
    gamma is a bonus for components that lie near already important ones; with
    lambda = 0 the procedure is AdaBoost with unnormalised weights.

Scaling convention used here
    Omega grows with the number of instances and K with the kernel constant mu, so
    the same lambda means different things on a few dozen face images and on 100,000
    cells. The selection score is therefore computed with Omega divided by a fixed
    scale (the initial total weight m for the exponential loss, so the edge lies in
    [-1, 1]; the initial absolute residual sum for the squared loss) and with K
    divided by its largest diagonal entry. For the Gaussian kernel G_aa = 1, so
    K_aa = mu - 1 and the normalised kernel is K / (mu - 1). lambda is then a
    dimensionless trade-off between edge and spatial bonus, fixed for the whole
    fit; the paper's unnormalised lambda equals lambda * scale / (mu - 1), and the
    paper's objective is recovered exactly with that lambda. The step size and the
    weight update are the paper's, written in those units.

Squared-loss extension
    The objective is (1/2) sum_i (y_i - F(x_i))^2 + lambda_paper beta^T K_paper beta.
    The negative gradient along stump h_j is Omega = sum_i r_i h_j(x_i) with r = y - F,
    and the exact minimiser of the objective along h_j is
    eps = (Omega + gamma_k scale) / (m + 2 lambda K_kk scale), using sum_i h_j^2 = m.
    The 1/2 convention is what makes Omega the gradient; it is stated here because
    the logged loss uses it too.

Use in this repository
    Components are the blocks of the nucleus patch (``features/patch_grid.py``),
    so the importance map beta says where in the 27 um neighbourhood the
    program signal lives. The classification task is "neighbourhood program
    score in the top quartile of its slide". A squared-loss variant is provided
    for the continuous target; it keeps the same selection bonus because the
    regulariser and its gradient are unchanged, and it is labelled as an
    extension of the published method, not part of it.

Reference
    Xiang Z. J., Xi Y. T., Hasson U., Ramadge P. J. (2009). Boosting with spatial
    regularization. Advances in Neural Information Processing Systems 22.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..logging_utils import say


@dataclass
class StumpPool:
    """All (component, threshold, sign) stumps evaluated once as an int8 matrix."""
    comp: np.ndarray          # (p,) component index s(j)
    thr: np.ndarray           # (p,) threshold
    sign: np.ndarray          # (p,) +1: h = +1 when x > thr; -1: the opposite
    H: np.ndarray             # (m, p) int8 stump outputs on the training set

    @property
    def p(self) -> int:
        return len(self.comp)


def build_stump_pool(X: np.ndarray, n_thresholds: int = 12) -> StumpPool:
    """Thresholds at evenly spaced quantiles of each component; both signs."""
    m, n = X.shape
    qs = np.linspace(0.05, 0.95, n_thresholds)
    comps, thrs, signs, cols = [], [], [], []
    for k in range(n):
        col = X[:, k]
        for t in np.unique(np.quantile(col, qs)):
            base = np.where(col > t, 1, -1).astype(np.int8)
            for s in (1, -1):
                comps.append(k)
                thrs.append(t)
                signs.append(s)
                cols.append(base * s)
    pool = StumpPool(np.asarray(comps), np.asarray(thrs, dtype=np.float32), np.asarray(signs, dtype=np.int8),
                     np.stack(cols, axis=1))
    say(f"stump pool: {pool.p} stumps over {n} components ({n_thresholds} quantile thresholds x 2 signs)")
    return pool


def spatial_kernel(locations: np.ndarray, radius: float) -> np.ndarray:
    """K = mu I - G with a Gaussian G of width ``radius``; mu = max column sum of G."""
    d2 = ((locations[:, None, :] - locations[None, :, :]) ** 2).sum(-1)
    G = np.exp(-d2 / (2.0 * radius ** 2))
    mu = G.sum(0).max()
    return (mu * np.eye(len(locations)) - G).astype(np.float64)


@dataclass
class SpatiallyRegularizedBoosting:
    """SRB (Algorithm 1 of the paper) with optional squared-loss extension.

    Parameters
        lam         regularisation strength lambda (0 recovers AdaBoost)
        radius      Gaussian kernel width in the units of ``locations`` (microns here)
        n_rounds    boosting rounds T
        loss        "exponential" (the paper, classification) or "squared" (extension, regression)
        n_thresholds stumps per component per sign
        step        "greedy" (paper's approximate minimiser, clipped by the conservative bound) or "fixed"
        fixed_eps   step size when step == "fixed"
    """
    lam: float = 0.1
    radius: float = 5.0
    n_rounds: int = 200
    loss: str = "exponential"
    n_thresholds: int = 12
    step: str = "greedy"
    fixed_eps: float = 0.1
    verbose_every: int = 25
    # fitted state
    alpha: Optional[np.ndarray] = None
    pool: Optional[StumpPool] = None
    K: Optional[np.ndarray] = None
    locations: Optional[np.ndarray] = None
    n_components: int = 0
    history: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray, locations: np.ndarray) -> "SpatiallyRegularizedBoosting":
        assert self.loss in ("exponential", "squared")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)        # float32 throughout: a float64 operand would upcast H every round
        m, n = X.shape
        self.n_components = n
        self.locations = np.asarray(locations, dtype=np.float32)
        assert len(self.locations) == n, "one location per component"
        self.pool = build_stump_pool(X, self.n_thresholds)
        self.K = spatial_kernel(self.locations, self.radius)
        self.K = self.K / np.diag(self.K).max()           # unit diagonal: lambda is dimensionless (see docstring)
        Kdiag = np.diag(self.K)
        H = self.pool.H.astype(np.float32)                 # (m, p)
        comp = self.pool.comp
        self.alpha = np.zeros(self.pool.p)
        self.history = []
        say(f"SRB fit: {m} instances, {n} components, lambda={self.lam}, radius={self.radius}, "
            f"loss={self.loss}, rounds={self.n_rounds}")

        if self.loss == "exponential":
            assert set(np.unique(y)) <= {-1.0, 1.0}, "exponential loss needs labels in {-1, +1}"
            w = np.ones(m, dtype=np.float32)                # unnormalised, as in the paper
            F = np.zeros(m, dtype=np.float32)
            scale = float(m)                                # frozen: lambda is relative to the initial loss
        else:
            F = np.zeros(m, dtype=np.float32)               # additive model F(x) = sum_j alpha_j h_j(x)
            scale = float(np.abs(y).sum()) + 1e-12          # frozen: initial absolute residual sum
        self.scale_ = scale
        kmax = float(np.diag(spatial_kernel(self.locations, self.radius)).max())
        self.lambda_paper_ = self.lam * scale / kmax
        say(f"SRB scale {scale:.4g}; lambda in the paper's units = {self.lambda_paper_:.4g} (K diagonal {kmax:.3g})")

        for t in range(self.n_rounds):
            beta = np.bincount(comp, weights=self.alpha, minlength=n)      # beta = Q alpha
            gamma = (-2.0 * self.lam * (self.K @ beta)).astype(np.float32)  # spatial compensation weight
            if self.loss == "exponential":
                omega = H.T @ (y * w).astype(np.float32)                    # Omega(h_j, w)
            else:
                resid = (y - F).astype(np.float32)
                omega = H.T @ resid                                         # correlation with the residual
            score = omega / scale + gamma[comp]                             # normalised edge plus spatial bonus
            j = int(np.argmax(score))
            k = comp[j]
            h = H[:, j]
            if self.loss == "exponential":
                correct = (y * h) > 0
                w_plus, w_minus = w[correct].sum(), w[~correct].sum()
                if self.step == "fixed":
                    eps = self.fixed_eps
                else:
                    greedy = (w_plus - w_minus + gamma[k] * scale) / (w_plus + w_minus + 2.0 * self.lam * Kdiag[k] * scale)
                    conservative = 3.0 * (w_plus - w_minus) / (w_plus + 1.36 * w_minus + 1e-12)
                    eps = float(min(conservative, greedy, 1.0))
                if eps <= 0:
                    say(f"round {t}: no stump has positive regularised edge; stopping", indent=1)
                    break
                self.alpha[j] += eps
                w = (w * np.exp(-eps * y * h)).astype(np.float32)
                F = (F + eps * h).astype(np.float32)
                beta_new = np.bincount(comp, weights=self.alpha, minlength=n)
                # the paper's objective in its own units: sum_i w_i + lambda_paper beta' K_paper beta
                loss_val = float(w.sum()) + float(self.lam * scale * beta_new @ self.K @ beta_new)
            else:
                # least-squares step along h with the regulariser's quadratic term; h^2 == 1 so sum h^2 = m
                eps = (omega[j] + gamma[k] * scale) / (m + 2.0 * self.lam * Kdiag[k] * scale)
                if abs(eps) < 1e-9:
                    say(f"round {t}: residual no longer correlates with any stump; stopping", indent=1)
                    break
                self.alpha[j] += eps
                F = (F + eps * h).astype(np.float32)
                beta_new = np.bincount(comp, weights=self.alpha, minlength=n)
                loss_val = 0.5 * float(((y - F) ** 2).sum()) + float(self.lam * scale * beta_new @ self.K @ beta_new)
            self.history.append(dict(round=t, stump=j, component=int(k), eps=float(eps), loss=float(loss_val),
                                     n_active_components=int((np.bincount(comp, weights=np.abs(self.alpha),
                                                                          minlength=n) > 0).sum())))
            if self.verbose_every and (t % self.verbose_every == 0 or t == self.n_rounds - 1):
                say(f"round {t:4d}: picked component {k} (loc {self.locations[k].round(1).tolist()} um), "
                    f"eps={eps:.4f}, loss={loss_val:.4g}, active components={self.history[-1]['n_active_components']}",
                    indent=1)
        return self

    # ------------------------------------------------------------------ prediction
    def _apply_pool(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        active = np.where(self.alpha != 0)[0]
        F = np.zeros(len(X))
        for j in active:
            h = np.where(X[:, self.pool.comp[j]] > self.pool.thr[j], 1.0, -1.0) * self.pool.sign[j]
            F += self.alpha[j] * h
        return F

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._apply_pool(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self._apply_pool(X)
        return np.sign(F) if self.loss == "exponential" else F

    # ------------------------------------------------------------------ interpretation
    def importance(self) -> np.ndarray:
        """beta = Q |alpha|: importance per component (absolute, so the squared-loss sign does not cancel)."""
        return np.bincount(self.pool.comp, weights=np.abs(self.alpha), minlength=self.n_components)

    def importance_map(self, grid: int) -> np.ndarray:
        """``(2, grid, grid)`` maps for the mean and std feature planes of ``features/patch_grid.py``."""
        beta = self.importance()
        return beta.reshape(2, grid, grid)

    def spatial_coherence(self) -> float:
        """Moran's I of the importance map over the component locations (adjacency = within ``radius``).

        Moran's I is the standard spatial autocorrelation statistic: about 0 when important
        components are scattered at random, positive (up to about 1) when importance is
        clustered in contiguous groups, which is what the regulariser is meant to produce.
        """
        beta = self.importance()
        if beta.sum() == 0:
            return float("nan")
        d2 = ((self.locations[:, None, :] - self.locations[None, :, :]) ** 2).sum(-1)
        W = ((d2 <= self.radius ** 2) & ~np.eye(len(beta), dtype=bool)).astype(np.float64)
        z = beta - beta.mean()
        denom = (z ** 2).sum()
        if denom == 0 or W.sum() == 0:
            return float("nan")
        return float(len(beta) / W.sum() * (z @ W @ z) / denom)
