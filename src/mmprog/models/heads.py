"""Loss functions and calibration helpers for the unified model.

gaussian_nll         heteroscedastic Gaussian negative log-likelihood per target; the
                     model predicts a mean and a log-variance (D30).
temperature_scale    post-hoc scaling of the predicted variance by a scalar fitted on the
                     inner validation patient (mean unchanged). temperature_scale and
                     fit_temperature here are the reference implementation (variance scale,
                     LBFGS); the training loop uses its own grid search on the standard
                     deviation (training/loop.py), which is what final_model.pt stores.
info_nce             symmetric InfoNCE between the pooled morphology representation and
                     the per-cell expression-side vector. This adapts the slide-level
                     contrastive alignment of TANGLE (Jaume et al., CVPR 2024) to the
                     cell level with the P-dimensional program-score vector as the
                     expression side; the adaptation is this repository's (D31).
laplacian_smoothness mean squared difference between a cell's prediction and its
                     neighbours' predictions (v2, D15).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None,
                 beta: float = 0.0) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, optionally beta-weighted.

    With ``beta > 0`` each element's NLL is multiplied by ``stop_gradient(sigma^2) ** beta``
    (the beta-NLL of Seitzer et al., ICLR 2022; venue not verified in docs/research). The
    plain NLL scales the gradient of the mean by 1 / sigma^2, so a head that inflates its
    variance early stops fitting the mean; beta = 0.5 restores a mean gradient comparable to
    the squared error while keeping the variance calibrated. beta = 0 is the plain NLL.
    """
    log_var = log_var.clamp(-10.0, 10.0)
    nll = 0.5 * (log_var + (y - mean) ** 2 / torch.exp(log_var))
    if beta > 0:
        nll = nll * torch.exp(log_var).detach() ** beta
    if mask is not None:
        nll = nll * mask
        return nll.sum() / mask.sum().clamp_min(1.0)
    return nll.mean()


def temperature_scale(log_var: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale the predicted variance by ``temperature`` (log-space shift); the mean is untouched."""
    return log_var + math.log(max(temperature, 1e-6))


def fit_temperature(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor, n_steps: int = 200) -> float:
    """One scalar temperature minimising the Gaussian NLL on held-out cells."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=n_steps)

    def closure():
        opt.zero_grad()
        loss = gaussian_nll(mean, log_var + log_t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach()))


def info_nce(morph_pooled: torch.Tensor, expr_vec: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric contrastive loss; positive pairs are the same cell's morphology and expression vectors."""
    if morph_pooled.shape[1] != expr_vec.shape[1]:
        raise ValueError("project both sides to the same dimension before info_nce")
    a = F.normalize(morph_pooled, dim=1)
    b = F.normalize(expr_vec, dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def laplacian_smoothness(mean: torch.Tensor, nb_mean: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Prediction-to-prediction Laplacian term (D15).

    ``mean`` (B, T) are the batch predictions, ``nb_mean`` (B, k, T) the predictions of the k
    spatial neighbours, ``mask`` (B, k) selects neighbours in the training set. In the v3 loop
    ``nb_mean`` is read from a detached cache that every training step refreshes with its own
    predictions, so the term costs no extra forward pass and never sees a neighbour's target
    (v2 computed the same quantity on the union of the batch and its neighbours).
    """
    d = (mean.unsqueeze(1) - nb_mean) ** 2
    if mask is not None:
        d = d * mask.unsqueeze(-1)
        return d.sum() / (mask.sum() * mean.shape[1]).clamp_min(1.0)
    return d.mean()
