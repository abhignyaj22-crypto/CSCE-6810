"""Loss terms.

    loss = MSE(program scores)
         + lambda_smooth * mean over valid neighbour pairs of ||y_hat_i - y_hat_j||^2
         + lambda_comp   * cross-entropy(lineage logits, lineage)

The Laplacian term asks neighbouring cells to receive similar predictions. It is
computed only over neighbours that are in the training set (``keep_mask``) so
held-out cells never influence training, even through smoothing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def program_loss(
    yhat_pos: torch.Tensor,
    y: torch.Tensor,
    yhat_nb: torch.Tensor,
    keep_mask: torch.Tensor,
    comp_logits: Optional[torch.Tensor],
    lin: torch.Tensor,
    lambda_smooth: float,
    lambda_comp: float,
    use_spatial: bool,
    use_comp: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine the three terms and return the total plus each part as a float.

    yhat_pos:    (B, T) predictions of the batch cells
    y:           (B, T) targets of the batch cells
    yhat_nb:     (B, k, T) predictions of each cell's k neighbours
    keep_mask:   (B, k) bool, True where the neighbour is a training cell
    comp_logits: (B, L) lineage logits or None
    lin:         (B,) int lineage labels
    """
    mse = F.mse_loss(yhat_pos, y)
    total = mse
    parts = {"mse": float(mse.item()), "smooth": 0.0, "comp": 0.0}

    if use_spatial and bool(keep_mask.any()):
        diff = (yhat_pos[:, None, :] - yhat_nb) ** 2       # (B, k, T)
        smooth = diff[keep_mask].mean()
        total = total + lambda_smooth * smooth
        parts["smooth"] = float(smooth.item())

    if use_comp and comp_logits is not None:
        comp = F.cross_entropy(comp_logits, lin)
        total = total + lambda_comp * comp
        parts["comp"] = float(comp.item())

    return total, parts
