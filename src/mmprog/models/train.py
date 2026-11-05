"""Training loop with neighbourhood-union minibatching and early stopping.

Why minibatch over neighbourhoods: the Laplacian term needs the prediction of
each cell's neighbours in the same forward pass. A batch is therefore the batch
cells plus the union of their training-set neighbours. Memory stays flat as the
cohort grows, and the smoothness term stays exact.

Why early stopping on a held-out patient: the v3 prototype showed that more
steps hurt held-out performance, so the inner validation patient decides when
to stop and which weights to keep.
"""
from __future__ import annotations

import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..logging_utils import say
from ..resources import seed_everything
from .losses import program_loss
from .program_net import ProgramNet


def _build_model(d_in: int, n_targets: int, n_lin: int, n_samples: int, cfg: Any,
                 use_spatial: bool, use_comp: bool) -> ProgramNet:
    return ProgramNet(
        d_in, n_targets, n_lin, n_samples,
        hidden=cfg.hidden, dropout=cfg.dropout,
        use_spatial=use_spatial, use_comp=use_comp,
        n_fourier=cfg.n_fourier, fourier_scale=cfg.fourier_scale, slide_emb_dim=cfg.slide_emb_dim,
    )


def train_model(
    X: np.ndarray, Y: np.ndarray, XY: np.ndarray, SID: np.ndarray, LIN: np.ndarray, NB: np.ndarray,
    tr_idx: np.ndarray, va_idx: np.ndarray, n_samples: int, n_lin: int, cfg: Any, device: str,
    use_spatial: bool = True, use_comp: bool = True, seed: int = 0, verbose: bool = True,
) -> Tuple[ProgramNet, float, int, pd.DataFrame]:
    """Train a ProgramNet on ``tr_idx`` cells, early-stopping on ``va_idx`` cells.

    Returns (model with the best weights loaded, best validation MSE, epochs run, history).
    """
    seed_everything(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    Yt = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=device)
    XYt = torch.as_tensor(np.asarray(XY, dtype=np.float32), device=device)
    St = torch.as_tensor(np.asarray(SID, dtype=np.int64), device=device)
    Lt = torch.as_tensor(np.asarray(LIN, dtype=np.int64), device=device)
    NBt = torch.as_tensor(np.asarray(NB, dtype=np.int64), device=device)

    model = _build_model(X.shape[1], Y.shape[1], n_lin, n_samples, cfg, use_spatial, use_comp).to(device)
    if verbose:
        model.describe()
        say(f"training on {len(tr_idx)} cells, validating on {len(va_idx)} cells, seed {seed}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    in_train = torch.zeros(len(X), dtype=torch.bool, device=device)
    in_train[torch.as_tensor(tr_idx, device=device)] = True

    best_val, best_state, bad_epochs = np.inf, None, 0
    history = []
    epochs_run = 0
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        epochs_run = ep + 1
        model.train()
        perm = np.random.permutation(tr_idx)
        sums = {"mse": 0.0, "smooth": 0.0, "comp": 0.0}
        n_batches = 0
        for b in range(0, len(perm), cfg.batch_cells):
            bi = torch.as_tensor(perm[b: b + cfg.batch_cells], device=device)
            nb = NBt[bi]                                   # (B, k)
            keep = in_train[nb]                            # neighbours outside the training set are masked
            # forward the union of batch cells and their kept neighbours in one pass
            union = torch.unique(torch.cat([bi, nb[keep]]))
            pos = torch.searchsorted(union, bi)
            # masked neighbours are absent from the union; searchsorted may point past the end, so clamp.
            # Their rows are never read because keep is False there.
            npos = torch.searchsorted(union, nb).clamp_(max=len(union) - 1)
            yhat, comp_logits = model(Xt[union], XYt[union], St[union])
            loss, parts = program_loss(
                yhat[pos], Yt[bi], yhat[npos], keep,
                None if comp_logits is None else comp_logits[pos], Lt[bi],
                cfg.lambda_smooth, cfg.lambda_comp, use_spatial, use_comp,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in sums:
                sums[k] += parts[k]
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vi = torch.as_tensor(va_idx, device=device)
            yv, _ = model(Xt[vi], XYt[vi], St[vi])
            val_mse = float(F.mse_loss(yv, Yt[vi]).item())
        row = {k: sums[k] / max(n_batches, 1) for k in sums}
        history.append(dict(epoch=ep, train_mse=row["mse"], train_smooth=row["smooth"],
                            train_comp=row["comp"], val_mse=val_mse))
        if verbose:
            say(f"epoch {ep:02d} | train mse {row['mse']:.4f} smooth {row['smooth']:.4f} "
                f"comp {row['comp']:.4f} | val mse {val_mse:.4f}", indent=1)

        if val_mse < best_val - 1e-4:
            best_val, bad_epochs = val_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    say(f"early stop at epoch {ep} (no improvement for {cfg.patience} epochs)", indent=1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if verbose:
        say(f"best val mse {best_val:.4f} after {epochs_run} epochs in {time.time() - t0:.1f}s")
    return model, best_val, epochs_run, pd.DataFrame(history)


@torch.no_grad()
def predict(model: ProgramNet, X: np.ndarray, XY: np.ndarray, SID: np.ndarray,
            device: str, batch: int = 8192) -> np.ndarray:
    """Predict program scores for every row, in batches."""
    model.eval()
    out = []
    for b in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[b: b + batch], dtype=np.float32), device=device)
        xyb = torch.as_tensor(np.asarray(XY[b: b + batch], dtype=np.float32), device=device)
        sb = torch.as_tensor(np.asarray(SID[b: b + batch], dtype=np.int64), device=device)
        y, _ = model(xb, xyb, sb)
        out.append(y.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.n_targets), dtype=np.float32)
