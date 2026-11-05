"""Fit the deployable model on all patients (early-stopped on one) and save it.

The checkpoint holds the network weights, the fitted feature space (scaler and
PCA), the fitted selector, the target names, the lineage names and the config,
so a slide can be scored later from the checkpoint alone.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from ..logging_utils import say, section
from ..models.program_net import ProgramNet
from ..models.train import train_model
from ..evaluation.cross_validation import CVData, build_features


def fit_final_model(
    data: CVData, EMB: np.ndarray, MOR: np.ndarray, cfg: Any, device: str, run: Any,
    feature_space_factory: Optional[Callable] = None, selector_factory: Optional[Callable] = None,
) -> Tuple[ProgramNet, Any, Any]:
    """Train on every patient but the last (used for early stopping) and save ``final_model.pt``."""
    section("S7 final fit on the full cohort")
    pats = np.unique(data.GROUPS)
    va_pat = pats[-1]
    va = np.where(data.GROUPS == va_pat)[0]
    tr = np.where(data.GROUPS != va_pat)[0]
    say(f"final fit: {len(pats) - 1} training patients ({len(tr)} cells), early-stop on {va_pat} ({len(va)} cells)")

    X, fs, selector = build_features(EMB, MOR, data, tr, cfg, cfg.seed, bool(cfg.use_morpho),
                                     feature_space_factory, selector_factory)
    model, val_mse, epochs, history = train_model(
        X, data.Y, data.XY, data.SID, data.LIN, data.NB, tr, va, data.n_samples, len(data.lin_names),
        cfg, device, seed=cfg.seed, verbose=True,
    )
    ckpt = dict(state=model.state_dict(), fs=fs, selector=selector, targets=data.targets,
                lineages=data.lin_names, n_samples=data.n_samples, d_in=X.shape[1],
                cfg=cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg)))
    torch.save(ckpt, run.art("final_model"))
    history.to_csv(run.art("final_fit_history.csv"), index=False)
    say(f"final model saved: val mse {val_mse:.4f}, {epochs} epochs, d_in {X.shape[1]}")
    return model, fs, selector


def load_final_model(run: Any, cfg: Any, device: str) -> Tuple[ProgramNet, Any, Any, dict]:
    """Rebuild the network from ``final_model.pt``; returns (model, feature_space, selector, checkpoint)."""
    ckpt = torch.load(run.art("final_model"), map_location=device, weights_only=False)
    # The architecture must match the weights on disk, so it is rebuilt from the checkpoint's own
    # config; the current cfg may have been overridden on the command line since the model was trained.
    saved = ckpt.get("cfg", {})
    arch = {}
    for key in ("hidden", "dropout", "n_fourier", "fourier_scale", "slide_emb_dim"):
        arch[key] = saved.get(key, getattr(cfg, key))
        if key in saved and saved[key] != getattr(cfg, key):
            say(f"final model: using checkpoint {key}={saved[key]} (current config says {getattr(cfg, key)})")
    model = ProgramNet(
        ckpt["d_in"], len(ckpt["targets"]), len(ckpt["lineages"]), ckpt["n_samples"],
        hidden=arch["hidden"], dropout=arch["dropout"], n_fourier=arch["n_fourier"],
        fourier_scale=arch["fourier_scale"], slide_emb_dim=arch["slide_emb_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    say(f"final model loaded: {len(ckpt['targets'])} targets, d_in {ckpt['d_in']}, {ckpt['n_samples']} training slides")
    return model, ckpt["fs"], ckpt.get("selector"), ckpt
