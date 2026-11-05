"""Figures written to the run directory.

All figures are produced with matplotlib's Agg backend so they work in a
headless job, and each function prints the path it wrote.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..logging_utils import say  # noqa: E402


def overlay_figure(obs: pd.DataFrame, pred_full: np.ndarray, main_idx: int, main_target: str, images,
                   path: str, n_samples: int = 4) -> str:
    """Tier 1: morphology, measured program, held-out predicted program, for the first slides."""
    samples = list(obs["sample"].unique())[:n_samples]
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3.6 * len(samples)), squeeze=False)
    for k, s in enumerate(samples):
        d = obs[obs["sample"] == s]
        img = images.get(s)
        ox, oy = images.offset(s)
        step = max(1, img.shape[0] // 2000)
        ax = axes[k]
        ax[0].imshow(img[::step, ::step], cmap="gray", extent=(0, img.shape[1], img.shape[0], 0))
        ax[0].set_title(f"{s} morphology")
        ax[1].scatter(d.x - ox, d.y - oy, c=d[main_target], s=1, cmap="magma", vmin=-2, vmax=2)
        ax[1].set_title("measured")
        ax[2].scatter(d.x - ox, d.y - oy, c=pred_full[d.index, main_idx], s=1, cmap="magma", vmin=-2, vmax=2)
        ax[2].set_title("predicted (held-out)")
        for a in ax[1:]:
            a.invert_yaxis()
            a.set_aspect("equal")
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(main_target)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    say(f"overlay figure for {len(samples)} slides -> {path}")
    return path


def results_figure(res: pd.DataFrame, path: str, ci_col: str = "full_95ci") -> str:
    """Grouped bars: per-patient median Pearson of every model variant for every target."""
    models = [c for c in res.columns if c != ci_col]
    x = np.arange(len(res.index))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(res.index) + 4, 4.5))
    for i, m in enumerate(models):
        ax.bar(x + i * width, res[m].values.astype(float), width, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(res.index, rotation=30, ha="right")
    ax.set_ylabel("per-patient median Pearson r")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    say(f"results figure -> {path}")
    return path


def reliability_figure(rel_summary: pd.DataFrame, res: Optional[pd.DataFrame], path: str) -> str:
    """Noise ceiling (sqrt of split-half reliability) next to the achieved correlation of the full model."""
    fig, ax = plt.subplots(figsize=(1.2 * len(rel_summary.index) + 4, 4))
    x = np.arange(len(rel_summary.index))
    ax.bar(x - 0.2, rel_summary["ceiling_sqrt"].values, 0.4, label="attainable ceiling sqrt(r_sb)")
    if res is not None and "full" in res.columns:
        achieved = res.reindex(rel_summary.index)["full"].values.astype(float)
        ax.bar(x + 0.2, achieved, 0.4, label="full model (held-out)")
    ax.set_xticks(x)
    ax.set_xticklabels(rel_summary.index, rotation=30, ha="right")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    say(f"reliability figure -> {path}")
    return path


def training_curves(histories: Dict[str, pd.DataFrame], path: str) -> str:
    """Validation MSE per epoch for each fold of the full model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in histories.items():
        if "val_mse" in h:
            ax.plot(h["epoch"], h["val_mse"], label=str(name), lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("inner-validation MSE")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    say(f"training curves -> {path}")
    return path


def importance_map_figure(maps: Dict[str, np.ndarray], grid: int, block_um: float, path: str) -> str:
    """Importance maps of spatially regularized boosting, one column per lambda, rows = mean and std planes."""
    keys = list(maps.keys())
    fig, axes = plt.subplots(2, len(keys), figsize=(2.6 * len(keys) + 1, 5.4), squeeze=False)
    extent = (-grid * block_um / 2, grid * block_um / 2, grid * block_um / 2, -grid * block_um / 2)
    for c, k in enumerate(keys):
        m = maps[k]
        vmax = m.max() + 1e-12
        for r, plane in enumerate(("block mean", "block std")):
            ax = axes[r, c]
            im = ax.imshow(m[r], cmap="viridis", vmin=0, vmax=vmax, extent=extent)
            ax.set_title(f"{k}\n{plane}" if r == 0 else plane, fontsize=8)
            ax.set_xlabel("um from nucleus centre", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.axhline(0, color="w", lw=0.3)
            ax.axvline(0, color="w", lw=0.3)
        fig.colorbar(im, ax=axes[:, c].tolist(), shrink=0.6)
    fig.suptitle("Where in the patch the program signal lives (importance beta = Q alpha)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    say(f"importance map figure -> {path}")
    return path
