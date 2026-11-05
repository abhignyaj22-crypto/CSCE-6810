"""Frozen-encoder feature extraction into two float16 memmaps: EMB and GRID.

EMB  (N, D)        pooled embedding of the 224-resized patch around each nucleus.
GRID (N, g*g, D)   the encoder's spatial feature map pooled to a g x g lattice over the
                   patch. One forward pass per cell gives both; running the encoder on
                   each block separately would cost g*g passes per cell and is not done.
                   For a ResNet-50 the map is 7 x 7, so g > 7 is clamped to 7 and the
                   reason is printed.

Data path (facts in docs/ARCHITECTURE_V3.md section 8): a Dataset returns the uint8 patch;
the DataLoader uses pinned memory, persistent workers and prefetching when workers > 0
(``prefetch_factor`` must be None when ``num_workers == 0``); the batch is moved with
``non_blocking=True``; the encoder runs under bf16 autocast on CUDA; the resize and
normalisation happen on the device. Outputs are written to ``<name>.part`` and renamed at
the end so an interrupted run never leaves a complete-looking file.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..logging_utils import say, timed, warn
from .patches import extract_patch


class _PatchDataset:
    """Index -> uint8 (p, p, 3) patch; one image per sample is kept by the ImageStore."""

    def __init__(self, obs: pd.DataFrame, images, patch_px: int, order: np.ndarray):
        self.obs = obs
        self.images = images
        self.patch_px = patch_px
        self.order = order                              # cell row indices in sample-contiguous order
        self.xs = obs["x"].to_numpy()
        self.ys = obs["y"].to_numpy()
        self.samples = obs["sample"].to_numpy()

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int):
        row = int(self.order[i])
        s = self.samples[row]
        img = self.images.get(s)
        ox, oy = self.images.offset(s)
        return row, extract_patch(img, self.xs[row] - ox, self.ys[row] - oy, self.patch_px)


def feature_cache_key(obs: pd.DataFrame, cfg, encoder_name: str) -> str:
    cells = obs.index.astype(str).to_numpy() if obs.index.dtype != np.int64 else obs.get("cell_id", obs.index).astype(str).to_numpy()
    h = hashlib.md5("|".join(cells[:: max(1, len(cells) // 5000)]).encode()).hexdigest()[:8]
    return f"{encoder_name}_p{cfg.patch_px}_g{cfg.grid}_n{len(obs)}_{h}"


def _normalise(x_uint8, card: Dict[str, Any], device: str):
    import torch
    import torch.nn.functional as F
    x = x_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=card["input_px"], mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(card["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(card["std"], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _store_dir(out_dir: str, cfg) -> str:
    """Directory that actually holds the EMB and GRID bytes.

    ``memmap_dir`` is for a filesystem with room (a tmpfs, for example). The component
    directory still receives a symlink, so the artefact names stay where the loader looks.
    """
    raw = getattr(cfg, "memmap_dir", "") or ""
    if not raw:
        return out_dir
    os.makedirs(raw, exist_ok=True)
    say(f"EMB and GRID bytes go to {raw}; the component directory keeps symlinks")
    return raw


def _require_space(path: str, nbytes: int) -> None:
    free = os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
    if free < nbytes:
        raise OSError(f"EMB+GRID need {nbytes / 1e9:.1f} GB but {path} has {free / 1e9:.1f} GB free")


def _publish(stored: str, out_dir: str, name: str) -> None:
    dest = os.path.join(out_dir, name)
    if os.path.abspath(stored) == os.path.abspath(dest):
        return
    if os.path.lexists(dest):
        os.remove(dest)
    os.symlink(stored, dest)


def extract_features(obs: pd.DataFrame, images, enc, card: Dict[str, Any], cfg, device: str,
                     out_dir: str) -> Tuple[np.memmap, np.memmap]:
    """Run the frozen encoder over every cell and return the (EMB, GRID) memmaps."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    n = len(obs)
    d = int(card["dim"])
    g = int(cfg.grid)
    # probe the feature-map size once so the grid can be clamped before any file is created
    with torch.no_grad():
        probe = torch.zeros(1, 3, card["input_px"], card["input_px"], device=device)
        fmap = enc.forward_features(probe)
    native_g = int(fmap.shape[-1])
    d_grid = int(fmap.shape[1])
    if g > native_g:
        warn(f"grid {g} exceeds the encoder feature map ({native_g} x {native_g}); using grid {native_g}")
        g = native_g
    say(f"feature extraction: {n} cells, encoder {card['name']} dim {d}, feature map {d_grid} x {native_g} x {native_g} "
        f"pooled to {g} x {g} grid tokens")

    store = _store_dir(out_dir, cfg)
    emb_path = os.path.join(store, "emb.f16.npy")
    grid_path = os.path.join(store, "grid.f16.npy")
    need = (n * d + n * g * g * d_grid) * np.dtype(np.float16).itemsize
    _require_space(store, need)
    emb = np.lib.format.open_memmap(emb_path + ".part", mode="w+", dtype=np.float16, shape=(n, d))
    grid = np.lib.format.open_memmap(grid_path + ".part", mode="w+", dtype=np.float16, shape=(n, g * g, d_grid))
    say(f"memmaps: EMB {emb.nbytes / 1e6:.0f} MB, GRID {grid.nbytes / 1e6:.0f} MB under {store}")

    # sample-contiguous order so the ImageStore holds one image at a time
    order = np.argsort(obs["sample"].to_numpy(), kind="stable")
    ds = _PatchDataset(obs, images, cfg.patch_px, order)
    workers = int(cfg.num_workers) if device == "cuda" else 0        # CPU tests: no worker processes
    loader_kwargs = dict(batch_size=int(cfg.emb_batch), shuffle=False, num_workers=workers,
                         pin_memory=bool(cfg.pin_memory) and device == "cuda")
    if workers > 0:
        loader_kwargs.update(persistent_workers=bool(cfg.persistent_workers), prefetch_factor=int(cfg.prefetch_factor))
    loader = DataLoader(ds, **loader_kwargs)
    say(f"dataloader: batch {cfg.emb_batch}, workers {workers}, pin_memory {loader_kwargs['pin_memory']}, "
        f"prefetch_factor {loader_kwargs.get('prefetch_factor', None)}")

    use_amp = bool(cfg.amp_enabled) and device == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    current_sample, t_sample, n_sample = None, time.time(), 0
    t0 = time.time()
    with torch.inference_mode(), timed("encoder pass"):
        for rows, patches in loader:
            rows = rows.numpy()
            x = _normalise(patches, card, device)
            if card.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                fmap = enc.forward_features(x)                                     # (B, D_grid, h, w)
                pooled = enc(x)                                                   # (B, D)
            blocks = F.adaptive_avg_pool2d(fmap.float(), g).flatten(2).transpose(1, 2)   # (B, g*g, D_grid)
            emb[rows] = pooled.float().cpu().numpy().astype(np.float16)
            grid[rows] = blocks.cpu().numpy().astype(np.float16)
            s = obs["sample"].to_numpy()[rows[-1]]
            if s != current_sample:
                if current_sample is not None:
                    say(f"{current_sample}: {n_sample} cells, {n_sample / max(time.time() - t_sample, 1e-6):.0f} cells/s", indent=1)
                current_sample, t_sample, n_sample = s, time.time(), 0
            n_sample += len(rows)
        if current_sample is not None:
            say(f"{current_sample}: {n_sample} cells, {n_sample / max(time.time() - t_sample, 1e-6):.0f} cells/s", indent=1)
    emb.flush()
    grid.flush()
    del emb, grid
    os.replace(emb_path + ".part", emb_path)
    os.replace(grid_path + ".part", grid_path)
    _publish(emb_path, out_dir, "emb.f16.npy")
    _publish(grid_path, out_dir, "grid.f16.npy")
    say(f"features written: {n} cells in {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-6):.0f} cells/s overall)")
    return np.load(emb_path, mmap_mode="r"), np.load(grid_path, mmap_mode="r")


def check_embeddings(emb: np.ndarray, n_expected: int, label: str) -> None:
    """Refuse a cached embedding whose row count is wrong or that holds a block of all-zero rows."""
    if emb.shape[0] != n_expected:
        raise ValueError(f"{label}: cached rows {emb.shape[0]} != expected {n_expected}; delete the cache")
    step = max(1, emb.shape[0] // 20000)
    sample = np.asarray(emb[::step])
    zero_rows = float((np.abs(sample).sum(axis=tuple(range(1, sample.ndim))) == 0).mean())
    say(f"{label}: {emb.shape}, all-zero rows in a {len(sample)}-row sample: {zero_rows:.1%}")
    if zero_rows > 0.01:
        raise ValueError(f"{label}: {zero_rows:.1%} all-zero rows suggests an interrupted extraction; delete the cache")
