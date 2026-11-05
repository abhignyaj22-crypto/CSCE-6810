"""Batching for the unified model.

A batch is a fixed number of cells. For each cell the batcher gathers
- the centre feature vector (PCA of the encoder embedding plus standardised morphometrics,
  through the fold's ``FeatureSpace``; the oracle variant appends the lineage one-hot),
- the g*g grid tokens of its patch (raw encoder block embeddings, float16 on disk,
  float32 in the batch; the model projects them),
- the k neighbour feature vectors (same feature space applied to the neighbours' rows),
- the neighbour offsets in microns (dx, dy from the cell to each neighbour),
- the neighbours' measured targets with a mask for neighbours outside the training set
  (the smoothness term compares a cell's prediction with what its neighbours measured),
- the sample index (the unseen-slide slot at inference), the lineage label, the targets,
  and the expression embedding used by the alignment loss.

Shapes are static: the last batch is padded with repeated cells and a boolean pad mask
excludes them from every loss, so ``torch.compile(dynamic=False)`` never recompiles.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from ..logging_utils import say


class CellBatcher:
    """Materialises fixed-size batches of cells as CPU tensors (pinned when a GPU is present)."""

    def __init__(self, X_centre: np.ndarray, GRID: Any, NB: np.ndarray, NB_DIST: Any, XY_um: np.ndarray,
                 SID: np.ndarray, Y: Optional[np.ndarray], LIN: Optional[np.ndarray],
                 expression_embedding: Optional[np.ndarray], cell_idx: np.ndarray, batch_cells: int,
                 train_mask: Optional[np.ndarray] = None, shuffle: bool = False, seed: int = 0,
                 pin_memory: bool = False, name: str = "batcher"):
        self.X = np.asarray(X_centre, dtype=np.float32)
        self.GRID = GRID
        self.NB = np.asarray(NB, dtype=np.int64)
        self.XY_um = np.asarray(XY_um, dtype=np.float32)
        self.SID = np.asarray(SID, dtype=np.int64)
        self.Y = None if Y is None else np.asarray(Y, dtype=np.float32)
        self.LIN = None if LIN is None else np.asarray(LIN, dtype=np.int64)
        self.EXPR = None if expression_embedding is None else np.asarray(expression_embedding, dtype=np.float32)
        self.idx = np.asarray(cell_idx, dtype=np.int64)
        self.B = int(batch_cells)
        self.train_mask = train_mask if train_mask is not None else np.ones(len(self.X), dtype=bool)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.pin = pin_memory
        self.name = name
        self.n_batches = max(1, math.ceil(len(self.idx) / self.B))
        self._announced = False
        # Detached prediction cache for the smoothness term (prediction-to-prediction, D15): the training loop
        # writes each batch's predictions here after the step; neighbours are read from it at gather time. Rows that
        # have not been visited yet in the first epoch hold their own target as a warm start (training cells only).
        # The cache is a shared-memory torch tensor so DataLoader worker processes (persistent, forked) read the
        # parent's updates; a plain numpy array would be a private copy in each worker after fork.
        self.pred_cache = None if self.Y is None else torch.from_numpy(self.Y.copy()).share_memory_()
        self.cache_visited = torch.zeros(len(self.X), dtype=torch.bool).share_memory_()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_batches

    def _order(self) -> np.ndarray:
        order = self.idx.copy()
        if self.shuffle:
            self.rng.shuffle(order)
        # pad with repeated cells so the last batch has the same shape
        pad = self.n_batches * self.B - len(order)
        if pad > 0:
            order = np.concatenate([order, order[:pad]]) if len(order) >= pad else np.resize(order, self.n_batches * self.B)
        return order

    def gather(self, rows: np.ndarray, n_real: int) -> Dict[str, torch.Tensor]:
        k = self.NB.shape[1]
        nb_rows = self.NB[rows]                                              # (B, k)
        batch = {
            "idx": torch.from_numpy(rows),
            "pad_mask": torch.from_numpy(np.arange(len(rows)) < n_real),
            "centre": torch.from_numpy(self.X[rows]),
            "nb": torch.from_numpy(self.X[nb_rows.ravel()].reshape(len(rows), k, -1)),
            "nb_offsets_um": torch.from_numpy(self.XY_um[nb_rows] - self.XY_um[rows][:, None, :]),
            "nb_train_mask": torch.from_numpy(self.train_mask[nb_rows]),
            "sid": torch.from_numpy(self.SID[rows]),
        }
        if self.GRID is not None:
            batch["grid"] = torch.from_numpy(np.asarray(self.GRID[rows], dtype=np.float32))
        if self.Y is not None:
            batch["y"] = torch.from_numpy(self.Y[rows])
            # neighbour PREDICTIONS from the detached cache (prediction-to-prediction smoothness); the loop never
            # sees neighbour targets, so a permuted-target run cannot leak true targets through this term either
            batch["nb_pred"] = self.pred_cache[torch.from_numpy(nb_rows.ravel())].reshape(len(rows), k, -1).clone()
        if self.LIN is not None:
            batch["lin"] = torch.from_numpy(self.LIN[rows])
        if self.EXPR is not None:
            batch["expr"] = torch.from_numpy(self.EXPR[rows])
        # A forked worker cannot call into CUDA once the parent has used the GPU. Pinning
        # initialises CUDA, so workers leave the tensors pageable and the parent pins them.
        if self.pin and torch.utils.data.get_worker_info() is None:
            batch = {k_: v.pin_memory() for k_, v in batch.items()}
        if not self._announced:
            self._announced = True
            say(f"{self.name}: {len(self.idx)} cells in {self.n_batches} batches of {self.B}; "
                f"centre dim {self.X.shape[1]}, {k} neighbours, grid {'none' if self.GRID is None else tuple(batch['grid'].shape[1:])}, "
                f"targets {'none' if self.Y is None else self.Y.shape[1]}", indent=1)
        return batch

    def update_cache(self, rows: np.ndarray, mean: np.ndarray) -> None:
        """Write a batch's detached predictions into the cache (training cells only)."""
        if self.pred_cache is None:
            return
        rows = torch.from_numpy(np.asarray(rows))
        self.pred_cache[rows] = torch.from_numpy(np.asarray(mean, dtype=np.float32))
        self.cache_visited[rows] = True

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        order = self._order()
        n = len(self.idx)
        for b in range(self.n_batches):
            rows = order[b * self.B:(b + 1) * self.B]
            n_real = min(self.B, n - b * self.B)
            yield self.gather(rows, n_real)


class _BatchDataset(torch.utils.data.Dataset):
    """One item = one batch, so a DataLoader with workers prefetches whole batches."""

    def __init__(self, batcher: CellBatcher):
        self.batcher = batcher
        self.order = batcher._order()

    def __len__(self) -> int:
        return self.batcher.n_batches

    def __getitem__(self, b: int) -> Dict[str, torch.Tensor]:
        rows = self.order[b * self.batcher.B:(b + 1) * self.batcher.B]
        n_real = min(self.batcher.B, len(self.batcher.idx) - b * self.batcher.B)
        return self.batcher.gather(rows, n_real)


def make_loader(batcher: CellBatcher, cfg: Any, device: str) -> Iterator[Dict[str, torch.Tensor]]:
    """DataLoader over whole batches with the H100 settings from the configuration.

    ``prefetch_factor`` must be None when ``num_workers`` is 0 (PyTorch raises otherwise);
    pinned memory and persistent workers only pay off on a GPU, so they are switched off
    on CPU. The memmap-backed arrays are shared with workers through fork, not pickled.
    """
    workers = int(getattr(cfg, "num_workers", 0)) if device == "cuda" else 0
    # pin_memory on the DataLoader pins inside the worker, which raises
    # cudaErrorInitializationError after feature extraction has already used the GPU.
    kw: Dict[str, Any] = dict(batch_size=None, shuffle=False, num_workers=workers, pin_memory=False)
    if workers > 0:
        kw["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
        kw["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 2))
    return torch.utils.data.DataLoader(_BatchDataset(batcher), **kw)


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    if device == "cuda":
        batch = {k: v if v.is_pinned() else v.pin_memory() for k, v in batch.items()}
    non_blocking = device == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def xy_um(XY_norm: np.ndarray, obs_xy_px: np.ndarray, um_per_px: float) -> np.ndarray:
    """Cell coordinates in microns for the neighbour offsets (pixels at the working level times um/px)."""
    return np.asarray(obs_xy_px, dtype=np.float32) * np.float32(um_per_px)


def variant_flags(variant: str) -> Dict[str, bool]:
    """Translate a variant name into the switches the model and the loss read."""
    # The alignment term is off by default (DESIGN_V3.md D31: it did not help on the synthetic fixture and sits
    # at its chance value); the 'align' variant turns it on with cfg.lambda_align_on so its effect is measured.
    flags = dict(use_neighbours=True, use_grid=True, use_program_tokens=True, use_comp=True, use_smooth=True,
                 use_align=False)
    if variant == "align":
        flags["use_align"] = True
        flags["align_variant"] = True
    elif variant == "no_neighbours":
        flags["use_neighbours"] = False
    elif variant == "no_grid":
        flags["use_grid"] = False
    elif variant == "no_program_tokens":
        flags["use_program_tokens"] = False
    elif variant == "no_comp":
        flags["use_comp"] = False
    elif variant == "no_smooth":
        flags["use_smooth"] = False
    elif variant == "no_align":
        flags["use_align"] = False
    return flags


ALL_VARIANTS: List[str] = ["full", "no_neighbours", "no_grid", "no_program_tokens", "no_comp", "no_smooth", "no_align",
                           "align", "shuffled", "oracle", "anchor_only_targets"]
