"""Frozen image encoder and chunked, memory-mapped cell embedding.

The encoder is never fine-tuned (textbook transfer learning: a fixed feature
extractor plus a small trained head). Embeddings are written straight into a
float16 memmap so a cohort of several hundred thousand cells never has to fit in
RAM.
"""
from __future__ import annotations

import hashlib
import os
import json
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import say
from .patches import extract_patch, preprocess_batch


def _tiny_test_encoder():
    """A three-layer conv net with ``num_features = 64``; for tests, never for results.

    Its weights are drawn from a fixed seed so that training and deployment, which construct
    the encoder separately, see the same network (pretrained encoders carry their own weights;
    the registry records a weight hash for provenance in both cases).
    """
    import torch
    import torch.nn as nn

    class TinyEncoder(nn.Module):
        num_features = 64

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )

        def forward(self, x):
            return self.net(x)

    state = torch.random.get_rng_state()
    torch.manual_seed(20260101)
    try:
        enc = TinyEncoder()
    finally:
        torch.random.set_rng_state(state)
    return enc


def get_encoder(name: str, device: str):
    """Return a frozen encoder in eval mode with a ``.num_features`` attribute."""
    import timm

    if name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    elif name == "resnet50_random":
        model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    elif name == "uni":
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    elif name == "tiny_test":
        model = _tiny_test_encoder()
    elif name == "conch":
        raise NotImplementedError("CONCH needs its own loader; see the MahmoodLab/CONCH README")
    else:
        raise ValueError(f"unknown encoder {name!r}")
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"encoder {name}: {n_params / 1e6:.1f} M parameters, {model.num_features} output features, frozen, on {device}")
    return model


def embed_cells(obs: pd.DataFrame, images, enc, out_path: str, cfg: Config, device: str,
                dtype=np.float16) -> np.memmap:
    """Embed a patch around every cell in ``obs`` into a memmap of shape ``(N, D)``.

    ``images`` provides ``get(sample)`` and ``offset(sample)``; the offset is the
    crop origin, subtracted from the cell coordinates before indexing.
    """
    import torch

    n_cells, n_feat = len(obs), enc.num_features
    say(f"embedding {n_cells} cells x {n_feat} features -> {out_path} ({dtype.__name__ if hasattr(dtype, '__name__') else dtype})")
    # Write to a .part file and rename at the end: open_memmap creates the full-size file up front,
    # so an interrupted run must not leave a complete-looking array of zeros under the final name.
    part_path = out_path + ".part"
    mm = np.lib.format.open_memmap(part_path, mode="w+", dtype=dtype, shape=(n_cells, n_feat))
    xs, ys = obs["x"].to_numpy(), obs["y"].to_numpy()
    samples = obs["sample"].to_numpy()
    use_autocast = str(device).startswith("cuda")
    for sample in pd.unique(samples):
        idx = np.where(samples == sample)[0]
        img = images.get(sample)
        ox, oy = images.offset(sample)
        t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(idx), cfg.emb_batch):
                batch_idx = idx[b:b + cfg.emb_batch]
                patches = np.stack([extract_patch(img, xs[i] - ox, ys[i] - oy, cfg.patch_px) for i in batch_idx])
                with torch.autocast(device_type="cuda", enabled=use_autocast):
                    feats = enc(preprocess_batch(patches, device))
                mm[batch_idx] = feats.float().cpu().numpy().astype(dtype)
        elapsed = max(time.time() - t0, 1e-6)
        say(f"{sample}: {len(idx)} cells in {elapsed:.1f}s ({len(idx) / elapsed:.0f} cells/s)", indent=1)
    mm.flush()
    del mm
    os.replace(part_path, out_path)
    mm = np.load(out_path, mmap_mode="r")
    say(f"embeddings written: shape {mm.shape}, {mm.nbytes / 1e6:.0f} MB on disk")
    return mm


def embedding_cache_key(manifest_rows: List[Dict[str, Any]], n_obs: int, cfg: Config) -> str:
    """Short hash of everything that changes the embedding values."""
    payload = json.dumps(dict(
        samples=[r["sample"] for r in manifest_rows], n_obs=int(n_obs), level=cfg.img_level,
        z=cfg.z_project, stain=cfg.stain_mode, patch=cfg.patch_px, encoder=cfg.encoder,
        # which morphology file was used (mip / zstack / focus) changes the pixels, so it changes the key
        image_kind=[r.get("image_kind") for r in manifest_rows],
    ), sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:8]
