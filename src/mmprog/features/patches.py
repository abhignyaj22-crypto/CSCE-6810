"""Patch extraction around cell centroids and batch preprocessing for the encoder."""
from __future__ import annotations

import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_patch(img: np.ndarray, x: float, y: float, p: int) -> np.ndarray:
    """Return a ``(p, p, 3)`` uint8 patch centred on ``(x, y)``, zero-padded at the image border.

    A single-channel image (DAPI, haematoxylin) is repeated to three channels so
    that an ImageNet-pretrained encoder can consume it unchanged.
    """
    h, w = img.shape[:2]
    r = p // 2
    x0, y0 = int(round(x)) - r, int(round(y)) - r
    pad_shape = (p, p) if img.ndim == 2 else (p, p, img.shape[2])
    pad = np.zeros(pad_shape, dtype=np.uint8)
    xs = slice(max(x0, 0), min(x0 + p, w))
    ys = slice(max(y0, 0), min(y0 + p, h))
    sub = img[ys, xs]
    if sub.size:
        pad[(ys.start - y0):(ys.start - y0) + sub.shape[0], (xs.start - x0):(xs.start - x0) + sub.shape[1]] = sub
    if pad.ndim == 2:
        return np.repeat(pad[..., None], 3, axis=-1)
    return pad


def preprocess_batch(patches_uint8: np.ndarray, device: str):
    """``(B, p, p, 3)`` uint8 -> ImageNet-normalised ``(B, 3, 224, 224)`` tensor on ``device``.

    Resizing happens on the device so that hundreds of thousands of patches do not
    pay the per-patch PIL overhead.
    """
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(patches_uint8)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std
