"""Stain harmonisation between DAPI and H&E inputs.

The training cohort is DAPI (single channel, bright nuclei). An H&E slide is RGB
with dark nuclei. Mapping H&E to its haematoxylin optical-density channel gives
a single nuclear channel that lives in roughly the same space as DAPI, so one
model can be applied to both (Phase 4). In ``native`` mode nothing is changed.
"""
from __future__ import annotations

import numpy as np
from skimage import color

from ..logging_utils import say


def harmonise_stain(img: np.ndarray, stain_mode: str = "native") -> np.ndarray:
    """Return the image as uint8, optionally reduced to the haematoxylin channel."""
    if stain_mode != "hematoxylin" or img.ndim == 2:
        return img
    hed = color.rgb2hed(img)
    h = hed[..., 0]
    lo, hi = np.percentile(h, [1, 99.5])
    out = np.clip((h - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    say(f"stain harmonised: RGB {img.shape} -> haematoxylin channel {out.shape}, "
        f"OD range [{lo:.3f}, {hi:.3f}] stretched to uint8")
    return out
