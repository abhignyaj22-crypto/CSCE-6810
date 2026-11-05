"""Image perturbations for the robustness validator.

A perturbation is named by a token: ``gamma_<g>`` (intensity gamma), ``blur_<sigma px>``
(Gaussian blur), ``rot90`` (rotate the image a quarter turn, with the nucleus coordinates
rotated the same way), ``noise_<sd>`` (additive Gaussian noise on the [0, 1] scale) and
``scale_<f>`` (multiply intensities, clip). They stand in for scanner, exposure and
focus differences between sites; they are not a substitute for a multi-site study.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

KINDS = ("gamma", "blur", "rot90", "noise", "scale")


def parse_perturbation(token: str) -> Tuple[str, float]:
    """``"gamma_0.8"`` -> ``("gamma", 0.8)``; ``"rot90"`` -> ``("rot90", 90.0)``. Raises on an unknown token."""
    t = str(token).strip()
    if t == "rot90":
        return "rot90", 90.0
    if "_" not in t:
        raise ValueError(f"unknown perturbation {token!r}; expected one of gamma_<g>, blur_<sigma>, rot90, noise_<sd>, scale_<f>")
    kind, _, value = t.partition("_")
    if kind not in KINDS or kind == "rot90":
        raise ValueError(f"unknown perturbation kind {kind!r} in {token!r}; expected one of {KINDS}")
    try:
        v = float(value)
    except ValueError as e:
        raise ValueError(f"perturbation {token!r}: parameter {value!r} is not a number") from e
    if kind in ("gamma", "scale") and v <= 0:
        raise ValueError(f"perturbation {token!r}: parameter must be positive")
    if kind in ("blur", "noise") and v < 0:
        raise ValueError(f"perturbation {token!r}: parameter must not be negative")
    return kind, v


def _to_float(img: np.ndarray) -> Tuple[np.ndarray, float]:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 65535.0
    mx = float(np.nanmax(img)) if img.size else 1.0
    return img.astype(np.float32) / (mx if mx > 1.0 else 1.0), (mx if mx > 1.0 else 1.0)


def _from_float(f: np.ndarray, scale: float, dtype: np.dtype) -> np.ndarray:
    out = np.clip(f, 0.0, 1.0) * scale
    if np.issubdtype(dtype, np.integer):
        return np.rint(out).astype(dtype)
    return out.astype(dtype)


def apply_perturbation(img: np.ndarray, xy: np.ndarray, token: str, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return the perturbed image and the nucleus coordinates expressed in its frame."""
    kind, v = parse_perturbation(token)
    xy = np.asarray(xy, dtype=np.float32)
    if kind == "rot90":
        # numpy rotates counter-clockwise: old (row r, col c) lands at (row W-1-c, col r); x is the column
        rot = np.rot90(img, k=1, axes=(0, 1)).copy()
        w = img.shape[1]
        new_xy = np.column_stack([xy[:, 1], (w - 1) - xy[:, 0]]).astype(np.float32)
        return rot, new_xy
    f, scale = _to_float(img)
    if kind == "gamma":
        f = np.power(f, v)
    elif kind == "blur":
        f = gaussian_filter(f, sigma=(v, v) + ((0,) if f.ndim == 3 else ())) if v > 0 else f
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        f = f + rng.normal(0.0, v, size=f.shape).astype(np.float32)
    elif kind == "scale":
        f = f * v
    return _from_float(f, scale, img.dtype), xy


def describe(token: str) -> str:
    kind, v = parse_perturbation(token)
    return {"gamma": f"intensity gamma {v}", "blur": f"Gaussian blur sigma {v} px", "rot90": "rotation by 90 degrees",
            "noise": f"additive Gaussian noise sd {v} (0 to 1 scale)", "scale": f"intensity scaled by {v}"}[kind]
