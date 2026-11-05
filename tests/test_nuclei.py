"""Synthetic discs are detected with both polarities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from sklearn.neighbors import KDTree

from mmprog.preprocessing.nuclei import detect_nuclei


def _discs(rng, n=60, size=512, r=5, light_background=True):
    base = 200.0 if light_background else 20.0
    img = rng.normal(base, 5, size=(size, size)).astype(np.float32)
    pts = rng.uniform(20, size - 20, size=(n, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy in pts:
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[disc] += -120 if light_background else 150
    return np.clip(img, 0, 255).astype(np.uint8), pts


def _recall(found, truth, tol=4.0):
    if len(found) == 0:
        return 0.0
    d, _ = KDTree(found).query(truth, k=1)
    return float((d[:, 0] <= tol).mean())


def test_dark_nuclei_light_background():
    rng = np.random.default_rng(0)
    img, pts = _discs(rng, light_background=True)
    found = detect_nuclei(img, tile=256, overlap=32)
    assert found.shape[1] == 2
    assert _recall(found, pts) > 0.8
    assert len(found) < 2 * len(pts)


def test_bright_nuclei_dark_background():
    rng = np.random.default_rng(1)
    img, pts = _discs(rng, light_background=False)
    found = detect_nuclei(img, tile=256, overlap=32)
    assert _recall(found, pts) > 0.8
    assert len(found) < 2 * len(pts)


def test_empty_image():
    img = np.full((128, 128), 200, dtype=np.uint8)
    found = detect_nuclei(img)
    assert found.shape == (0, 2)
