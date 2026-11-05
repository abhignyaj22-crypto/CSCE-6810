"""Classical nucleus detection for morphology-only inference.

At deployment there is no transcriptomics and no cell segmentation model: the
pipeline has to find nuclei in a plain image. The detector is deliberately
classical (background subtraction, local threshold, distance-transform
watershed), so that the deployed model has no dependency beyond scikit-image.

Two details matter in practice:

* Polarity. DAPI has bright nuclei on a dark background; H&E rendered to grey,
  and the synthetic fixture, have dark nuclei on a light background. The image
  median decides which case we are in (background dominates the pixel count).
* Tiling. A trephine core at level 1 is a few hundred megapixels. Processing
  it as one float32 array with several intermediate copies exhausts Colab RAM,
  so the image is processed in overlapping tiles and only centroids from each
  tile's interior are kept.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.ndimage as ndi
from skimage import color, feature, filters, measure, morphology, segmentation

from ..logging_utils import say


def _nuclear_channel(img: np.ndarray) -> np.ndarray:
    """Return a float image in [0, 1] where nuclei are BRIGHT, plus the polarity label."""
    if img.ndim == 3:
        # H&E RGB: the haematoxylin OD channel is already "bright nuclei".
        g = color.rgb2hed(img)[..., 0].astype(np.float32)
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        return g, "rgb_hematoxylin"
    g = img.astype(np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    if np.median(g) < 0.5:
        return g, "bright_nuclei_dark_background"
    return 1.0 - g, "dark_nuclei_light_background"


def _detect_tile(img: np.ndarray, min_area: int, max_area: int, bg_sigma: float) -> tuple[np.ndarray, str]:
    """Detect nuclei in one tile; returns (N, 2) float32 centroids as (x, y)."""
    nuc, polarity = _nuclear_channel(img)
    # Remove slow background variation (illumination, tissue autofluorescence).
    nuc = np.clip(nuc - ndi.gaussian_filter(nuc, bg_sigma), 0, None)
    mask = (nuc > filters.threshold_local(nuc, block_size=51, offset=-0.02)) & (nuc > 0.02)
    mask = morphology.opening(mask, morphology.disk(1))
    try:
        mask = morphology.remove_small_objects(mask, max_size=min_area - 1)
    except TypeError:  # scikit-image < 0.26
        mask = morphology.remove_small_objects(mask, min_size=min_area)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32), polarity

    # Split touching nuclei with a watershed seeded at distance-transform peaks.
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=3, labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    pts = [[p.centroid[1], p.centroid[0]] for p in measure.regionprops(labels) if min_area <= p.area <= max_area]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2), polarity


def detect_nuclei(img: np.ndarray, min_area: int = 8, max_area: int = 400, bg_sigma: float = 25,
                  tile: int = 2048, overlap: int = 64) -> np.ndarray:
    """Detect nuclei over a whole image, tile by tile.

    Returns an (N, 2) float32 array of (x, y) centroids in image pixels.
    """
    t0 = time.time()
    H, W = img.shape[:2]
    out = []
    polarity = "n/a"
    n_tiles = 0
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            n_tiles += 1
            ya, xa = max(y0 - overlap, 0), max(x0 - overlap, 0)
            yb, xb = min(y0 + tile + overlap, H), min(x0 + tile + overlap, W)
            pts, polarity = _detect_tile(img[ya:yb, xa:xb], min_area, max_area, bg_sigma)
            if len(pts) == 0:
                continue
            pts += np.array([xa, ya], dtype=np.float32)
            # Keep only centroids inside this tile's own (non-overlap) region.
            inside = (pts[:, 0] >= x0) & (pts[:, 0] < min(x0 + tile, W)) & \
                     (pts[:, 1] >= y0) & (pts[:, 1] < min(y0 + tile, H))
            out.append(pts[inside])
    result = np.vstack(out) if out else np.zeros((0, 2), dtype=np.float32)
    say(f"nucleus detection: {len(result)} nuclei in {W}x{H} image, {n_tiles} tiles, "
        f"polarity={polarity}, area window [{min_area}, {max_area}] px, {time.time() - t0:.1f}s")
    return result
