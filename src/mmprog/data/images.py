"""Whole-slide morphology images: OME-TIFF reading and a per-sample crop cache.

The Xenium ``morphology.ome.tif`` is a pyramidal DAPI z-stack (13 planes at
0.2125 um/px at level 0). A level-1 plane of one trephine is a few hundred
megapixels; the full z-stack is 13 times that. Three rules keep this inside
Colab RAM:

1. Decode the pyramid level into a disk-backed memmap (``tifffile`` ``out="memmap"``),
   never into a RAM array.
2. Project over z only inside the bounding box of the sample's cells, in
   512-row blocks, so the (Z, H, W) crop is never materialised.
3. Convert to uint8 in row blocks rather than through one float32 copy.

Axis handling goes by the ``axes`` string tifffile reports for the level
(``ZYX``, ``CYX``, ``YXS``, ``YX``), never by guessing from the shape: an RGB
H&E slide of height 13 and a 13-plane z-stack must not be confused.

The resulting crop (100 to 400 MB) is cached as ``.npz`` under ``cfg.crop_dir``
(Google Drive on Colab) together with the bbox, the intensity stretch and the
transposition decision, so it survives session restarts; afterwards the
multi-gigabyte decompressed TIFF is deleted.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from skimage import filters

from ..config import Config
from ..logging_utils import say, warn
from ..preprocessing.stain import harmonise_stain
from .download import ensure_decompressed, ensure_raw


# --------------------------------------------------------------------------- metadata
def um_per_px_level0(path: str, default: float = 0.2125) -> float:
    """Level-0 pixel size from the OME-XML ``PhysicalSizeX``; ``default`` if absent."""
    try:
        with tifffile.TiffFile(ensure_decompressed(path)) as tif:
            match = re.search(r'PhysicalSizeX="([\d.]+)"', tif.ome_metadata or "")
            if match:
                value = float(match.group(1))
                say(f"OME metadata PhysicalSizeX = {value} um/px for {os.path.basename(path)}")
                return value
    except Exception as exc:  # unreadable metadata is not fatal; the default is the Xenium spec
        warn(f"could not read OME metadata from {path}: {exc}")
    say(f"using default pixel size {default} um/px")
    return default


def _select_level(tif: tifffile.TiffFile, level: int):
    """The requested pyramid level, or a clear error listing what the file holds.

    Falling back to level 0 silently would apply level-1 cell coordinates to a
    level-0 image and crop the wrong region at half scale, so this raises.
    """
    series = tif.series[0]
    levels = list(series.levels) if hasattr(series, "levels") else [series]
    if level >= len(levels):
        available = ", ".join(f"level {i}: {tuple(lv.shape)} axes {lv.axes}" for i, lv in enumerate(levels))
        raise ValueError(f"{tif.filename}: pyramid level {level} does not exist. Available: {available}. "
                         f"Set img_level to one of these or supply a pyramidal OME-TIFF.")
    return levels[level]


def _axes_layout(lvl) -> Tuple[str, Tuple[int, ...]]:
    """Axes string and shape of a level with singleton axes removed."""
    axes = str(getattr(lvl, "axes", "") or "")
    shape = tuple(int(s) for s in lvl.shape)
    if not axes or len(axes) != len(shape):
        # No usable axes string: fall back to the plain interpretation of the trailing axes.
        axes = "YX" if len(shape) == 2 else ("ZYX" if len(shape) == 3 and shape[0] > 4 else "YXS")
    keep = [(a, s) for a, s in zip(axes, shape) if s > 1 or a in "YX"]
    return "".join(a for a, _ in keep), tuple(s for _, s in keep)


def level_shape(path: str, level: int) -> Tuple[int, int]:
    """(H, W) of a pyramid level, read off the Y and X axes."""
    with tifffile.TiffFile(ensure_decompressed(path)) as tif:
        axes, shape = _axes_layout(_select_level(tif, level))
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


# --------------------------------------------------------------------------- decoding
def _to_uint8_chunked(arr: np.ndarray, lo: float, hi: float, rows: int = 1024) -> np.ndarray:
    """Percentile-stretch to uint8 in row blocks, never allocating a full float32 copy."""
    out = np.empty(arr.shape, dtype=np.uint8)
    for r in range(0, arr.shape[0], rows):
        block = arr[r:r + rows].astype(np.float32)
        out[r:r + rows] = np.clip((block - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def _stretch_bounds(arr: np.ndarray, whole_slide: bool) -> Tuple[float, float]:
    """1 / 99.5 percentile bounds of the tissue.

    Inside a cell bbox every pixel is tissue. On a whole slide (``bbox=None``,
    deployment) most pixels are empty glass, which would pull the stretch down
    and make the deployed intensities differ from the training crops; so the
    percentiles are taken over the foreground (Otsu on a 1/16 subsample) only.
    """
    sub = np.asarray(arr[::16, ::16], dtype=np.float32)
    if sub.ndim == 3:
        sub = sub.mean(axis=-1)
    if whole_slide and sub.size > 64:
        thr = filters.threshold_otsu(sub)
        fg = sub[sub > thr] if np.median(sub) < thr else sub[sub < thr]  # bright-on-dark (DAPI) or dark-on-light
        if fg.size >= 64:
            sub = fg
    lo, hi = np.percentile(sub, [1, 99.5])
    return float(lo), float(hi)


def _read_level_region(lvl, bbox: Optional[Tuple[int, int, int, int]], z_project: str, keep_rgb: bool
                       ) -> Tuple[np.ndarray, Tuple[int, ...], str]:
    """Decode one level into a (H, W) or (H, W, 3) array for the bbox, by axes."""
    axes, _ = _axes_layout(lvl)
    mm = lvl.asarray(out="memmap")
    arr = np.squeeze(mm)
    if arr.ndim != len(axes):
        raise ValueError(f"axes {axes!r} do not match decoded shape {arr.shape}")
    full_shape = arr.shape
    H, W = arr.shape[axes.index("Y")], arr.shape[axes.index("X")]
    x0, y0, x1, y1 = bbox if bbox is not None else (0, 0, W, H)

    if axes == "ZYX":
        # z-stack: project inside the bbox only, block by block, so (Z, h, w) never exists in RAM
        Z = arr.shape[0]
        out = np.zeros((y1 - y0, x1 - x0), dtype=arr.dtype)
        for r in range(y0, y1, 512):
            r1 = min(r + 512, y1)
            if z_project == "mid":
                out[r - y0:r1 - y0] = np.asarray(arr[Z // 2, r:r1, x0:x1])
            else:
                out[r - y0:r1 - y0] = np.asarray(arr[:, r:r1, x0:x1]).max(axis=0)
        note = f"z-stack of {Z} planes projected ({z_project})"
    elif axes == "CYX":
        # multi-channel (e.g. morphology_focus: DAPI + 3 stains): channel 0 is DAPI
        out = np.array(arr[0, y0:y1, x0:x1])
        note = f"{arr.shape[0]}-channel image, channel 0 (DAPI) taken"
    elif axes == "YXS":
        if keep_rgb and arr.shape[-1] >= 3:
            out = np.array(arr[y0:y1, x0:x1, :3])
            note = "interleaved RGB kept"
        else:
            out = np.array(arr[y0:y1, x0:x1, 0])
            note = f"interleaved {arr.shape[-1]}-sample image, sample 0 taken"
    elif axes == "YX":
        out = np.array(arr[y0:y1, x0:x1])
        note = "single plane"
    else:
        raise ValueError(f"unsupported axes layout {axes!r} with shape {full_shape}")
    del mm  # releases the temporary memmap file
    return out, full_shape, note


def load_image(path: str, level: int, bbox: Optional[Tuple[int, int, int, int]] = None,
               z_project: str = "max", stain_mode: str = "native",
               stretch: Optional[Tuple[float, float]] = None, return_stretch: bool = False):
    """Load one pyramid level (optionally a bbox crop) as a uint8 2-D or RGB array.

    ``bbox`` is ``(x0, y0, x1, y1)`` in pixels of ``level``. RGB is kept only for
    interleaved (``YXS``) files, i.e. H&E; Xenium multi-channel stacks yield DAPI.
    ``stretch`` fixes the (lo, hi) intensity bounds (use the training run's bounds
    at deployment); otherwise they are measured on the tissue. With
    ``return_stretch`` the bounds used are returned alongside the image.
    """
    t0 = time.time()
    with tifffile.TiffFile(ensure_decompressed(path)) as tif:
        lvl = _select_level(tif, level)
        arr, full_shape, note = _read_level_region(lvl, bbox, z_project, keep_rgb=True)
        src_dtype = lvl.dtype
    if arr.size == 0:
        raise ValueError(f"empty crop: level {level} image shape {full_shape}, bbox (x0,y0,x1,y1)={bbox}")
    say(f"{os.path.basename(path)} level {level} {full_shape}: {note}; region {arr.shape}", indent=1)
    lo, hi = (0.0, 255.0)
    if arr.dtype != np.uint8:
        lo, hi = stretch if stretch is not None else _stretch_bounds(arr, whole_slide=bbox is None)
        arr = _to_uint8_chunked(arr, lo, hi)
        say(f"converted {src_dtype} to uint8 with stretch [{lo:.0f}, {hi:.0f}] "
            f"({'fixed' if stretch is not None else 'tissue percentiles 1/99.5'})", indent=1)
    say(f"loaded {os.path.basename(path)}: {arr.shape} uint8, {arr.nbytes / 1e6:.0f} MB, {time.time() - t0:.1f}s", indent=1)
    arr = harmonise_stain(arr, stain_mode)
    return (arr, (float(lo), float(hi))) if return_stretch else arr


# --------------------------------------------------------------------------- coordinates
def check_axes_against_image(xy: np.ndarray, width: int, height: int, sample: str) -> Tuple[np.ndarray, bool]:
    """Swap x and y when the cells only fit the image transposed; raise when they fit neither way.

    Done once, on the cell table, before anything downstream (summaries, crops,
    patches) uses the coordinates, so every consumer sees the same frame.
    """
    xy = np.asarray(xy, dtype=np.float32)
    xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())
    if xmax <= width and ymax <= height:
        say(f"{sample}: cell coordinates fit the {width}x{height} image as (x, y)", indent=1)
        return xy, False
    if xmax <= height and ymax <= width:
        warn(f"{sample}: cells (x<={xmax:.0f}, y<={ymax:.0f}) only fit the {width}x{height} image transposed; "
             "swapping x and y for this slide")
        return xy[:, ::-1].copy(), True
    raise ValueError(f"{sample}: cell coordinates (x<={xmax:.0f}, y<={ymax:.0f}) lie outside the {width}x{height} "
                     f"image in both orientations. Check um/px and img_level.")


# --------------------------------------------------------------------------- store
class ImageStore:
    """Per-sample morphology crops with an on-disk cache and one image in memory.

    ``obs_xy_by_sample[sample]`` is an (N, 2) float32 array of working-level
    pixel coordinates of that sample's cells, ideally every QC-passing cell so
    the crop covers any later subsample. The crop is their bounding box plus
    ``margin`` pixels. ``offset(sample)`` returns the crop origin, which callers
    subtract from cell coordinates before indexing into the crop.
    """

    def __init__(self, cfg: Config, manifest_rows: List[Dict[str, Any]],
                 obs_xy_by_sample: Dict[str, np.ndarray],
                 demo_images: Optional[Dict[str, np.ndarray]] = None, margin: Optional[int] = None):
        self.cfg = cfg
        self.rows = {r["sample"]: r for r in manifest_rows}
        self.xy = {s: np.asarray(v, dtype=np.float32) for s, v in obs_xy_by_sample.items()}
        self.demo_images = demo_images or {}
        self.margin = margin or cfg.patch_px
        self._cache: Dict[str, np.ndarray] = {}
        self._offset: Dict[str, Tuple[int, int]] = {}
        self.bbox: Dict[str, Tuple[int, int, int, int]] = {}
        self.stretch: Dict[str, Tuple[float, float]] = {}
        os.makedirs(cfg.crop_dir, exist_ok=True)

    # ------------------------------------------------------------------ public
    def get(self, sample: str) -> np.ndarray:
        if sample in self._cache:
            return self._cache[sample]
        self._cache.clear()  # one decoded crop in RAM at a time
        if sample in self.demo_images:
            self._cache[sample] = self.demo_images[sample]
            self._offset[sample] = (0, 0)
            return self._cache[sample]
        cache_path = self._cache_path(sample)
        if os.path.exists(cache_path) and self._load_cached(sample, cache_path):
            self._release_raw_image(sample)
            return self._cache[sample]
        img, bbox, stretch = self._decode_crop(sample)
        self._remember(sample, img, bbox, stretch)
        np.savez_compressed(cache_path, img=img, bbox=np.array(bbox), stretch=np.array(stretch))
        say(f"image {sample}: crop cached to {cache_path}")
        self._release_raw_image(sample)
        return img

    def prepare(self, sample: str, xy: np.ndarray) -> None:
        """Crop, cache and release this sample now (used inside the ingest loop).

        Doing it per sample during S1 means only one decompressed multi-GB TIFF
        exists at any time, instead of one per slide until S3.
        """
        self.xy[sample] = np.asarray(xy, dtype=np.float32)
        self.get(sample)
        self._cache.clear()

    def offset(self, sample: str) -> Tuple[int, int]:
        if sample not in self._offset:
            self.get(sample)
        return self._offset[sample]

    # ----------------------------------------------------------------- private
    def _release_raw_image(self, sample: str) -> None:
        """Drop the morphology file once the crop is on disk. The matrix and cell table stay."""
        if self.cfg.retain_raw_images:
            return
        row = self.rows.get(sample) or {}
        path = row.get("image")
        if not path:
            return
        candidates = [path, path + ".gz"]
        if path.endswith(".gz"):
            candidates.append(path[:-3])
            candidates.append(path[:-3] + ".part")
        for p in candidates:
            if p and os.path.isfile(p):
                nbytes = os.path.getsize(p)
                os.remove(p)
                say(f"deleted raw image {os.path.basename(p)} ({nbytes / 1e9:.2f} GB) after the crop was saved", indent=1)

    def _cache_path(self, sample: str) -> str:
        cfg = self.cfg
        name = f"{sample}_L{cfg.img_level}_z{cfg.z_project}_m{self.margin}_{cfg.stain_mode}.npz"
        return os.path.join(cfg.crop_dir, name)

    def _remember(self, sample: str, img: np.ndarray, bbox: Tuple[int, int, int, int],
                  stretch: Tuple[float, float]) -> None:
        self._cache[sample] = img
        self._offset[sample] = (int(bbox[0]), int(bbox[1]))
        self.bbox[sample] = tuple(int(v) for v in bbox)
        self.stretch[sample] = (float(stretch[0]), float(stretch[1]))

    def _load_cached(self, sample: str, cache_path: str) -> bool:
        """Load a cached crop if it covers the cells we have; otherwise say why and recompute."""
        import zipfile
        try:
            z = np.load(cache_path)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            warn(f"image {sample}: cache {os.path.basename(cache_path)} is unreadable ({exc}); deleting it and recomputing")
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return False
        if "bbox" not in z:
            warn(f"image {sample}: cache {os.path.basename(cache_path)} is from an older version without a bbox; recomputing")
            return False
        bbox = tuple(int(v) for v in z["bbox"])
        xy = self.xy.get(sample)
        if xy is not None and len(xy):
            x0, y0, x1, y1 = bbox
            inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            if not inside.all():
                warn(f"image {sample}: cached crop bbox {bbox} misses {int((~inside).sum())} of {len(xy)} cells "
                     "(different cell set than when it was cached); recomputing")
                return False
        self._remember(sample, z["img"], bbox, tuple(float(v) for v in z["stretch"]))
        say(f"image {sample}: crop {self._cache[sample].shape} loaded from cache, bbox {bbox}, "
            f"stretch [{self.stretch[sample][0]:.0f}, {self.stretch[sample][1]:.0f}]")
        return True

    def _decode_crop(self, sample: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]:
        cfg = self.cfg
        row = ensure_raw(self.rows[sample], cfg)
        path = row.get("he_image") or row["image"]
        xy = self.xy[sample]
        H, W = level_shape(path, cfg.img_level)
        xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
        ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
        say(f"{sample}: level-{cfg.img_level} image {W}x{H} (WxH) | cells x [{xmin:.0f}, {xmax:.0f}] "
            f"y [{ymin:.0f}, {ymax:.0f}] | um/px L0 {row.get('um_per_px_l0')}")
        if xmax > W or ymax > H:
            # the pipeline swaps axes in S1 before any coordinate is used; reaching here means it did not run
            raise ValueError(f"{sample}: cell coordinates exceed the image ({W}x{H}); run check_axes_against_image "
                             "on the cell table first, and check um/px and img_level")

        m = self.margin
        x0, y0 = max(int(xmin) - m, 0), max(int(ymin) - m, 0)
        x1, y1 = min(int(xmax) + m + 1, W), min(int(ymax) + m + 1, H)
        if x1 - x0 < cfg.patch_px or y1 - y0 < cfg.patch_px:
            raise ValueError(f"{sample}: degenerate bbox {(x0, y0, x1, y1)} for image {W}x{H}")
        bbox = (x0, y0, x1, y1)
        img, stretch = load_image(path, cfg.img_level, bbox, cfg.z_project, cfg.stain_mode, return_stretch=True)

        decompressed = path[:-3] if path.endswith(".gz") else path
        from ..observations import record_image_crop
        record_image_crop(cfg, sample, img, bbox, (H, W), decompressed if os.path.exists(decompressed) else None)
        decompressed = path[:-3] if path.endswith(".gz") else None
        if decompressed and os.path.exists(decompressed):
            os.remove(decompressed)  # keep the .gz, drop the multi-GB copy
            say(f"deleted decompressed copy {os.path.basename(decompressed)}", indent=1)
        say(f"image {sample}: crop {x1 - x0}x{y1 - y0} of {W}x{H} at level {cfg.img_level}, "
            f"{img.nbytes / 1e6:.0f} MB, origin ({x0}, {y0}), stretch [{stretch[0]:.0f}, {stretch[1]:.0f}]")
        return img, bbox, stretch
