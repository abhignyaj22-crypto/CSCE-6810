"""Data access: GEO manifest, downloads, 10x readers, whole-slide images, synthetic fixture."""
from .manifest import build_manifest_from_geo, load_manifest, save_manifest, select_samples
from .download import ensure_decompressed, ensure_raw, fetch_sample_files
from .tenx import load_row, load_visium_hd, load_xenium, read_10x_h5
from .images import ImageStore, level_shape, load_image, um_per_px_level0
from .synthetic import make_demo

__all__ = [
    "build_manifest_from_geo", "load_manifest", "save_manifest", "select_samples",
    "ensure_decompressed", "ensure_raw", "fetch_sample_files",
    "load_row", "load_visium_hd", "load_xenium", "read_10x_h5",
    "ImageStore", "level_shape", "load_image", "um_per_px_level0",
    "make_demo",
]
