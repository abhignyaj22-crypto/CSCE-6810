"""load_image projects a (Z, H, W) OME-TIFF z-stack inside a bbox and converts to uint8."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
import tifffile

from mmprog.data.images import level_shape, load_image


def test_zstack_max_projection_in_bbox(tmp_path):
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 4000, size=(5, 300, 400), dtype=np.uint16)  # (Z, H, W)
    path = str(tmp_path / "morphology.ome.tif")
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})
    assert level_shape(path, 0) == (300, 400)
    bbox = (50, 20, 250, 220)  # x0, y0, x1, y1
    img = load_image(path, level=0, bbox=bbox, z_project="max")
    assert img.shape == (200, 200) and img.dtype == np.uint8
    expected = stack[:, 20:220, 50:250].max(axis=0)
    # monotone mapping: ordering of pixel values is preserved after the percentile stretch
    assert np.corrcoef(img.ravel().astype(float), expected.ravel().astype(float))[0, 1] > 0.99
    mid = load_image(path, level=0, bbox=bbox, z_project="mid")
    assert np.corrcoef(mid.ravel().astype(float), stack[2, 20:220, 50:250].ravel().astype(float))[0, 1] > 0.99


def test_plain_2d_image(tmp_path):
    img2d = (np.arange(100 * 120) % 255).reshape(100, 120).astype(np.uint8)
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, img2d)
    out = load_image(path, level=0)
    assert np.array_equal(out, img2d)


def _write_pyramid(path, stack):
    """Level 0 and a 2x downsampled level 1, the layout tifffile reads back as series[0].levels."""
    with tifffile.TiffWriter(path) as tw:
        tw.write(stack, subifds=1, metadata={"axes": "ZYX"}, tile=(32, 32))
        tw.write(stack[:, ::2, ::2], subfiletype=1, tile=(32, 32))


def test_pyramid_level1_zstack_projection(tmp_path):
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 4000, size=(3, 128, 192), dtype=np.uint16)
    path = str(tmp_path / "morphology.ome.tif")
    _write_pyramid(path, stack)
    assert level_shape(path, 0) == (128, 192)
    assert level_shape(path, 1) == (64, 96)
    bbox = (10, 5, 50, 45)
    img = load_image(path, level=1, bbox=bbox)
    expected = stack[:, ::2, ::2][:, 5:45, 10:50].max(axis=0)
    assert img.shape == (40, 40)
    assert np.corrcoef(img.ravel().astype(float), expected.ravel().astype(float))[0, 1] > 0.99


def test_missing_pyramid_level_raises(tmp_path):
    stack = np.zeros((3, 64, 64), dtype=np.uint16)
    path = str(tmp_path / "flat.ome.tif")
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="level 1 does not exist"):
        level_shape(path, 1)
    with pytest.raises(ValueError, match="level 1 does not exist"):
        load_image(path, level=1)


def test_rgb_interleaved_stays_rgb(tmp_path):
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 255, size=(13, 150, 3), dtype=np.uint8)   # height 13: would look like a z-stack by shape
    path = str(tmp_path / "he.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")
    assert level_shape(path, 0) == (13, 150)
    out = load_image(path, level=0)
    assert out.shape == (13, 150, 3)
    assert np.array_equal(out, rgb)


def test_multichannel_cyx_takes_channel0(tmp_path):
    rng = np.random.default_rng(3)
    stack = rng.integers(0, 4000, size=(4, 100, 150), dtype=np.uint16)   # morphology_focus: DAPI + 3 stains
    path = str(tmp_path / "morphology_focus.ome.tif")
    tifffile.imwrite(path, stack, metadata={"axes": "CYX"})
    assert level_shape(path, 0) == (100, 150)
    out = load_image(path, level=0, bbox=(10, 20, 60, 70))
    assert out.shape == (50, 50)
    expected = stack[0, 20:70, 10:60]
    assert np.corrcoef(out.ravel().astype(float), expected.ravel().astype(float))[0, 1] > 0.99


def test_whole_slide_stretch_ignores_empty_glass(tmp_path):
    """With bbox=None the stretch is measured on tissue, so tissue pixels are not crushed by glass."""
    img = np.full((400, 400), 100, dtype=np.uint16)          # glass
    img[150:250, 150:250] = 3000                              # tissue
    img[180:220, 180:220] = 3500
    path = str(tmp_path / "slide.tif")
    tifffile.imwrite(path, img)
    out = load_image(path, level=0)
    assert out[200, 200] > out[160, 160]                      # tissue contrast survives
    assert out[200, 200] >= 250 and out[10, 10] == 0


def _store_cfg(tmp_path):
    from mmprog.config import Config
    return Config(run_name="t", root=str(tmp_path / "ws"), local_data=str(tmp_path / "raw"),
                  img_level=0, patch_px=16)


def test_image_store_cache_round_trip_and_bbox_check(tmp_path):
    from mmprog.data.images import ImageStore
    rng = np.random.default_rng(4)
    stack = rng.integers(0, 4000, size=(2, 200, 300), dtype=np.uint16)
    path = str(tmp_path / "morphology.ome.tif")
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})
    cfg = _store_cfg(tmp_path)
    row = dict(sample="S1", h5="x", cells="y", image=path, he_image=None)
    xy_small = np.array([[100, 100], [120, 110]], dtype=np.float32)
    store = ImageStore(cfg, [row], {"S1": xy_small}, margin=16)
    img = store.get("S1")
    assert store.offset("S1") == (84, 84) and store.bbox["S1"] == (84, 84, 137, 127)
    assert img.shape == (127 - 84, 137 - 84)
    cache = store._cache_path("S1")
    assert os.path.exists(cache)
    z = np.load(cache)
    assert "bbox" in z and "stretch" in z
    # same cells: cache is reused
    store2 = ImageStore(cfg, [row], {"S1": xy_small}, margin=16)
    assert np.array_equal(store2.get("S1"), img) and store2.offset("S1") == (84, 84)
    # a cell outside the cached crop: cache is rejected and the crop recomputed to cover it
    xy_big = np.vstack([xy_small, [[250, 150]]]).astype(np.float32)
    store3 = ImageStore(cfg, [row], {"S1": xy_big}, margin=16)
    img3 = store3.get("S1")
    assert store3.bbox["S1"][2] > 250 and img3.shape[1] > img.shape[1]


def test_check_axes_against_image():
    from mmprog.data.images import check_axes_against_image
    xy = np.array([[10, 20], [90, 140]], dtype=np.float32)
    out, swapped = check_axes_against_image(xy, width=100, height=150, sample="a")
    assert not swapped and np.array_equal(out, xy)
    out, swapped = check_axes_against_image(xy, width=150, height=100, sample="b")   # fits only transposed
    assert swapped and np.array_equal(out, xy[:, ::-1])
    with pytest.raises(ValueError):
        check_axes_against_image(xy, width=50, height=50, sample="c")
