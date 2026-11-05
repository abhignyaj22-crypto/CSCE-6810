import numpy as np
import pandas as pd

from mmprog.config import Config
from mmprog.features import FeatureSpace, MORPHO_NAMES, embed_cells, extract_patch, get_encoder, morphometrics


class _Images:
    def __init__(self, img):
        self.img = img

    def get(self, sample):
        return self.img

    def offset(self, sample):
        return (0, 0)


def test_extract_patch_pads_at_border():
    img = np.full((100, 100), 200, dtype=np.uint8)
    p = extract_patch(img, 2, 2, 16)
    assert p.shape == (16, 16, 3)
    assert p[0, 0, 0] == 0                 # outside the image: zero padded
    assert p[10, 10, 0] == 200
    center = extract_patch(img, 50, 50, 16)
    assert (center == 200).all()
    rgb = extract_patch(np.full((40, 40, 3), 7, dtype=np.uint8), 39, 39, 8)
    assert rgb.shape == (8, 8, 3)


def test_feature_space_shapes_and_keep_dims():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, 40)).astype(np.float16)
    mor = rng.normal(size=(300, 8)).astype(np.float32)
    fs = FeatureSpace(pca_dim=10, use_morpho=True, fit_max=200, chunk=64).fit(
        emb, mor, idx=np.arange(250), seed=1, keep_emb_dims=np.arange(30), keep_mor_dims=np.array([0, 2, 4]))
    X = fs.transform(emb, mor)
    assert X.shape == (300, 13)
    assert fs.dim == 13
    assert fs.explained_variance_ratio_.shape == (10,)
    fs2 = FeatureSpace(pca_dim=10, use_morpho=False).fit(emb, mor)
    assert fs2.transform(emb, mor).shape == (300, 10)


def test_embed_cells_tiny_encoder_writes_memmap(tmp_path):
    cfg = Config(patch_px=32, emb_batch=16)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    obs = pd.DataFrame(dict(x=rng.uniform(0, 256, 50), y=rng.uniform(0, 256, 50), sample="s1"))
    enc = get_encoder("tiny_test", "cpu")
    out = tmp_path / "emb.npy"
    mm = embed_cells(obs, _Images(img), enc, str(out), cfg, "cpu")
    assert mm.shape == (50, 64) and mm.dtype == np.float16
    loaded = np.load(out, mmap_mode="r")
    assert loaded.shape == (50, 64)
    assert np.isfinite(np.asarray(loaded, dtype=np.float32)).all()


def test_morphometrics_shape_both_density_paths():
    cfg = Config(patch_px=16, density_radii_px=(10, 20))
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(128, 128), dtype=np.uint8)
    obs = pd.DataFrame(dict(x=rng.uniform(0, 128, 40), y=rng.uniform(0, 128, 40), sample="s1"))
    feats = morphometrics(obs, _Images(img), cfg)
    assert feats.shape == (40, 7)
    assert len(MORPHO_NAMES(cfg)) == 7
    obs["density_r10"] = 3.0
    obs["density_r20"] = 9.0
    feats2 = morphometrics(obs, _Images(img), cfg)
    assert (feats2[:, 5] == 3.0).all() and (feats2[:, 6] == 9.0).all()
