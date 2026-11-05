"""Tests for mmprog.inference: final fit round trip, virtual map with stand-ins, exports."""
import numpy as np
import pandas as pd

from mmprog.paths import RunDir
from mmprog.evaluation import build_cv_data
from mmprog.inference import (fit_final_model, load_final_model, virtual_program_map,
                              sanity_check_on_training_slide, export_qupath, export_xenium_explorer, batch_inference)
from tests.test_evaluation import FakeFeatureSpace, FakeSelector, _fixture, _cfg


def test_final_fit_round_trip(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path)
    run = RunDir(cfg)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    model, fs, sel = fit_final_model(data, EMB, MOR, cfg, "cpu", run,
                                     feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    assert run.done("final_model")
    model2, fs2, sel2, ckpt = load_final_model(run, cfg, "cpu")
    assert ckpt["targets"] == ["t0_z", "t1_z"] and ckpt["d_in"] == fs.dim
    X = fs2.transform(EMB, MOR)
    from mmprog.models import predict
    p1 = predict(model, X[:10], data.XY[:10], data.SID[:10], "cpu")
    p2 = predict(model2, X[:10], data.XY[:10], data.SID[:10], "cpu")
    assert np.allclose(p1, p2, atol=1e-5)


def _fake_detect(img):
    rng = np.random.default_rng(0)
    return rng.uniform(5, img.shape[0] - 5, size=(40, 2)).astype(np.float32)


def _fake_embed(obs, store, enc, out_path, cfg, device):
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(obs), 6))
    mm[:] = np.random.default_rng(1).normal(size=(len(obs), 6)).astype(np.float16)
    mm.flush()
    return mm


def _fake_morpho(obs, store, cfg):
    return np.random.default_rng(2).normal(size=(len(obs), 3)).astype(np.float32)


def test_virtual_program_map_and_sanity(tmp_path):
    obs, EMB, MOR = _fixture()
    cfg = _cfg(tmp_path)
    run = RunDir(cfg)
    data = build_cv_data(obs, ["t0_z", "t1_z"], ["plasma", "T", "B"], cfg)
    model, fs, sel = fit_final_model(data, EMB, MOR, cfg, "cpu", run,
                                     feature_space_factory=FakeFeatureSpace, selector_factory=FakeSelector)
    img = np.zeros((100, 100), dtype=np.uint8)
    vmap = virtual_program_map(img, model, fs, sel, cfg, "cpu", data.targets, data.n_samples, encoder=object(),
                               detect_fn=_fake_detect, embed_fn=_fake_embed, morpho_fn=_fake_morpho)
    assert list(vmap.columns) == ["x", "y", "sample", "t0_pred", "t1_pred"]
    assert len(vmap) == 40 and (vmap["sample"] == "NEW").all()
    # The annotated slide lives in whole-slide coordinates: the crop origin (ox, oy) is added to every
    # centroid, and the measured scores are a noisy version of the predictions. A correct offset must
    # recover the match; the wrong sign must not.
    ox, oy = 300.0, 150.0
    rng = np.random.default_rng(5)
    slide = pd.DataFrame({"x": vmap["x"] + ox, "y": vmap["y"] + oy,
                          "t0_z": vmap["t0_pred"] + 0.05 * vmap["t0_pred"].std() * rng.normal(size=len(vmap))})
    r_ok = sanity_check_on_training_slide(vmap, slide, (ox, oy), "t0_z", cfg)
    assert r_ok > 0.9
    r_bad = sanity_check_on_training_slide(vmap, slide, (-ox, -oy), "t0_z", cfg)
    assert np.isnan(r_bad) or r_bad < 0.5


def test_exports(tmp_path):
    vmap = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "sample": "NEW", "a_pred": [0.1, 0.2], "b_pred": [1.0, 2.0]})
    q = tmp_path / "q.tsv"
    x = tmp_path / "x.csv"
    export_qupath(vmap, str(q), um_per_px=0.5)
    export_xenium_explorer(vmap, str(x), um_per_px=0.5)
    qd = pd.read_csv(q, sep="\t")
    assert list(qd.columns) == ["x_um", "y_um", "Measurement: a_pred", "Measurement: b_pred"]
    assert qd["x_um"].tolist() == [0.5, 1.0]
    xd = pd.read_csv(x)
    assert list(xd.columns) == ["cell_id", "x_centroid", "y_centroid", "a_pred", "b_pred"]
    assert xd["cell_id"].tolist() == ["n0", "n1"]


def test_batch_inference_with_stubs(tmp_path):
    cfg = _cfg(tmp_path)
    run = RunDir(cfg)
    slides = tmp_path / "slides"
    slides.mkdir()
    for n in ("s1.tif", "s2.ome.tif"):
        (slides / n).write_bytes(b"")
    vm = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "sample": "NEW", "m_pred": [0.0, 1.0, 2.0]})
    out = batch_inference(str(slides), None, None, None, cfg, "cpu", ["m_z"], 1, run, "m_z", 0.5,
                          load_image_fn=lambda *a: np.zeros((4, 4), np.uint8), vmap_fn=lambda *a, **k: vm)
    assert out["slide"].tolist() == ["s1", "s2.ome"]
    assert np.allclose(out["burden"], 2 / 3)
    assert run.done("batch_burden") and run.done("vmap_s1_qupath.tsv")
