"""CODEX bridge: marker matching, lineage calls, niche flag, composition metrics, and the gated stage."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.config import Config  # noqa: E402
from mmprog.data.codex import (assign_lineage_from_markers, codex_to_obs, load_codex_table,  # noqa: E402
                               load_marker_mapping, match_markers)
from mmprog.evaluation.codex_validation import composition_agreement, niche_contrast, run_codex_stage  # noqa: E402
from mmprog.paths import RunDir  # noqa: E402


def _cfg(tmp_path, **kw) -> Config:
    base = dict(root=str(tmp_path / "ws"), local_data=str(tmp_path / "raw"), run_name="codex_test",
                demo_mode=True, encoder="tiny_test", pca_dim=8, device="cpu")
    base.update(kw)
    cfg = Config(**base)
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def demo_slide():
    """One synthetic slide with planted lineages, built the same way the pipeline builds its fixture."""
    from mmprog.data.synthetic import make_demo
    from mmprog.resources import load_lineages, load_programs
    cfg = Config(root="/tmp/unused", local_data="/tmp/unused", demo_mode=True)
    adata, images, manifest = make_demo(load_programs(), load_lineages(), cfg, n_patients=1, cells_per=700, img_size=512)
    sample = manifest[0]["sample"]
    obs = adata.obs[adata.obs["sample"] == sample].reset_index(drop=True)
    return obs, images[sample], obs["lineage_true"].values.astype(str), cfg


def test_alias_matching_is_case_insensitive():
    mapping = load_marker_mapping()
    matched = match_markers(["syndecan-1", "cd3E", "Foxp3", "aSMA", "Area", "PD-1"], mapping)
    assert matched["CD138"] == "syndecan-1"
    assert matched["CD3"] == "cd3E"
    assert matched["FOXP3"] == "Foxp3"
    assert matched["SMA"] == "aSMA"
    assert matched["PD1"] == "PD-1"
    assert "CD20" not in matched


def test_lineage_assignment_recovers_planted_lineages(demo_slide, tmp_path):
    from mmprog.data.synthetic_codex import write_demo_codex
    obs, img, lin_true, cfg = demo_slide
    csv_path, _ = write_demo_codex(str(tmp_path), obs, img, lin_true, cfg)
    table = load_codex_table(csv_path)
    assert {"cell_id", "x_um", "y_um"} <= set(table.columns)
    assert "Area" not in table.columns, "geometry columns must be stripped"
    mapping = load_marker_mapping()
    matched = match_markers(list(table.columns), mapping)
    lineage = assign_lineage_from_markers(table, mapping, matched)
    # NK and plasma share CD56 in the synthetic marker model; count a plasma/NK confusion as agreement on the rest
    comparable = ~np.isin(lin_true, ["NK"])
    agreement = (lineage.values[comparable] == lin_true[comparable]).mean()
    assert agreement > 0.8, agreement


def test_micron_to_pixel_round_trip(demo_slide, tmp_path):
    from mmprog.data.synthetic_codex import write_demo_codex
    obs, img, lin_true, cfg = demo_slide
    csv_path, _ = write_demo_codex(str(tmp_path), obs, img, lin_true, cfg)
    table = load_codex_table(csv_path)
    lineage = pd.Series(lin_true, index=table.index)
    cobs = codex_to_obs(table, lineage, cfg)
    np.testing.assert_allclose(cobs["x"].values, obs["x"].values, atol=0.05)
    np.testing.assert_allclose(cobs["y"].values, obs["y"].values, atol=0.05)


def test_plasma_dense_flag_on_five_cells():
    cfg = Config(root="/tmp/unused", local_data="/tmp/unused", nb_radius_um=30)
    # three plasma cells within 10 um of the origin, one far T cell, one far B cell
    table = pd.DataFrame({"cell_id": list("abcde"),
                          "x_um": [0.0, 5.0, 8.0, 200.0, 400.0], "y_um": [0.0, 0.0, 3.0, 200.0, 400.0]})
    lineage = pd.Series(["plasma", "plasma", "plasma", "T", "B"])
    cobs = codex_to_obs(table, lineage, cfg, min_plasma=3)
    # each CD138+ cell has two CD138+ neighbours (it does not count itself)
    assert cobs["n_plasma_in_radius"].tolist()[:3] == [2, 2, 2]
    assert cobs["plasma_dense"].tolist() == [False, False, False, False, False]
    cobs2 = codex_to_obs(table, lineage, cfg, min_plasma=2)
    assert cobs2["plasma_dense"].tolist() == [True, True, True, False, False]


def test_composition_agreement_identity_and_shuffle():
    rng = np.random.default_rng(0)
    lin_names = ["plasma", "T", "B", "stromal"]
    labels = rng.choice(lin_names, size=2000, p=[0.5, 0.3, 0.15, 0.05])
    same = composition_agreement(labels, labels, lin_names)
    assert same["clr_pearson_r"] == pytest.approx(1.0)
    assert same["aitchison_distance"] == pytest.approx(0.0)
    assert same["cell_agreement"] == pytest.approx(1.0)
    other = rng.choice(lin_names, size=2000, p=[0.05, 0.15, 0.3, 0.5])
    diff = composition_agreement(labels, other, lin_names)
    assert diff["clr_pearson_r"] < same["clr_pearson_r"]
    assert diff["aitchison_distance"] > 1.0


def test_niche_contrast_columns_and_direction():
    rng = np.random.default_rng(1)
    n = 500
    flag = np.arange(n) < 200
    vmap = pd.DataFrame({"a_pred": np.where(flag, 1.0, 0.0) + rng.normal(0, 0.2, n),
                         "b_pred": rng.normal(0, 1, n), "plasma_dense": flag,
                         "sample": rng.choice(["s1", "s2", "s3", "s4", "s5"], n)})
    out = niche_contrast(vmap, ["a_pred", "b_pred"], groups=vmap["sample"].values)
    assert {"program", "mean_dense", "mean_other", "cohens_d", "mannwhitney_p_descriptive", "n_slides_paired"} <= set(out.columns)
    assert out.set_index("program").loc["a_pred", "cohens_d"] > 3
    assert abs(out.set_index("program").loc["b_pred", "cohens_d"]) < 0.5
    assert out.set_index("program").loc["a_pred", "slide_wilcoxon_p"] < 0.1


def test_stage_is_gated_without_inputs(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    run = RunDir(cfg)
    out = run_codex_stage(cfg, run, model=None, fs=None, selector=None, targets=[], n_samples=1, device="cpu", lin_names=[])
    assert out is None
    assert "gate: set codex_table" in capsys.readouterr().out


def test_stage_runs_end_to_end_on_synthetic_fixture(demo_slide, tmp_path):
    """Plumbing test: tiny encoder, untrained ProgramNet, synthetic CODEX table and DAPI."""
    from mmprog.data.synthetic_codex import write_demo_codex
    from mmprog.features.encoder import embed_cells, get_encoder
    from mmprog.features.feature_space import FeatureSpace
    from mmprog.features.morphometrics import morphometrics
    from mmprog.inference.virtual_map import _SingleImageStore
    from mmprog.models.program_net import ProgramNet

    obs, img, lin_true, cfg0 = demo_slide
    csv_path, tif_path = write_demo_codex(str(tmp_path), obs, img, lin_true, cfg0)
    cfg = _cfg(tmp_path, codex_table=csv_path, codex_dapi=tif_path, codex_dapi_um_per_px=cfg0.um_per_px)
    run = RunDir(cfg)

    # a feature space fit on this slide's cells with the tiny encoder, and an untrained model of matching width
    enc = get_encoder("tiny_test", "cpu")
    store = _SingleImageStore(img)
    emb = embed_cells(obs.assign(sample="S"), store, enc, str(tmp_path / "emb.npy"), cfg, "cpu")
    mor = morphometrics(obs.assign(sample="S"), store, cfg)
    fs = FeatureSpace(cfg.pca_dim, True).fit(np.asarray(emb), mor)
    lin_names = ["plasma", "T", "NK", "B", "granulocyte", "monocyte", "erythroid", "megakaryocyte", "stromal"]
    targets = ["prog_wnt_camdr_nb_z", "prog_t_exhaustion_nb_z"]
    torch_seed = __import__("torch").manual_seed(0)  # noqa: F841
    model = ProgramNet(fs.dim, len(targets), len(lin_names), n_samples=1, hidden=16).eval()

    out = run_codex_stage(cfg, run, model, fs, None, targets, n_samples=1, device="cpu", lin_names=lin_names, encoder=enc)
    assert out is not None
    assert out["n_matched"] > 0.5 * out["n_detected"]
    assert set(out["composition"]) >= {"clr_pearson_r", "aitchison_distance", "cell_agreement"}
    assert {r["program"] for r in out["niche_contrast"]} == {"prog_wnt_camdr_nb_pred", "prog_t_exhaustion_nb_pred"}
    assert os.path.exists(run.art("codex"))
    assert json.load(open(run.art("codex")))["n_codex_cells"] == len(obs)
    assert os.path.exists(run.art("codex_matched_nuclei.csv"))


def test_dapi_pixel_size_resampling_and_missing_metadata(tmp_path):
    """A 0.325 um/px DAPI is resampled to the working frame; an image with no pixel size raises."""
    import tifffile
    from mmprog.evaluation.codex_validation import dapi_native_um_per_px, resample_to_working_frame

    cfg = Config(root=str(tmp_path), local_data=str(tmp_path / "raw"))
    img = (np.random.default_rng(0).uniform(0, 255, size=(400, 300))).astype(np.uint8)
    out = resample_to_working_frame(img, 0.325, cfg.um_per_px)
    expected = 0.325 / cfg.um_per_px
    assert abs(out.shape[0] / img.shape[0] - expected) < 0.02 and abs(out.shape[1] / img.shape[1] - expected) < 0.02
    assert out.dtype == np.uint8
    same = resample_to_working_frame(img, cfg.um_per_px, cfg.um_per_px)
    assert same.shape == img.shape

    bare = tmp_path / "bare.tif"
    tifffile.imwrite(str(bare), img)
    with pytest.raises(ValueError, match="codex_dapi_um_per_px"):
        dapi_native_um_per_px(str(bare), None)
    assert dapi_native_um_per_px(str(bare), 0.5) == 0.5

    # resolution tags in pixels per centimetre: 0.325 um/px = 30769.23 px/cm
    tagged = tmp_path / "tagged.tif"
    tifffile.imwrite(str(tagged), img, resolution=(1e4 / 0.325, 1e4 / 0.325), resolutionunit="CENTIMETER")
    assert abs(dapi_native_um_per_px(str(tagged), None) - 0.325) < 1e-3
