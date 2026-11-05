import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mmprog.config import Config
from mmprog.targets import build_targets, program_score, score_lineages, standardise


def test_program_score_formula():
    cnt = np.array([0, 5, 10])
    total = np.array([100, 100, 0])
    expected = np.log1p(1e4 * cnt / np.maximum(total, 1))
    assert np.allclose(program_score(cnt, total), expected)
    assert program_score(np.array([0]), np.array([0]))[0] == 0.0


def test_standardise_modes():
    v = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    cohort = standardise(v, g, "cohort")
    assert abs(cohort.mean()) < 1e-6
    assert cohort[-1] > cohort[2]
    slide = standardise(v, g, "slide")
    assert np.allclose(slide[:3], slide[3:])          # each slide centred separately


def _toy_adata(n=60, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["WNT5A", "WNT5B", "ROR2", "DKK1", "CD38", "XBP1", "SDC1", "CD3D", "CD3E", "TRAC"] + [f"G{i}" for i in range(40)]
    X = rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)
    lineage_true = np.array(["plasma"] * (n // 2) + ["T"] * (n - n // 2))
    gi = {g: i for i, g in enumerate(genes)}
    for g in ["CD38", "XBP1", "SDC1"]:
        X[lineage_true == "plasma", gi[g]] += 8
    for g in ["CD3D", "CD3E", "TRAC"]:
        X[lineage_true == "T", gi[g]] += 8
    obs = pd.DataFrame(dict(sample=np.where(np.arange(n) % 2 == 0, "s1", "s2"), lineage_true=lineage_true),
                       index=[f"c{i}" for i in range(n)])
    total = X.sum(1)
    wnt = X[:, [gi[g] for g in ["WNT5A", "WNT5B", "ROR2", "DKK1"]]].sum(1)
    obs["cnt_total"] = total
    obs["cnt_wnt_camdr"] = wnt
    obs["nb_cnt_total"] = total * 3
    obs["nb_cnt_wnt_camdr"] = wnt * 3
    a = ad.AnnData(csr_matrix(np.log1p(X)), obs=obs, var=pd.DataFrame(index=genes))
    return a


def test_build_targets_tiny():
    cfg = Config()
    a = _toy_adata()
    programs = {"wnt_camdr": ["WNT5A", "WNT5B", "ROR2", "DKK1", "MISSING1"], "absent": ["ZZ1", "ZZ2", "ZZ3"]}
    a, targets, kept = build_targets(a, programs, cfg)
    assert targets == ["prog_wnt_camdr_cell_z", "prog_wnt_camdr_nb_z"]
    assert kept == {"wnt_camdr": ["WNT5A", "WNT5B", "ROR2", "DKK1"]}
    assert abs(a.obs["prog_wnt_camdr_cell_z"].mean()) < 1e-6
    assert "prog_wnt_camdr_cell" in a.obs


def test_score_lineages_recovers_planted():
    cfg = Config()
    a = _toy_adata()
    lineages = {"plasma": ["CD38", "XBP1", "SDC1"], "T": ["CD3D", "CD3E", "TRAC"], "tiny": ["G1"]}
    a, lin_names = score_lineages(a, lineages, cfg)
    assert lin_names == ["plasma", "T"]
    agreement = (a.obs["lineage_true"] == a.obs["lineage_name"]).mean()
    assert agreement > 0.95
    assert set(a.obs["lineage"].unique()) <= {0, 1}
