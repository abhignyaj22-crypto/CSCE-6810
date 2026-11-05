"""Patient-level MANOVA: planted stage effect is detected, a null is not, degenerate input is skipped."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.evaluation.manova import (manova_by_stage, patient_level_table, permutation_manova,  # noqa: E402
                                      run_manova_stage)


def _table(effect: float, seed: int = 0, n_per_group=(6, 6, 6, 6)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g, (stage, n) in enumerate(zip(["control", "MGUS", "SMM", "MM"], n_per_group)):
        for i in range(n):
            v = rng.normal(size=4) + effect * g * np.array([1.0, 0.5, 0.0, -0.5])
            rows.append(dict(patient=f"{stage}_{i}", stage=stage, a=v[0], b=v[1], c=v[2], d=v[3]))
    return pd.DataFrame(rows)


def test_planted_effect_detected_by_both_tests():
    tab = _table(effect=2.5)
    par = manova_by_stage(tab, ["a", "b", "c", "d"])
    perm = permutation_manova(tab, ["a", "b", "c", "d"], n_perm=299, seed=1)
    assert par["status"] == "ok" and par["p_value"] < 0.01
    assert 0.0 <= par["partial_eta2"] <= 1.0
    assert perm["p_value"] < 0.02


def test_null_is_not_rejected():
    tab = _table(effect=0.0, seed=3)
    par = manova_by_stage(tab, ["a", "b", "c", "d"])
    perm = permutation_manova(tab, ["a", "b", "c", "d"], n_perm=299, seed=4)
    assert par["p_value"] > 0.05
    assert perm["p_value"] > 0.05


def test_degenerate_cases_are_skipped():
    tab = _table(effect=1.0, n_per_group=(3, 0, 0, 0))
    assert manova_by_stage(tab, ["a", "b", "c", "d"])["status"] == "skipped"
    tab = _table(effect=1.0, n_per_group=(2, 2, 1, 1))   # groups of 1 dropped, n too small for 4 DVs
    assert permutation_manova(tab, ["a", "b", "c", "d"])["status"] == "skipped"


def test_patient_table_and_stage_runner(tmp_path):
    rng = np.random.default_rng(0)
    n_pat, cells = 8, 200
    obs = pd.DataFrame({
        "patient": np.repeat([f"P{i}" for i in range(n_pat)], cells),
        "stage": np.repeat(["control"] * 4 + ["MM"] * 4, cells),
    })
    stage_effect = np.repeat([0.0] * 4 + [1.0] * 4, cells)
    obs["prog_x_cell_z"] = rng.normal(size=len(obs))
    obs["prog_x_nb_z"] = rng.normal(size=len(obs)) + stage_effect
    obs["prog_y_nb_z"] = rng.normal(size=len(obs)) - stage_effect
    targets = ["prog_x_cell_z", "prog_x_nb_z", "prog_y_nb_z"]
    pred = obs[targets].to_numpy() * 0.8 + rng.normal(scale=0.3, size=(len(obs), 3))
    table = patient_level_table(obs, pred, targets)
    assert list(table.columns[:3]) == ["patient", "stage", "n_cells"]
    assert "prog_x_nb_z_pred" in table and "prog_x_cell_z" not in table   # nb targets only

    class Run:
        def art(self, name):
            return str(tmp_path / {"manova": "manova_results.csv"}.get(name, name))

    class Cfg:
        seed = 0

    res = run_manova_stage(obs, pred, targets, Run(), Cfg(), {"control": 0, "MM": 3})
    assert (tmp_path / "manova_results.csv").exists()
    measured = res[res["vectors"] == "measured"].iloc[0]
    assert measured["param_status"] == "ok" and measured["param_p_value"] < 0.05
    between = res[res["vectors"].str.startswith("between_patient_r")]
    assert len(between) == 2 and between["value"].min() > 0.9 and between["param_pillai"].isna().all()


def test_categorical_stage_with_unused_level_does_not_crash():
    """A categorical stage column carrying a level no patient has must not reach statsmodels as a zero column."""
    import pandas as pd
    from mmprog.evaluation.manova import manova_by_stage, permutation_manova
    rng = np.random.default_rng(1)
    rows = []
    for g, n in (("control", 6), ("precursor", 6), ("myeloma", 8)):
        shift = {"control": 0.0, "precursor": 0.5, "myeloma": 1.5}[g]
        for _ in range(n):
            rows.append(dict(patient=len(rows), stage=g, a=rng.normal() + shift, b=rng.normal(), c=rng.normal() - shift))
    tab = pd.DataFrame(rows)
    tab["stage"] = pd.Categorical(tab["stage"], categories=["control", "precursor", "myeloma", "relapse"])
    par = manova_by_stage(tab, ["a", "b", "c"])
    perm = permutation_manova(tab, ["a", "b", "c"], n_perm=99, seed=0)
    assert par["status"] == "ok" and perm["status"] == "ok" and par["n_groups"] == 3
