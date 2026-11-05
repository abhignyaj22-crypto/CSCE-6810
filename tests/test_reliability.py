import numpy as np
import pandas as pd

from mmprog.config import Config
from mmprog.targets import reliability_lookup, reliability_summary, reliability_table, split_half


def test_split_half_conserves_counts():
    rng = np.random.default_rng(0)
    cnt = rng.poisson(5, 200)
    total = cnt + rng.poisson(50, 200)
    (a, ta), (b, tb) = split_half(cnt, total, rng)
    assert np.array_equal(a + b, cnt)
    assert np.array_equal(ta + tb, total)
    assert (a <= ta).all() and (b <= tb).all()


def _obs(structured: bool, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    total = rng.poisson(100, n)
    if structured:
        rate = np.where(np.arange(n) < n // 2, 0.30, 0.02)   # two populations with very different program fractions
    else:
        rate = np.full(n, 0.05)                              # program counts carry no cell-to-cell structure
    cnt = rng.binomial(total, rate)
    obs = pd.DataFrame(dict(sample="s1", cnt_total=total, cnt_prog=cnt))
    # a 20-cell neighbourhood sum (cells grouped in blocks) raises counts and so raises reliability
    block = np.arange(n) // 20
    obs["nb_cnt_total"] = pd.Series(total).groupby(block).transform("sum").to_numpy()
    obs["nb_cnt_prog"] = pd.Series(cnt).groupby(block).transform("sum").to_numpy()
    return obs


def test_null_reliability_near_zero_and_planted_high():
    cfg = Config()
    tab_null = reliability_table(_obs(False), {"prog": ["x", "y", "z"]}, cfg)
    tab_plant = reliability_table(_obs(True), {"prog": ["x", "y", "z"]}, cfg)
    null = tab_null.set_index("target")["r_sb"]
    plant = tab_plant.set_index("target")["r_sb"]
    assert abs(null["prog_prog_cell_z"]) < 0.15
    assert plant["prog_prog_cell_z"] > 0.6
    assert plant["prog_prog_nb_z"] > 0.95
    summary = reliability_summary(tab_plant)
    assert "ceiling_sqrt" in summary.columns
    assert summary.loc["prog_prog_nb_z", "ceiling_sqrt"] > 0.97
    look = reliability_lookup(tab_plant)
    assert ("prog_prog_nb_z", "s1") in look
