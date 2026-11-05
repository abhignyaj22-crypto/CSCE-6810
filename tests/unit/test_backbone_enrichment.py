"""Stability selection recovers the planted anchors; enrichment ranks the matching set first."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmprog.interpretation.backbone import stability_selection  # noqa: E402
from mmprog.interpretation.enrichment import benjamini_hochberg, hypergeometric_enrichment  # noqa: E402
from mmprog.programs.anchored_nmf import AnchoredNMF  # noqa: E402


def _toy(seed=0, n_cells=600, n_patients=6):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(60)]
    anchors = {"wnt": genes[:8], "stress": genes[20:26]}
    H_true = np.zeros((3, 60))
    H_true[0, :8] = 1
    H_true[1, 20:26] = 1
    H_true[2, 40:50] = 1
    W = rng.uniform(0, 2, size=(n_cells, 3))
    X = W @ H_true + rng.uniform(0, 0.05, size=(n_cells, 60))
    obs = pd.DataFrame({"patient": [f"P{i % n_patients}" for i in range(n_cells)]})
    return X, genes, anchors, obs


def test_stability_selection_recovers_anchors():
    X, genes, anchors, obs = _toy()
    full = AnchoredNMF(n_extra=1, max_iter=80, seed=0, verbose_every=100).fit(X, genes, anchors)

    def fit_fn(idx):
        return AnchoredNMF(n_extra=1, max_iter=80, seed=1, verbose_every=100).fit(X[idx], genes, anchors).H_

    bb = stability_selection(fit_fn, obs, full.H_, n_subsamples=6, patient_fraction=0.5, threshold=0.2,
                             min_fold_fraction=0.8, seed=0, annotations={})
    wnt = bb[bb["program"] == "wnt"]
    assert set(anchors["wnt"]) <= set(wnt["gene"])
    assert (wnt.set_index("gene").loc[anchors["wnt"], "selection_fraction"] == 1.0).all()
    extra = bb[bb["program"] == "extra_0"]
    assert set(genes[40:50]) <= set(extra["gene"])


def test_enrichment_ranks_matching_set_first():
    genes = [f"g{i}" for i in range(60)]
    sets = {"wnt": genes[:8], "stress": genes[20:26], "cyto": genes[30:40]}
    backbone = {"wnt": genes[:7] + ["g55"], "stress": genes[20:25]}
    tab = hypergeometric_enrichment(backbone, sets, genes)
    best = tab.sort_values(["program", "q_value"]).groupby("program").head(1).set_index("program")["gene_set"]
    assert best["wnt"] == "wnt" and best["stress"] == "stress"
    assert tab["q_value"].between(0, 1).all()


def test_benjamini_hochberg_monotone():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q.min() >= 0.01 and q.max() <= 1 and q[3] == 0.5
