"""BPE motif discovery: merge order, encode consistency, planted-motif enrichment, speed."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.analysis.motifs import (NeighbourhoodBPE, motif_enrichment,  # noqa: E402
                                    motif_indicator, run_motif_stage)


def test_bpe_merges_most_frequent_pair_first_and_encodes_consistently():
    corpus = [["plasma", "plasma", "stromal"]] * 30 + [["T", "NK", "B"]] * 10 + [["plasma", "T", "B"]] * 5
    bpe = NeighbourhoodBPE(n_merges=3, min_pair_count=2).fit(corpus)
    assert bpe.merges[0] == ("plasma", "plasma")
    assert bpe.vocabulary()[0] == "plasma|plasma"
    # the second merge absorbs the first, so the full sentence becomes one token
    assert bpe.merges[1] == ("plasma|plasma", "stromal")
    assert bpe.encode(["plasma", "plasma", "stromal"]) == ["plasma|plasma|stromal"]
    assert bpe.encode(["plasma", "plasma", "plasma", "plasma"]) == ["plasma|plasma", "plasma|plasma"]
    # vectorised encode agrees with the per-sentence encode
    assert bpe.encode_all(corpus) == [bpe.encode(s) for s in corpus]
    # untouched sentence passes through
    assert bpe.encode(["erythroid", "monocyte"]) == ["erythroid", "monocyte"]


def test_enrichment_finds_planted_motif():
    rng = np.random.default_rng(0)
    n = 4000
    has = rng.uniform(size=n) < 0.3
    sentences = [["plasma", "plasma", "stromal"] if h else list(rng.choice(["T", "B", "NK", "monocyte"], 3)) for h in has]
    bpe = NeighbourhoodBPE(n_merges=5, min_pair_count=20).fit(sentences)
    vocab = bpe.vocabulary()
    ind = motif_indicator(bpe.encode_all(sentences), vocab)
    score = rng.normal(size=n) + 1.0 * has
    groups = np.repeat(["P0", "P1", "P2", "P3"], n // 4)
    df = motif_enrichment(ind, vocab, score, groups, n_perm=100, seed=0)
    top = df.iloc[0]
    assert top["motif"].startswith("plasma|plasma")
    assert top["difference"] > 0.7 and top["perm_p"] < 0.02
    assert top["frac_patients_consistent"] == 1.0


def test_stage_runner_writes_artefacts(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    xy = rng.uniform(0, 1000, size=(n, 2))
    blob = np.linalg.norm(xy - 500, axis=1) < 200
    lineage = np.where(blob, "plasma", rng.choice(["T", "B", "stromal"], n))
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sample": np.where(xy[:, 0] < 500, "S0", "S1"),
                        "patient": np.where(xy[:, 0] < 500, "S0", "S1"), "lineage_name": lineage,
                        "prog_wnt_camdr_nb_z": rng.normal(size=n) + 1.5 * blob})

    class Run:
        def art(self, name):
            return str(tmp_path / {"motifs": "motif_enrichment.csv", "motif_vocab": "motif_vocabulary.json"}.get(name, name))

    class Cfg:
        knn_k = 6
        seed = 0

    df = run_motif_stage(obs, "prog_wnt_camdr_nb_z", Cfg(), Run())
    assert (tmp_path / "motif_enrichment.csv").exists() and (tmp_path / "motif_vocabulary.json").exists()
    assert df.iloc[0]["motif"].startswith("plasma|plasma") and df.iloc[0]["difference"] > 0


def test_speed_on_50k_sentences():
    rng = np.random.default_rng(2)
    toks = np.array(["plasma", "T", "B", "NK", "stromal", "monocyte", "erythroid"])
    sentences = [list(rng.choice(toks, 7, p=[0.4, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1])) for _ in range(50_000)]
    t0 = time.time()
    bpe = NeighbourhoodBPE(n_merges=30, min_pair_count=50).fit(sentences)
    encoded = bpe.encode_all(sentences)
    ind = motif_indicator(encoded, bpe.vocabulary())
    elapsed = time.time() - t0
    assert len(bpe.merges) == 30 and ind.shape == (50_000, 30)
    assert elapsed < 30, f"took {elapsed:.1f}s"


def test_motif_counted_inside_longer_merged_token():
    """plasma|plasma is counted for a cell whose pair was absorbed into a four-plasma token."""
    from mmprog.analysis.motifs import NeighbourhoodBPE, motif_indicator
    sentences = [["plasma", "plasma", "plasma", "plasma", "T"]] * 60 + [["T", "B", "NK", "T", "B"]] * 60
    bpe = NeighbourhoodBPE(n_merges=3, min_pair_count=10).fit(sentences)
    vocab = bpe.vocabulary()
    assert "plasma|plasma" in vocab
    ind = motif_indicator(bpe.encode_all(sentences), vocab)
    j = vocab.index("plasma|plasma")
    assert ind[:60, j].all() and not ind[60:, j].any()
