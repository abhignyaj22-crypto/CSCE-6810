"""Neighbourhood motif discovery with byte-pair encoding over lineage sequences.

Each cell is written as a short "sentence": its own lineage followed by the
lineages of its k nearest same-slide neighbours, ordered by increasing distance.
Byte-pair encoding (Gage, 1994; Sennrich et al., 2016) repeatedly merges the most
frequent adjacent token pair in the corpus into a new token, so recurring
multi-cell arrangements such as ``plasma|plasma|stromal`` become single
vocabulary items without anyone naming them in advance.

This is an exploratory heuristic, analogous to cellular-neighbourhood analysis
(Schurch et al., 2020, Cell), not a statistical model of tissue. The ordering by
distance makes a motif an ordered co-occurrence, and the merge order depends on
corpus frequency, so motifs are read as descriptive labels. The enrichment test
treats the patient as the unit of replication by permuting scores within patient.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..logging_utils import say, section, show_table

SEP = "|"


def build_sentences(obs: pd.DataFrame, nb_table: np.ndarray) -> List[List[str]]:
    """Own lineage first, then the k neighbours in the (distance-ordered) kNN table."""
    names = obs["lineage_name"].astype(str).to_numpy()
    sentences = [[names[i]] + list(names[nb_table[i]]) for i in range(len(obs))]
    say(f"built {len(sentences)} neighbourhood sentences of length {nb_table.shape[1] + 1}")
    return sentences


class NeighbourhoodBPE:
    """Byte-pair encoding over integer-coded lineage sequences.

    ``fit`` learns an ordered list of merges; ``encode`` applies them in that order.
    Pair counting is vectorised: every sentence is a row of an integer matrix, so
    one merge round is a handful of numpy operations rather than a Python loop
    over 200k sentences.
    """

    def __init__(self, n_merges: int = 30, min_pair_count: int = 50):
        self.n_merges = n_merges
        self.min_pair_count = min_pair_count
        self.merges: List[Tuple[str, str]] = []
        self.merge_counts: List[int] = []
        self._vocab: Dict[str, int] = {}
        self._names: List[str] = []

    # ---- integer coding helpers
    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._names)
            self._names.append(token)
        return self._vocab[token]

    def _encode_matrix(self, sentences: Sequence[Sequence[str]]) -> np.ndarray:
        """(N, L) int matrix padded with -1; sentences may shrink as merges apply."""
        L = max(len(s) for s in sentences)
        mat = np.full((len(sentences), L), -1, dtype=np.int64)
        for i, s in enumerate(sentences):
            mat[i, : len(s)] = [self._id(t) for t in s]
        return mat

    @staticmethod
    def _pair_counts(mat: np.ndarray) -> Counter:
        left, right = mat[:, :-1], mat[:, 1:]
        valid = (left >= 0) & (right >= 0)
        if not valid.any():
            return Counter()
        key = left[valid] * (mat.max() + 1) + right[valid]
        ids, counts = np.unique(key, return_counts=True)
        base = mat.max() + 1
        return Counter({(int(k // base), int(k % base)): int(c) for k, c in zip(ids, counts)})

    @staticmethod
    def _apply_merge(mat: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
        """Replace every non-overlapping adjacent (a, b) by new_id and re-pack rows to the left."""
        hit = (mat[:, :-1] == a) & (mat[:, 1:] == b)
        if not hit.any():
            return mat
        # resolve overlaps such as a,b,a,b handled left to right: a hit at j cancels a hit at j+1
        for j in range(1, hit.shape[1]):
            hit[:, j] &= ~hit[:, j - 1]
        out = mat.copy()
        out[:, :-1][hit] = new_id
        drop = np.zeros_like(mat, dtype=bool)
        drop[:, 1:][hit] = True
        keep = ~drop
        # left-pack: stable sort by "dropped" within each row keeps token order
        order = np.argsort(drop, axis=1, kind="stable")
        packed = np.take_along_axis(out, order, axis=1)
        counts = keep.sum(axis=1)
        mask = np.arange(mat.shape[1])[None, :] < counts[:, None]
        packed[~mask] = -1
        return packed

    def fit(self, sentences: Sequence[Sequence[str]]) -> "NeighbourhoodBPE":
        t0 = time.time()
        mat = self._encode_matrix(sentences)
        say(f"BPE: {len(sentences)} sentences, {len(self._names)} base tokens, up to {self.n_merges} merges "
            f"(min pair count {self.min_pair_count})")
        for step in range(self.n_merges):
            counts = self._pair_counts(mat)
            if not counts:
                break
            (a, b), c = counts.most_common(1)[0]
            if c < self.min_pair_count:
                say(f"stop at merge {step}: best pair count {c} below {self.min_pair_count}", indent=1)
                break
            new_token = self._names[a] + SEP + self._names[b]
            new_id = self._id(new_token)
            self.merges.append((self._names[a], self._names[b]))
            self.merge_counts.append(c)
            mat = self._apply_merge(mat, a, b, new_id)
            say(f"merge {step + 1:>2}: {self._names[a]} + {self._names[b]} -> {new_token}  ({c} occurrences)", indent=1)
        say(f"BPE fit: {len(self.merges)} merges in {time.time() - t0:.1f}s")
        return self

    def encode(self, sentence: Sequence[str]) -> List[str]:
        """Apply the learned merges, in order, to one sentence."""
        toks = list(sentence)
        for a, b in self.merges:
            merged: List[str] = []
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                    merged.append(a + SEP + b)
                    i += 2
                else:
                    merged.append(toks[i])
                    i += 1
            toks = merged
        return toks

    def encode_all(self, sentences: Sequence[Sequence[str]]) -> List[List[str]]:
        """Vectorised encode of a corpus (same result as ``encode`` per sentence)."""
        mat = self._encode_matrix(sentences)
        for a, b in self.merges:
            mat = self._apply_merge(mat, self._vocab[a], self._vocab[b], self._vocab[a + SEP + b])
        return [[self._names[t] for t in row if t >= 0] for row in mat]

    def vocabulary(self) -> List[str]:
        """Merged tokens only (the motifs), in merge order."""
        return [a + SEP + b for a, b in self.merges]


def motif_indicator(encoded_sentences: Sequence[Sequence[str]], vocab: Sequence[str]) -> np.ndarray:
    """(N, V) bool: does sentence i contain motif v as a contiguous run of base lineages.

    Each encoded sentence is expanded back to its base lineages (a merged token
    ``a|b|c`` is the run ``a, b, c``) and a motif counts when its lineage sequence
    occurs contiguously anywhere in that run. Testing the base run rather than the
    surviving tokens means ``plasma|plasma`` is still counted for a cell whose pair
    was later absorbed into ``plasma|plasma|plasma|plasma``, so motif counts do not
    depend on which later merges happened.
    """
    motifs = [tuple(v.split("|")) for v in vocab]
    out = np.zeros((len(encoded_sentences), len(vocab)), dtype=bool)
    for i, toks in enumerate(encoded_sentences):
        base = tuple(part for t in toks for part in t.split("|"))
        n = len(base)
        for j, motif in enumerate(motifs):
            L = len(motif)
            if L > n:
                continue
            for start in range(n - L + 1):
                if base[start:start + L] == motif:
                    out[i, j] = True
                    break
    return out


def motif_enrichment(indicator: np.ndarray, vocab: Sequence[str], score: np.ndarray, groups: np.ndarray,
                     n_perm: int = 200, seed: int = 0, min_cells_per_patient: int = 20) -> pd.DataFrame:
    """Mean score with vs without each motif, with a within-patient permutation p-value.

    Scores are permuted inside each patient so that between-patient differences
    in composition and in program level cannot create a spurious enrichment.
    """
    rng = np.random.default_rng(seed)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups)
    patients = np.unique(groups)
    patient_idx = [np.where(groups == p)[0] for p in patients]

    def diff_for(s: np.ndarray) -> np.ndarray:
        with_sum = indicator.T @ s
        with_n = indicator.sum(axis=0)
        total = s.sum()
        n = len(s)
        mean_with = np.where(with_n > 0, with_sum / np.maximum(with_n, 1), np.nan)
        mean_without = np.where(n - with_n > 0, (total - with_sum) / np.maximum(n - with_n, 1), np.nan)
        return mean_with - mean_without, mean_with, mean_without

    observed, mean_with, mean_without = diff_for(score)
    null = np.empty((n_perm, len(vocab)))
    for k in range(n_perm):
        s = score.copy()
        for idx in patient_idx:
            s[idx] = s[idx][rng.permutation(len(idx))]
        null[k] = diff_for(s)[0]
    p = (np.sum(np.abs(null) >= np.abs(observed)[None, :], axis=0) + 1) / (n_perm + 1)

    # per-patient consistency: in how many patients with enough motif cells is with > without
    n_pat_with, n_pat_consistent = np.zeros(len(vocab), int), np.zeros(len(vocab), int)
    for idx in patient_idx:
        ind_p, s_p = indicator[idx], score[idx]
        cnt = ind_p.sum(axis=0)
        enough = cnt >= min_cells_per_patient
        for j in np.where(enough & (cnt < len(idx)))[0]:
            n_pat_with[j] += 1
            if s_p[ind_p[:, j]].mean() > s_p[~ind_p[:, j]].mean():
                n_pat_consistent[j] += 1

    df = pd.DataFrame({
        "motif": list(vocab), "n_cells": indicator.sum(axis=0), "mean_with": mean_with,
        "mean_without": mean_without, "difference": observed, "perm_p": p,
        "n_patients_with_motif": n_pat_with,
        "frac_patients_consistent": np.where(n_pat_with > 0, n_pat_consistent / np.maximum(n_pat_with, 1), np.nan),
    })
    df = df.reindex(df["difference"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    show_table(df.head(10).round(4), title="top motifs by |difference| (within-patient permutation p):")
    return df


def run_motif_stage(obs: pd.DataFrame, main_target: str, cfg, run,
                    nb_table: np.ndarray | None = None) -> pd.DataFrame:
    """Sentences -> BPE vocabulary -> enrichment against the measured main target; writes the artefacts."""
    section(f"S5c neighbourhood motifs (BPE over lineage sequences) vs {main_target}")
    from ..models.neighbours import knn_table

    if nb_table is None:
        nb_table = knn_table(obs, cfg.knn_k)
    sentences = build_sentences(obs, nb_table)
    bpe = NeighbourhoodBPE(n_merges=30, min_pair_count=max(50, len(obs) // 2000)).fit(sentences)
    vocab = bpe.vocabulary()
    if not vocab:
        say("no motif reached the minimum pair count; nothing to test")
        return pd.DataFrame()
    encoded = bpe.encode_all(sentences)
    indicator = motif_indicator(encoded, vocab)
    enrich = motif_enrichment(indicator, vocab, obs[main_target].to_numpy(), obs["patient"].to_numpy(),
                              seed=cfg.seed)
    enrich.to_csv(run.art("motifs"), index=False)
    with open(run.art("motif_vocab"), "w") as fh:
        json.dump({"merges": bpe.merges, "counts": bpe.merge_counts, "vocabulary": vocab}, fh, indent=2)

    top = enrich.iloc[0]
    direction = "enriched" if top["difference"] > 0 else "depleted"
    n_cons = int(round(top["frac_patients_consistent"] * top["n_patients_with_motif"])) if top["n_patients_with_motif"] else 0
    say(f"interpretation: the {top['motif']} motif is {direction} for {main_target} "
        f"(difference {top['difference']:+.3f} z, permutation p = {top['perm_p']:.3g}) in "
        f"{n_cons}/{int(top['n_patients_with_motif'])} patients with at least 20 motif cells")
    say(f"motif enrichment -> {run.art('motifs')}; vocabulary -> {run.art('motif_vocab')}")
    return enrich
