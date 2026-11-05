"""Gene backbone of each program by stability selection over patient subsamples.

Stability selection (Meinshausen and Buhlmann, 2010, Journal of the Royal Statistical
Society B 72(4):417-473) declares a variable selected when it is chosen in a large
fraction of refits on random subsamples. Here the refits are anchored NMF fits on
random halves of the patients (the patient, not the cell, is the unit that varies), a
gene is "selected" in a refit when its weight in the program exceeds ``threshold`` of
that program's largest weight, and the backbone is the set of genes selected in at
least ``min_fold_fraction`` of the refits. Seeded programs are aligned by name; extra
programs are aligned to the full fit by maximum cosine similarity.

Each backbone gene carries its class and evidence grade from
``resources/gene_annotations.json`` (D24), so a gene whose literature evidence points
toward drug sensitivity is flagged in the same table that names it.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..logging_utils import say, section, show_table
from ..resources import load_gene_annotations


def _align_extra(H_sub: pd.DataFrame, H_full: pd.DataFrame) -> pd.DataFrame:
    """Rename the extra_* rows of a refit to the full-fit extra program they resemble most."""
    extras_full = [p for p in H_full.index if p.startswith("extra_")]
    extras_sub = [p for p in H_sub.index if p.startswith("extra_")]
    if not extras_full or not extras_sub:
        return H_sub
    cols = H_full.columns.intersection(H_sub.columns)
    A = H_full.loc[extras_full, cols].to_numpy()
    B = H_sub.loc[extras_sub, cols].to_numpy()
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = B @ A.T
    mapping = {}
    used = set()
    for i in np.argsort(-sim.max(axis=1)):          # greedy, best matches first
        j = int(np.argmax([sim[i, k] if k not in used else -np.inf for k in range(len(extras_full))]))
        mapping[extras_sub[i]] = extras_full[j]
        used.add(j)
    return H_sub.rename(index=mapping)


def stability_selection(fit_fn: Callable[[np.ndarray], pd.DataFrame], obs: pd.DataFrame, H_full: pd.DataFrame,
                        n_subsamples: int = 20, patient_fraction: float = 0.5, threshold: float = 0.05,
                        min_fold_fraction: float = 0.8, seed: int = 0,
                        annotations: Optional[dict] = None) -> pd.DataFrame:
    """Return one row per (program, gene) in the backbone with its selection fraction and annotation."""
    section(f"Gene backbone: stability selection over {n_subsamples} patient subsamples")
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(pd.unique(obs["patient"].astype(str))))
    n_pick = max(2, int(round(len(patients) * patient_fraction)))
    counts = pd.DataFrame(0.0, index=H_full.index, columns=H_full.columns)
    for b in range(n_subsamples):
        chosen = rng.choice(patients, n_pick, replace=False)
        idx = np.where(obs["patient"].astype(str).isin(chosen))[0]
        H_sub = _align_extra(fit_fn(idx), H_full).reindex(index=H_full.index, columns=H_full.columns, fill_value=0.0)
        rel = H_sub.to_numpy() / (H_sub.to_numpy().max(axis=1, keepdims=True) + 1e-12)
        counts += (rel >= threshold).astype(float)
        say(f"subsample {b + 1}/{n_subsamples}: patients {list(chosen)}, {len(idx)} cells, "
            f"genes above threshold per program {dict(zip(H_full.index, (rel >= threshold).sum(axis=1)))}", indent=1)
    fraction = counts / n_subsamples
    ann = annotations if annotations is not None else load_gene_annotations()
    flat = {g: a for prog in ann.values() for g, a in prog.items()}
    rel_full = H_full.to_numpy() / (H_full.to_numpy().max(axis=1, keepdims=True) + 1e-12)
    rows = []
    for pi, program in enumerate(H_full.index):
        sel = fraction.loc[program]
        genes = sel[sel >= min_fold_fraction].sort_values(ascending=False)
        for g, f in genes.items():
            a = flat.get(g, {})
            rows.append(dict(program=program, gene=g, selection_fraction=float(f),
                             weight_relative_to_max=float(rel_full[pi, H_full.columns.get_loc(g)]),
                             gene_class=a.get("class", ""), evidence_grade=a.get("evidence_grade", ""),
                             is_wnt_component=a.get("is_wnt_component", ""),
                             note=a.get("mm_evidence", "")[:120]))
        say(f"program {program}: backbone of {len(genes)} genes (selected in >= {min_fold_fraction:.0%} of refits)")
    backbone = pd.DataFrame(rows, columns=["program", "gene", "selection_fraction", "weight_relative_to_max",
                                           "gene_class", "evidence_grade", "is_wnt_component", "note"])
    show_table(backbone.drop(columns=["note"]), max_rows=60, title="gene backbone with evidence grades:")
    return backbone
