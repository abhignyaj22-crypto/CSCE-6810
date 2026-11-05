"""Pathway annotation of a gene backbone by hypergeometric enrichment.

For each (program backbone, curated gene set) pair the test asks whether the overlap is
larger than expected when drawing ``len(backbone)`` genes at random from the panel
(the measured panel genes, those passing gene_mask, are the background, never the genome). P-values are
corrected over all tests with Benjamini-Hochberg. The ranked variant of this idea is
GSEA (Subramanian et al., PNAS 2005); the backbone is an unordered set, so the
hypergeometric test is the matching statistic.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import say, section, show_table


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def hypergeometric_enrichment(backbone_genes_by_program: Dict[str, List[str]], gene_sets: Dict[str, List[str]],
                              background: List[str]) -> pd.DataFrame:
    section("Pathway annotation: hypergeometric enrichment of each backbone against the curated sets")
    bg = set(background)
    M = len(bg)
    rows = []
    for program, genes in backbone_genes_by_program.items():
        drawn = set(genes) & bg
        for set_name, members in gene_sets.items():
            inset = set(members) & bg
            k = len(drawn & inset)
            p = float(stats.hypergeom.sf(k - 1, M, len(inset), len(drawn))) if drawn and inset else 1.0
            rows.append(dict(program=program, gene_set=set_name, backbone_size=len(drawn), set_size_in_panel=len(inset),
                             overlap=k, p_value=p, overlapping_genes=", ".join(sorted(drawn & inset))))
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["q_value"] = benjamini_hochberg(tab["p_value"].to_numpy())
        tab = tab.sort_values(["program", "q_value"]).reset_index(drop=True)
    show_table(tab.drop(columns=["overlapping_genes"]) if len(tab) else tab, max_rows=60, title=f"enrichment (background {M} panel genes):")
    for program in backbone_genes_by_program:
        sub = tab[tab["program"] == program]
        if len(sub):
            best = sub.iloc[0]
            say(f"{program}: best-matching set {best['gene_set']} (overlap {best['overlap']}, q = {best['q_value']:.3g})")
    return tab
