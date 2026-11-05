"""CODEX / PhenoCycler spatial proteomics: loading, marker matching, lineage calls.

Why this module exists
----------------------
A CODEX run images an FFPE trephine section with a DAPI nuclear channel plus 30
to 60 protein markers, cell by cell. The morphology model in this repository is
trained on DAPI, so it can be applied directly to the CODEX DAPI channel. The
protein markers then give an independent, transcript-free check of two things:

1. the composition head: predicted lineage proportions versus protein-defined
   lineage proportions on the same section;
2. whether predicted program scores differ between protein-defined
   plasma-cell-dense niches (CD138+ cells within ``cfg.nb_radius_um``) and the
   rest of the marrow.

No public dataset currently pairs CODEX with Xenium on the same myeloma
trephine, so this is a data-gated stage: it runs only when ``cfg.codex_table``
and ``cfg.codex_dapi`` are set. A synthetic fixture (``synthetic_codex.py``)
exercises every function in the tests.

Lineage calls use the same marker-score argmax idea as the RNA lineage labels
(``targets.programs.score_lineages``) so the two labelings are comparable.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from ..config import RESOURCE_DIR, Config
from ..logging_utils import say, show_table

# Column names commonly used for centroids in CODEX / PhenoCycler / QuPath exports.
_X_CANDIDATES = ["x", "X", "x_centroid", "centroid_x", "X_cent", "x_cent", "Centroid X", "Centroid X µm",
                 "cx", "x_coord", "X_centroid", "x:x", "nucleus_x", "Centroid_X"]
_Y_CANDIDATES = ["y", "Y", "y_centroid", "centroid_y", "Y_cent", "y_cent", "Centroid Y", "Centroid Y µm",
                 "cy", "y_coord", "Y_centroid", "y:y", "nucleus_y", "Centroid_Y"]
_ID_CANDIDATES = ["cell_id", "CellID", "cell", "id", "Object ID", "label", "cellLabel", "cell_label"]
_NON_MARKER_HINTS = {"area", "size", "perimeter", "eccentricity", "solidity", "region", "tile", "fov",
                     "sample", "patient", "image", "cluster", "celltype", "cell_type", "phenotype", "class"}


def load_marker_mapping(path: Optional[str] = None) -> dict:
    """Read ``resources/codex_markers.json`` (or another mapping with the same layout)."""
    path = path or os.path.join(RESOURCE_DIR, "codex_markers.json")
    with open(path) as fh:
        mapping = json.load(fh)
    say(f"CODEX marker mapping: {len(mapping['lineages'])} lineages, {len(mapping['aliases'])} markers with aliases")
    return mapping


def _find_column(columns: List[str], candidates: List[str], user_choice: Optional[str]) -> Optional[str]:
    if user_choice is not None:
        if user_choice not in columns:
            raise KeyError(f"column {user_choice!r} not in table; columns start with {columns[:10]}")
        return user_choice
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_codex_table(path: str, x_col: Optional[str] = None, y_col: Optional[str] = None,
                     um_per_px_codex: Optional[float] = None) -> pd.DataFrame:
    """Read a cell-by-marker table and return ``cell_id, x_um, y_um`` plus the marker columns.

    ``x`` and ``y`` are auto-detected from common export names unless given. When
    the coordinates are in pixels, pass ``um_per_px_codex`` and they are converted
    to microns; the decision is printed.
    """
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    cols = list(table.columns)
    say(f"CODEX table {os.path.basename(path)}: {len(table)} cells x {len(cols)} columns")
    xc = _find_column(cols, _X_CANDIDATES, x_col)
    yc = _find_column(cols, _Y_CANDIDATES, y_col)
    if xc is None or yc is None:
        raise KeyError(f"could not find x/y columns; pass x_col and y_col. Columns: {cols[:20]}")
    idc = _find_column(cols, _ID_CANDIDATES, None)
    scale = 1.0 if um_per_px_codex is None else float(um_per_px_codex)
    say(f"coordinates from columns {xc!r}, {yc!r}; "
        + ("treated as microns" if um_per_px_codex is None else f"converted from pixels at {scale} um/px"))

    out = pd.DataFrame({
        "cell_id": table[idc].astype(str).values if idc else [f"c{i}" for i in range(len(table))],
        "x_um": table[xc].astype(np.float32).values * scale,
        "y_um": table[yc].astype(np.float32).values * scale,
    })
    drop = {xc, yc} | ({idc} if idc else set())
    marker_cols = []
    for c in cols:
        if c in drop or not pd.api.types.is_numeric_dtype(table[c]):
            continue
        if any(h in c.lower() for h in _NON_MARKER_HINTS):
            continue
        marker_cols.append(c)
    for c in marker_cols:
        out[c] = table[c].astype(np.float32).values
    say(f"kept {len(marker_cols)} numeric marker columns; dropped {len(cols) - len(marker_cols) - len(drop)} "
        "non-marker columns (geometry, ids, annotations)")
    return out


def match_markers(columns: List[str], mapping: dict) -> Dict[str, str]:
    """Map canonical marker names to table columns, case-insensitively and through aliases.

    Returns ``{canonical: column}`` for the markers found and prints both the
    matched and the unmatched ones so a panel mismatch is visible at once.
    """
    lower = {c.lower(): c for c in columns}
    matched: Dict[str, str] = {}
    for canonical, aliases in mapping["aliases"].items():
        for name in [canonical] + list(aliases):
            key = name.lower()
            if key in lower:
                matched[canonical] = lower[key]
                break
    needed = {m for lin in mapping["lineages"].values() for m in lin["positive"] + lin["negative"]}
    missing = sorted(needed - set(matched))
    say(f"matched {len(matched)} markers: " + ", ".join(f"{k}<-{v}" for k, v in sorted(matched.items())))
    if missing:
        say(f"unmatched markers (lineage scores will use the remaining ones): {missing}")
    return matched


def _transform(values: np.ndarray) -> np.ndarray:
    # arcsinh with cofactor 5 is the standard CODEX / CyTOF intensity transform; z-scoring per
    # marker makes positive and negative marker averages comparable across antibodies.
    v = np.arcsinh(values / 5.0)
    return (v - v.mean(0)) / (v.std(0) + 1e-6)


def assign_lineage_from_markers(table: pd.DataFrame, mapping: dict, matched: Dict[str, str]) -> pd.Series:
    """Argmax lineage per cell from protein markers; prints the composition table."""
    canon = [m for m in matched]
    Z = _transform(table[[matched[m] for m in canon]].values.astype(np.float32))
    zcol = {m: Z[:, i] for i, m in enumerate(canon)}
    scores = {}
    for lin, spec in mapping["lineages"].items():
        pos = [m for m in spec["positive"] if m in zcol]
        neg = [m for m in spec["negative"] if m in zcol]
        if not pos:
            say(f"lineage {lin}: none of its positive markers {spec['positive']} in the panel, skipped", indent=1)
            continue
        s = np.mean([zcol[m] for m in pos], axis=0)
        if neg:
            s = s - np.mean([zcol[m] for m in neg], axis=0)
        scores[lin] = s
    if not scores:
        raise ValueError("no lineage could be scored from the matched markers")
    names = list(scores)
    S = np.stack([scores[n] for n in names], axis=1)
    lineage = pd.Series(np.array(names)[S.argmax(1)], index=table.index, name="lineage_name")
    comp = lineage.value_counts().rename("cells").to_frame()
    comp["fraction"] = (comp["cells"] / len(lineage)).round(3)
    show_table(comp, title="protein-defined composition:")
    return lineage


def codex_to_obs(table: pd.DataFrame, lineage: pd.Series, cfg: Config, sample: str = "CODEX",
                 mapping: Optional[dict] = None, matched: Optional[Dict[str, str]] = None,
                 min_plasma: int = 5) -> pd.DataFrame:
    """Obs-like frame in working-level pixels with a protein-defined plasma-dense niche flag.

    ``plasma_dense`` is True when at least ``min_plasma`` CD138+ cells (protein
    lineage ``plasma``, or the CD138 marker above its 90th percentile when the
    mapping is given) lie within ``cfg.nb_radius_um`` of the cell, the same radius
    the transcript neighbourhood targets use.
    """
    px_per_um = cfg.px_per_um
    obs = pd.DataFrame({
        "cell_id": table["cell_id"].values,
        "x": (table["x_um"].values * px_per_um).astype(np.float32),
        "y": (table["y_um"].values * px_per_um).astype(np.float32),
        "sample": sample, "patient": sample,
        "lineage_name": lineage.values,
    })
    is_plasma = obs["lineage_name"].values == "plasma"
    if mapping is not None and matched is not None and mapping.get("plasma_marker") in matched:
        col = matched[mapping["plasma_marker"]]
        thr = np.percentile(table[col].values, 90)
        is_plasma = is_plasma | (table[col].values >= thr)
        say(f"CD138 positivity: lineage call OR {col} >= 90th percentile ({thr:.1f})", indent=1)
    xy_um = table[["x_um", "y_um"]].values.astype(np.float32)
    tree = KDTree(xy_um[is_plasma]) if is_plasma.any() else None
    if tree is None:
        n_plasma_near = np.zeros(len(obs), dtype=int)
    else:
        n_plasma_near = tree.query_radius(xy_um, r=cfg.nb_radius_um, count_only=True)
        n_plasma_near = n_plasma_near - is_plasma.astype(int)   # a CD138+ cell does not count itself
    obs["n_plasma_in_radius"] = n_plasma_near
    obs["plasma_dense"] = n_plasma_near >= min_plasma
    say(f"{sample}: {len(obs)} cells, {int(is_plasma.sum())} CD138+ ({is_plasma.mean():.1%}); "
        f"{obs['plasma_dense'].mean():.1%} of cells lie in plasma-dense niches "
        f"(>= {min_plasma} CD138+ cells within {cfg.nb_radius_um} um)")
    return obs
