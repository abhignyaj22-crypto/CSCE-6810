"""Spatial fidelity: does the predicted map reproduce the measured spatial structure?

A per-cell correlation says nothing about whether the predicted map has the right
spatial organisation. Two checks, per patient, on the out-of-fold predictions:

1. Moran's I of the predicted and of the measured target on the same k-nearest-neighbour
   graph (binary weights, the graph the model's neighbour tokens use). A faithful map
   has an autocorrelation close to the measured one; a map that is spatially smoother or
   noisier than the tissue shows up as a ratio far from 1.
2. Plasma-cell-dense niche contrast: the difference of each target between cells that
   sit in a plasma-cell-dense neighbourhood (at least ``min_plasma`` plasma cells within
   ``nb_radius_um``, the rule the protein bridge uses) and the rest, as Cohen's d, for the
   predicted and the measured values side by side. The Ohlstrom et al. (2026, preprint)
   niche is plasma-cell dense, so the main target's measured contrast is expected to be
   positive on real data, and the predicted contrast should have the same sign and a
   comparable size.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from ..evaluation.codex_validation import niche_contrast
from ..logging_utils import say, show_table


def morans_i(x: np.ndarray, nb: np.ndarray) -> float:
    """Moran's I with binary kNN weights: ``nb`` is (n, k) row-local neighbour indices."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or x.std() == 0:
        return np.nan
    z = x - x.mean()
    num = float((z[:, None] * z[nb]).sum())
    k = nb.shape[1]
    return num / (k * float((z * z).sum()))


def remap_neighbours(nb_local: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Neighbour table of the kept rows, with indices into the kept subset.

    ``nb_local`` is (n, k) with indices into the full local set; ``keep`` is (n,) bool. Rows not kept are
    dropped; a reference to a dropped cell is replaced by the referring cell's own kept index, which adds
    z_i * z_i to Moran's numerator only for those rare edges and keeps the (n_kept, k) shape.
    """
    keep = np.asarray(keep, dtype=bool)
    new_index = np.full(len(keep), -1, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    out = new_index[nb_local[keep]]
    own = np.arange(int(keep.sum()))[:, None]
    return np.where(out >= 0, out, own)


def plasma_dense_flag(obs: pd.DataFrame, lineage: np.ndarray, lin_names: Sequence[str], cfg,
                      um_by_sample: Dict[str, float], min_plasma: int = 5) -> np.ndarray:
    """True for cells with at least ``min_plasma`` plasma cells (other than itself) within ``nb_radius_um``."""
    if "plasma" not in lin_names:
        return np.zeros(len(obs), dtype=bool)
    is_plasma = np.asarray(lineage) == list(lin_names).index("plasma")
    samples = obs["sample"].astype(str).to_numpy()
    xy = obs[["x", "y"]].to_numpy(dtype=np.float32)
    n_near = np.zeros(len(obs), dtype=int)
    for s in pd.unique(samples):
        idx = np.where(samples == s)[0]
        pl = idx[is_plasma[idx]]
        if len(pl) == 0:
            continue
        r_px = cfg.nb_radius_um / float(um_by_sample.get(s, cfg.um_per_px))
        n_near[idx] = KDTree(xy[pl]).query_radius(xy[idx], r=r_px, count_only=True)
    n_near = n_near - is_plasma.astype(int)
    return n_near >= min_plasma


def spatial_fidelity(preds: np.ndarray, Y: np.ndarray, targets: Sequence[str], test_targets: Sequence[str],
                     groups: np.ndarray, NB: np.ndarray, obs: pd.DataFrame, lineage: np.ndarray, lin_names: Sequence[str],
                     cfg, um_by_sample: Dict[str, float], min_cells: int = 30) -> Tuple[pd.DataFrame, dict]:
    """Return a long table (check in {moran, niche_contrast}) and headline numbers for the first test target."""
    rows: List[dict] = []
    patients = np.unique(groups)
    # row-local neighbour tables per patient (NB holds global indices, same slide)
    for t in test_targets:
        if t not in targets:
            continue
        j = list(targets).index(t)
        for p in patients:
            idx = np.where(groups == p)[0]
            if len(idx) <= min_cells:
                continue
            local = np.full(len(groups), -1, dtype=np.int64)
            local[idx] = np.arange(len(idx))
            nb_local = local[NB[idx]]
            keep = (nb_local >= 0).all(axis=1)
            if keep.sum() <= min_cells:
                continue
            # keep only cells whose neighbours are all in this patient, and remap the neighbour indices into the
            # kept subset; a neighbour that was itself dropped points at the cell's own row (zero contribution
            # in the numerator) so the k-regular denominator stays honest
            nb_local = remap_neighbours(nb_local, keep)
            pred = np.asarray(preds[idx, j], dtype=np.float64)
            meas = np.asarray(Y[idx, j], dtype=np.float64)
            ok = np.isfinite(pred)
            if ok.sum() <= min_cells:
                continue
            # Moran's I needs every neighbour present: fill missing predictions with the patient mean (rare)
            pred_f = np.where(ok, pred, np.nanmean(pred))
            i_pred = morans_i(pred_f[keep], nb_local)
            i_meas = morans_i(meas[keep], nb_local)
            rows.append(dict(check="moran", target=t, patient=str(p), n_cells=int(keep.sum()), moran_pred=i_pred,
                             moran_measured=i_meas,
                             moran_ratio=(i_pred / i_meas) if np.isfinite(i_meas) and i_meas != 0 else np.nan))
    say(f"spatial fidelity: Moran's I on the k={NB.shape[1]} graph for {len(rows)} target x patient pairs")

    dense = plasma_dense_flag(obs, lineage, lin_names, cfg, um_by_sample)
    say(f"plasma-cell-dense niches: {dense.mean():.1%} of cells ({int(dense.sum())}) have >= 5 plasma cells within {cfg.nb_radius_um} um")
    frame = pd.DataFrame({"plasma_dense": dense})
    cols: List[str] = []
    for t in test_targets:
        if t not in targets:
            continue
        j = list(targets).index(t)
        frame[f"{t}__pred"] = preds[:, j]
        frame[f"{t}__measured"] = Y[:, j]
        cols += [f"{t}__pred", f"{t}__measured"]
    niche = pd.DataFrame()
    if dense.sum() >= 5 and (~dense).sum() >= 5 and cols:
        finite = np.isfinite(frame[cols].to_numpy()).all(axis=1)
        frame = frame[finite].reset_index(drop=True)
        niche = niche_contrast(frame, cols, groups=np.asarray(groups)[finite])
        for t in test_targets:
            if f"{t}__pred" not in set(niche.get("program", [])):
                continue
            rp = niche[niche["program"] == f"{t}__pred"].iloc[0]
            rm = niche[niche["program"] == f"{t}__measured"].iloc[0]
            rows.append(dict(check="niche_contrast", target=t, patient="all", n_cells=int(rp.get("n_dense", 0) + rp.get("n_other", 0)),
                             n_dense=int(rp.get("n_dense", 0)), n_other=int(rp.get("n_other", 0)),
                             cohens_d_pred=rp.get("cohens_d", np.nan), cohens_d_measured=rm.get("cohens_d", np.nan),
                             sign_agreement=bool(np.sign(rp.get("cohens_d", np.nan)) == np.sign(rm.get("cohens_d", np.nan)))
                             if np.isfinite(rp.get("cohens_d", np.nan)) and np.isfinite(rm.get("cohens_d", np.nan)) else np.nan))
    else:
        say("niche contrast skipped: fewer than 5 cells on one side of the plasma-dense flag")
    table = pd.DataFrame(rows)
    headline: dict = {}
    if not table.empty:
        mor = table[table["check"] == "moran"]
        t0 = test_targets[0] if len(test_targets) else None
        m0 = mor[mor["target"] == t0]
        headline = dict(main_target=t0, n_patients=int(m0["patient"].nunique()) if len(m0) else 0,
                        median_moran_pred=float(m0["moran_pred"].median()) if len(m0) else np.nan,
                        median_moran_measured=float(m0["moran_measured"].median()) if len(m0) else np.nan,
                        median_moran_ratio=float(m0["moran_ratio"].median()) if len(m0) else np.nan)
        nc = table[(table["check"] == "niche_contrast") & (table["target"] == t0)]
        if len(nc):
            headline.update(cohens_d_pred=float(nc["cohens_d_pred"].iloc[0]), cohens_d_measured=float(nc["cohens_d_measured"].iloc[0]),
                            niche_sign_agreement=nc["sign_agreement"].iloc[0])
        show_table(mor.groupby("target")[["moran_pred", "moran_measured", "moran_ratio"]].median().round(3),
                   title="Moran's I, median over patients (predicted vs measured):")
    return table, headline
