"""Independent cohort: score a cohort the model never saw, morphology only, and compare with its expression.

The manifest (``validation_cohort_manifest``) is a JSON file with either
``{"demo": {"n_patients": 2, "cells_per": 400, "img_size": 256, "seed": 11}}``, which builds a
synthetic cohort with a different seed (exercises the code path), or a list of rows in the
ingest manifest format with local paths (``sample``, ``patient``, ``stage``, ``h5``,
``cells``, ``image``; Xenium or Visium HD with bins treated as cells). Real rows go through
the same per-sample ingest body as training (``ingest_one_sample``), then QC and log
normalisation, so the cohort is processed exactly as the training cohort was.

Scoring: the final checkpoint is run on each slide's image (nucleus detection, features,
prediction with the unseen-slide embedding), detected nuclei are matched to the cohort's
cells within 8 um, and the predictions are compared per patient with two measured
quantities built from the cohort's own expression: the fixed-set anchor scores from the
count summaries, and the learned programs projected onto the cohort's genes as a
weight-normalised mean of log expression (genes missing from the cohort's panel are
masked and counted). A within-slide permutation of the measured values gives the chance
level, and the same decision margin as the main cross-validation (0.05) is applied.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import KDTree

from ..logging_utils import say, show_table, timed, warn
from ..programs.targets_builder import neighbourhood_mean
from ..targets.programs import program_score, standardise


def read_cohort_manifest(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def load_cohort(manifest: Any, cfg, programs: Dict[str, List[str]], lineages: Dict[str, List[str]], seed: int):
    """Return ``(adata with log-normalised X and counts layer, image store, manifest rows)``."""
    from ..data.images import ImageStore
    from ..preprocessing.qc import apply_cell_qc, calculate_qc_metrics, normalize_log1p

    if isinstance(manifest, dict) and "demo" in manifest:
        from ..data.synthetic import make_demo
        d = manifest["demo"] or {}
        adata, images, rows = make_demo(programs, lineages, cfg, n_patients=int(d.get("n_patients", 2)),
                                        cells_per=int(d.get("cells_per", 400)), img_size=int(d.get("img_size", 256)),
                                        seed=int(d.get("seed", seed + 1000)))
        for r in rows:
            r["sample"] = f"EXT_{r['sample']}"
            r.setdefault("patient", r["sample"])
        adata.obs["sample"] = "EXT_" + adata.obs["sample"].astype(str)
        if "patient" in adata.obs:
            adata.obs["patient"] = "EXT_" + adata.obs["patient"].astype(str)
        else:
            adata.obs["patient"] = adata.obs["sample"]
        images = {f"EXT_{k}": v for k, v in images.items()}
        say(f"independent cohort (synthetic, seed {int(d.get('seed', seed + 1000))}): {adata.n_obs} cells, {len(rows)} slides")
        xy_by_sample = {s: adata.obs.loc[adata.obs["sample"] == s, ["x", "y"]].to_numpy(np.float32) for s in adata.obs["sample"].unique()}
        store = ImageStore(cfg, rows, xy_by_sample, demo_images=images)
    else:
        import anndata as ad
        from ..pipeline import ingest_one_sample
        from ..preprocessing.spatial_summaries import attach_summaries, count_summaries, subsample_cells
        rows = list(manifest)
        store = ImageStore(cfg, [], {})
        parts = []
        for row in rows:
            row.setdefault("patient", row["sample"])
            for key in ("h5", "cells", "image"):
                if not row.get(key) or not os.path.exists(str(row[key])):
                    raise FileNotFoundError(f"cohort row {row.get('sample')}: {key} path missing or not found: {row.get(key)}")
            store.rows[row["sample"]] = row
            a_sub, _a_full = ingest_one_sample(row, programs, cfg, store, count_summaries, attach_summaries, subsample_cells)
            if a_sub is not None:
                a_sub.obs["patient"] = str(row["patient"])
                parts.append(a_sub)
        if not parts:
            raise RuntimeError("no cohort slide survived ingest")
        adata = ad.concat(parts)
        adata.obs_names_make_unique()
        say(f"independent cohort: {adata.n_obs} cells from {len(parts)} slides")
    if "total_counts" not in adata.obs:
        calculate_qc_metrics(adata)
    adata = apply_cell_qc(adata, cfg)
    adata = normalize_log1p(adata)
    return adata, store, rows


def _learned_proxy(adata, program_weights: pd.DataFrame, usable_genes: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Weight-normalised mean of log expression per program on the cohort's genes; returns (N x P frame, missing per program)."""
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    var = pd.Index(adata.var_names)
    shared = [g for g in usable_genes if g in var]
    H = program_weights.reindex(columns=shared, fill_value=0.0)
    missing = {}
    for p in H.index:
        w_all = program_weights.loc[p]
        missing[p] = int(((w_all > 0.01 * w_all.max()) & ~w_all.index.isin(shared)).sum())
    cols = np.array([var.get_loc(g) for g in shared])
    W = csr_matrix(H.to_numpy(dtype=np.float32).T)               # (len(shared), P)
    S = np.asarray((X[:, cols] @ W).todense()) if issparse(X) else X[:, cols] @ W.toarray()
    S = S / np.maximum(np.asarray(W.sum(axis=0)).ravel(), 1e-9)[None, :]
    return pd.DataFrame(np.asarray(S, dtype=np.float32), columns=list(H.index)), missing


def score_cohort(adata, store, rows: List[dict], checkpoint: Dict[str, Any], cfg, device: str,
                 program_weights: pd.DataFrame, usable_genes: List[str], programs: Dict[str, List[str]],
                 vmap_fn: Callable, seed: int, match_um: float = 8.0, min_cells: int = 30, margin: float = 0.05
                 ) -> Tuple[pd.DataFrame, dict]:
    targets = list(checkpoint["targets"])
    pred_cols = {t: t.replace("_z", "_pred") for t in targets}
    um_by_sample = {str(r["sample"]): float(r["um_per_px_l0"]) * (2 ** cfg.img_level) if r.get("um_per_px_l0") else cfg.um_per_px
                    for r in rows}
    obs = adata.obs.reset_index(drop=True)
    samples = obs["sample"].astype(str).to_numpy()
    # measured quantities on every cohort cell
    proxy, missing = _learned_proxy(adata, program_weights, usable_genes)
    say("learned programs projected onto the cohort panel; weighted genes missing per program: "
        + ", ".join(f"{p}={n}" for p, n in missing.items()))
    measured: Dict[str, np.ndarray] = {}
    kinds: Dict[str, str] = {}
    scales = list(cfg.target_spaces)
    for scale in scales:
        vals = proxy.to_numpy() if scale == "cell" else neighbourhood_mean(proxy.to_numpy(), obs, cfg, um_by_sample)
        for pi, p in enumerate(proxy.columns):
            name = f"prog_{p}_{scale}_z"
            if name in targets:
                measured[name] = standardise(vals[:, pi], samples, cfg.target_norm)
                kinds[name] = "learned_proxy"
    for p in programs:
        if f"cnt_{p}" in obs and "cnt_total" in obs:
            for scale in scales:
                prefix = "" if scale == "cell" else "nb_"
                name = f"anchor_{p}_{scale}_z"
                if name in targets and f"{prefix}cnt_{p}" in obs and f"{prefix}cnt_total" in obs:
                    raw = program_score(obs[f"{prefix}cnt_{p}"].to_numpy(), obs[f"{prefix}cnt_total"].to_numpy())
                    measured[name] = standardise(raw, samples, cfg.target_norm)
                    kinds[name] = "anchor"
    if not measured:
        raise RuntimeError("no checkpoint target could be measured on the cohort (no shared programs or count summaries)")
    rng = np.random.default_rng(seed)
    rows_out: List[dict] = []
    for s in pd.unique(samples):
        img = store.get(s)
        ox, oy = store.offset(s)
        with timed(f"morphology-only inference on cohort slide {s}"):
            vmap = vmap_fn(img, checkpoint, cfg, device, s, None, um_per_px=um_by_sample.get(s, cfg.um_per_px))
        if len(vmap) == 0:
            warn(f"{s}: no nuclei detected; slide skipped")
            continue
        idx = np.where(samples == s)[0]
        ref = obs.loc[idx, ["x", "y"]].to_numpy(dtype=np.float32) - np.array([ox, oy], dtype=np.float32)
        d, nn = KDTree(ref).query(vmap[["x", "y"]].to_numpy(dtype=np.float32), k=1)
        ok = d[:, 0] < match_um / float(um_by_sample.get(s, cfg.um_per_px))
        say(f"{s}: matched {int(ok.sum())}/{len(vmap)} detected nuclei to cohort cells within {match_um} um", indent=1)
        if ok.sum() <= min_cells:
            continue
        cell_idx = idx[nn[ok, 0]]
        perm = rng.permutation(len(cell_idx))
        patient = str(obs.loc[idx[0], "patient"]) if "patient" in obs else s
        for t, meas_all in measured.items():
            col = pred_cols[t]
            if col not in vmap:
                continue
            pred = vmap.loc[ok, col].to_numpy(dtype=np.float64)
            meas = meas_all[cell_idx]
            fin = np.isfinite(pred) & np.isfinite(meas)
            if fin.sum() <= min_cells or pred[fin].std() == 0 or meas[fin].std() == 0:
                continue
            r = float(stats.pearsonr(pred[fin], meas[fin])[0])
            r_sh = float(stats.pearsonr(pred[fin], meas[perm][fin])[0])
            rows_out.append(dict(target=t, measured_kind=kinds[t], patient=patient, slide=s, n_matched=int(fin.sum()),
                                 pearson=r, pearson_shuffled=r_sh, pred_mean=float(pred[fin].mean()), measured_mean=float(meas[fin].mean())))
    table = pd.DataFrame(rows_out)
    headline: dict = dict(n_slides=int(len(pd.unique(samples))), n_cells=int(len(obs)),
                          missing_weighted_genes=missing)
    if not table.empty:
        per_t = table.groupby("target").agg(n_patients=("patient", "nunique"), median_r=("pearson", "median"),
                                            median_r_shuffled=("pearson_shuffled", "median")).reset_index()
        per_t["above_chance"] = (per_t["median_r"] - per_t["median_r_shuffled"]) > margin
        show_table(per_t.round(3), title="independent cohort: per-patient median Pearson, morphology-only vs measured:")
        main = next((t for t in ("prog_wnt_camdr_nb_z", "anchor_wnt_camdr_nb_z") if t in set(per_t["target"])), per_t["target"].iloc[0])
        r0 = per_t[per_t["target"] == main].iloc[0]
        headline.update(main_target=main, median_r=float(r0["median_r"]), median_r_shuffled=float(r0["median_r_shuffled"]),
                        above_chance=bool(r0["above_chance"]), n_patients=int(r0["n_patients"]),
                        targets_above_chance=int(per_t["above_chance"].sum()), targets_tested=int(len(per_t)))
    return table, headline


def cohort_status_note() -> str:
    return ("no independent cohort manifest; set validation.validation_cohort_manifest to a JSON manifest "
            "(Xenium or Visium HD rows with local paths, or {\"demo\": {...}} for a synthetic cohort). The Visium HD trephine "
            "cohort of Ohlstrom et al. 2026 (preprint) is embargoed as of writing (docs/DATA.md)")


def optional_path(path: Optional[str]) -> Optional[str]:
    return path if path and os.path.exists(path) else None
