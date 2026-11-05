"""Transcript-free validation of the morphology model against CODEX spatial proteomics.

Design
------
The model is applied to the CODEX DAPI channel exactly as it would be to any
new slide (nucleus detection, patch features, prediction with the unseen-slide
embedding). Detected nuclei are matched to the CODEX cell table within 8 um,
which transfers two protein-defined facts to every nucleus: its lineage and
whether it sits in a plasma-cell-dense niche (CD138+ cells within the same
30 um radius used for the transcript targets). Two questions follow:

1. Composition agreement: do the lineage proportions from the model's
   composition head agree with the protein-defined proportions? Compared in
   centred log-ratio space (Aitchison geometry), because proportions are
   compositional and a Pearson r on raw fractions is dominated by the largest
   lineage.
2. Niche contrast: do predicted program scores differ between plasma-dense
   niches and the rest of the marrow? The Ohlstrom et al. (2026, bioRxiv
   preprint) study reports non-canonical Wnt expression as a property of
   plasma-cell-dense niches, so a positive contrast for the investigator-defined
   Wnt and adhesion set is the expected direction.

Cells of one section are not independent, so the per-cell Mann-Whitney p-value
is reported as descriptive only; a slide-level paired comparison is added when
four or more sections are available.

No public dataset pairs CODEX and Xenium on the same myeloma trephine today, so
the stage is gated on ``cfg.codex_table`` and ``cfg.codex_dapi`` and is
exercised on a synthetic fixture in the tests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neighbors import KDTree

from ..logging_utils import say, section, show_table, timed
from ..models.neighbours import norm_xy


def dapi_native_um_per_px(path: str, configured: Optional[float] = None) -> float:
    """Native pixel size of a DAPI image: the config value, else OME PhysicalSizeX, else TIFF resolution tags.

    Raises ``ValueError`` naming ``codex_dapi_um_per_px`` when none is available, because a
    wrong pixel size silently matches nuclei to the wrong CODEX cells.
    """
    if configured is not None:
        say(f"CODEX DAPI pixel size from config: {configured} um/px")
        return float(configured)
    import re
    import tifffile
    try:
        with tifffile.TiffFile(path) as tif:
            match = re.search(r'PhysicalSizeX="([\d.]+)"', tif.ome_metadata or "")
            if match:
                value = float(match.group(1))
                say(f"CODEX DAPI pixel size from OME metadata: {value} um/px")
                return value
            page = tif.pages[0]
            xres = page.tags.get("XResolution")
            unit = page.tags.get("ResolutionUnit")
            if xres is not None:
                num, den = xres.value if isinstance(xres.value, tuple) else (xres.value, 1)
                if den and num:
                    px_per_unit = num / den
                    unit_code = getattr(unit.value, "value", unit.value) if unit is not None else 2
                    unit_um = {2: 25400.0, 3: 10000.0}.get(int(unit_code))   # inch, centimetre
                    if unit_um and px_per_unit > 1:
                        value = unit_um / px_per_unit
                        say(f"CODEX DAPI pixel size from TIFF resolution tags: {value:.4f} um/px")
                        return value
    except Exception as exc:  # fall through to the error below
        say(f"could not read pixel size metadata from {path}: {exc}")
    raise ValueError(f"pixel size of {path} is unknown: set codex_dapi_um_per_px in the config (microns per pixel)")


def resample_to_working_frame(img: np.ndarray, native_um_per_px: float, working_um_per_px: float) -> np.ndarray:
    """Rescale a native-resolution DAPI image to the pixel size the model was trained at (uint8)."""
    factor = native_um_per_px / working_um_per_px
    if abs(factor - 1.0) < 1e-3:
        say(f"DAPI already at the working pixel size ({working_um_per_px:.4f} um/px); no resampling")
        return img
    from skimage.transform import rescale
    out = rescale(img, factor, anti_aliasing=factor < 1, preserve_range=True, channel_axis=-1 if img.ndim == 3 else None)
    out = np.clip(out, 0, 255).astype(np.uint8)
    say(f"DAPI resampled from {native_um_per_px:.4f} to {working_um_per_px:.4f} um/px (factor {factor:.3f}): "
        f"{img.shape} -> {out.shape}")
    return out


# ----------------------------------------------------------------------------- composition
def _clr(counts: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    v = np.log(counts + pseudo)
    return v - v.mean()


def composition_agreement(pred_lineage_names: np.ndarray, codex_lineage_names: np.ndarray,
                          lin_names: List[str]) -> dict:
    """Compare two labelings of the same cells as compositions.

    Returns the proportions, the Pearson r between centred log-ratio vectors, the
    Aitchison distance (Euclidean distance between CLR vectors), the per-lineage
    absolute proportion difference, and the per-cell agreement where a lineage
    name exists in both labelings.
    """
    pred = pd.Series(pred_lineage_names)
    codex = pd.Series(codex_lineage_names)
    names = list(lin_names) + [n for n in sorted(set(codex.unique()) | set(pred.unique())) if n not in lin_names]
    n_pred = np.array([(pred == n).sum() for n in names], dtype=float)
    n_codex = np.array([(codex == n).sum() for n in names], dtype=float)
    p_pred, p_codex = n_pred / max(n_pred.sum(), 1), n_codex / max(n_codex.sum(), 1)
    clr_pred, clr_codex = _clr(n_pred), _clr(n_codex)
    r = float(stats.pearsonr(clr_pred, clr_codex)[0]) if len(names) > 2 else np.nan
    dist = float(np.linalg.norm(clr_pred - clr_codex))
    shared = np.array([n in set(lin_names) for n in codex.values])
    cell_agreement = float((pred.values[shared] == codex.values[shared]).mean()) if shared.any() else np.nan
    table = pd.DataFrame({"lineage": names, "pred_fraction": p_pred.round(4), "codex_fraction": p_codex.round(4),
                          "abs_diff": np.abs(p_pred - p_codex).round(4)})
    show_table(table, title="composition: model head vs protein-defined lineage")
    say(f"composition agreement: CLR Pearson r = {r:.3f}, Aitchison distance = {dist:.3f}, "
        f"per-cell label agreement = {cell_agreement:.3f}")
    return dict(lineages=names, pred_fraction=p_pred.tolist(), codex_fraction=p_codex.tolist(),
                abs_diff=np.abs(p_pred - p_codex).tolist(), clr_pearson_r=r, aitchison_distance=dist,
                cell_agreement=cell_agreement)


# ----------------------------------------------------------------------------- niche contrast
def niche_contrast(vmap: pd.DataFrame, pred_columns: List[str], groups: Optional[np.ndarray] = None,
                   flag_col: str = "plasma_dense") -> pd.DataFrame:
    """Per predicted program: mean inside vs outside plasma-dense niches, Cohen's d, Mann-Whitney p.

    The p-value is descriptive: cells of one section are spatially correlated and
    not independent. With ``groups`` (slide id per cell) covering at least four
    slides, a slide-level paired comparison (Wilcoxon on per-slide mean
    differences) is reported as well, which is the honest unit of replication.
    """
    flag = vmap[flag_col].values.astype(bool)
    rows = []
    for c in pred_columns:
        a, b = vmap.loc[flag, c].values, vmap.loc[~flag, c].values
        if len(a) < 2 or len(b) < 2:
            rows.append(dict(program=c, n_dense=len(a), n_other=len(b)))
            continue
        # pooled SD weighted by degrees of freedom: with n_dense << n_other an unweighted average overstates d
        na, nb = len(a), len(b)
        pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)) + 1e-9
        d = float((a.mean() - b.mean()) / pooled)
        p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        row = dict(program=c, n_dense=len(a), n_other=len(b), mean_dense=float(a.mean()), mean_other=float(b.mean()),
                   cohens_d=d, mannwhitney_p_descriptive=p)
        if groups is not None:
            g = np.asarray(groups)
            diffs = []
            for s in np.unique(g):
                m = g == s
                if (flag & m).sum() >= 5 and (~flag & m).sum() >= 5:
                    diffs.append(vmap.loc[flag & m, c].mean() - vmap.loc[~flag & m, c].mean())
            row["n_slides_paired"] = len(diffs)
            if len(diffs) >= 4:
                row["slide_mean_diff"] = float(np.mean(diffs))
                row["slide_wilcoxon_p"] = float(stats.wilcoxon(diffs).pvalue)
        rows.append(row)
    out = pd.DataFrame(rows)
    show_table(out.round(4), title="predicted programs: plasma-dense niche vs rest (p-values descriptive at cell level)")
    return out


# ----------------------------------------------------------------------------- lineage from the model head
@torch.no_grad()
def predicted_lineage_names(model: Any, X: np.ndarray, XY: np.ndarray, SID: np.ndarray, device: str,
                            lin_names: List[str], batch: int = 8192) -> np.ndarray:
    """Argmax of the composition head, as lineage names. Raises if the model has no composition head."""
    if getattr(model, "comp", None) is None:
        raise ValueError("model was trained without a composition head (use_comp=False); no lineage prediction")
    model.eval()
    out = []
    for b in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[b: b + batch], dtype=np.float32), device=device)
        xyb = torch.as_tensor(np.asarray(XY[b: b + batch], dtype=np.float32), device=device)
        sb = torch.as_tensor(np.asarray(SID[b: b + batch], dtype=np.int64), device=device)
        _, logits = model(xb, xyb, sb)
        out.append(logits.argmax(1).cpu().numpy())
    idx = np.concatenate(out) if out else np.zeros(0, dtype=int)
    return np.asarray(lin_names, dtype=object)[idx]


# ----------------------------------------------------------------------------- the stage
def run_codex_stage(cfg: Any, run: Any, model: Any, fs: Any, selector: Any, targets: List[str], n_samples: int,
                    device: str, lin_names: List[str], encoder: Optional[Any] = None,
                    match_um: float = 8.0, **vmap_hooks) -> Optional[dict]:
    """Data-gated CODEX validation. Returns the result dict (also written to ``codex_validation.json``) or None.

    ``vmap_hooks`` are passed to ``virtual_program_map`` (``detect_fn``, ``embed_fn``,
    ``morpho_fn``, ``encoder_fn``) so tests can avoid a real encoder.
    """
    section("CODEX bridge: transcript-free validation on spatial proteomics")
    if not getattr(cfg, "codex_table", None) or not getattr(cfg, "codex_dapi", None):
        say("gate: set codex_table (cell-by-marker CSV/parquet with x, y) and codex_dapi (DAPI image of the same "
            "section) to run this stage. No public dataset currently pairs CODEX with Xenium on the same myeloma "
            "trephine; the stage is validated on a synthetic fixture only.")
        return None

    from ..data.codex import assign_lineage_from_markers, codex_to_obs, load_codex_table, load_marker_mapping, match_markers
    from ..data.images import load_image
    from ..inference.virtual_map import virtual_program_map

    mapping = load_marker_mapping()
    table = load_codex_table(cfg.codex_table, um_per_px_codex=getattr(cfg, "codex_um_per_px", None))
    matched = match_markers(list(table.columns), mapping)
    lineage = assign_lineage_from_markers(table, mapping, matched)
    cobs = codex_to_obs(table, lineage, cfg, sample="CODEX", mapping=mapping, matched=matched)

    # The DAPI image is read at pyramid level 0 at its native pixel size and resampled to the working
    # pixel size the model was trained at, so patches, density radii and the nucleus matching all share one frame.
    native = dapi_native_um_per_px(cfg.codex_dapi, getattr(cfg, "codex_dapi_um_per_px", None))
    say(f"reading CODEX DAPI {cfg.codex_dapi} at level 0 (native {native:.4f} um/px; working {cfg.um_per_px:.4f} um/px)")
    img = resample_to_working_frame(load_image(cfg.codex_dapi, 0, None, cfg.z_project, cfg.stain_mode), native, cfg.um_per_px)

    captured: Dict[str, np.ndarray] = {}

    def _capture_embed(obs, store, enc, out_path, cfg_, device_):
        embed_impl = vmap_hooks.get("embed_fn")
        if embed_impl is None:
            from ..features.encoder import embed_cells as embed_impl
        emb = embed_impl(obs, store, enc, out_path, cfg_, device_)
        captured["emb"] = np.asarray(emb, dtype=np.float32).copy()
        return emb

    def _capture_morpho(obs, store, cfg_):
        morpho_impl = vmap_hooks.get("morpho_fn")
        if morpho_impl is None:
            from ..features.morphometrics import morphometrics as morpho_impl
        mor = morpho_impl(obs, store, cfg_)
        captured["mor"] = np.asarray(mor, dtype=np.float32)
        return mor

    hooks = {k: v for k, v in vmap_hooks.items() if k in ("detect_fn", "encoder_fn")}
    with timed("morphology-only inference on the CODEX DAPI channel"):
        vmap = virtual_program_map(img, model, fs, selector, cfg, device, targets, n_samples, encoder=encoder,
                                   embed_fn=_capture_embed, morpho_fn=_capture_morpho, **hooks)
    if len(vmap) == 0:
        say("no nuclei detected on the CODEX DAPI image; stage aborted")
        return None

    # lineage from the model's own composition head at the detected nuclei
    X = fs.transform(captured["emb"], captured["mor"])
    XY = norm_xy(vmap[["x", "y", "sample"]])
    SID = np.full(len(vmap), n_samples, dtype=np.int64)
    vmap["pred_lineage"] = predicted_lineage_names(model, X, XY, SID, device, lin_names)

    # match detected nuclei to CODEX cells within match_um
    tree = KDTree(cobs[["x", "y"]].values.astype(np.float32))
    d, j = tree.query(vmap[["x", "y"]].values.astype(np.float32), k=1)
    ok = d[:, 0] < match_um * cfg.px_per_um
    say(f"matched {int(ok.sum())}/{len(vmap)} detected nuclei to CODEX cells within {match_um} um")
    if ok.sum() < 10:
        say("too few matches; check that the CODEX table and DAPI image are in the same frame and scale")
        return None
    matched_vmap = vmap.loc[ok].copy()
    matched_vmap["codex_lineage"] = cobs["lineage_name"].values[j[ok, 0]]
    matched_vmap["plasma_dense"] = cobs["plasma_dense"].values[j[ok, 0]]

    comp = composition_agreement(matched_vmap["pred_lineage"].values, matched_vmap["codex_lineage"].values, lin_names)
    pred_cols = [t.replace("_z", "_pred") for t in targets]
    contrast = niche_contrast(matched_vmap, pred_cols)

    result = dict(n_codex_cells=int(len(cobs)), n_detected=int(len(vmap)), n_matched=int(ok.sum()),
                  match_um=match_um, composition=comp, niche_contrast=contrast.to_dict("records"),
                  caveat="cell-level p-values are descriptive; one section; no paired Xenium on the same tissue")
    with open(run.art("codex"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    matched_vmap.to_csv(run.art("codex_matched_nuclei.csv"), index=False)
    say(f"CODEX validation written to {run.art('codex')}")
    return result
