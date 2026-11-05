"""Morphology-only inference with the unified model (v3).

Input: one DAPI or H&E image and a training checkpoint. Output: one row per detected
nucleus with x, y, the predicted score and predictive standard deviation of every
target, and the lineage probabilities of the composition head. No transcriptomics is
needed; the slide embedding is the zero slot reserved for unseen slides. The neighbour
block mask is still applied when neighbour tokens are present, so the flash attention
kernel is eligible only in the program cross-attention layer; its selection must be
checked with the SDPA kernel warning, not assumed.

Checkpoint contract (written by ``training/cv.py``): keys ``state``, ``fs``, ``selector``,
``spec``, ``targets``, ``lin_names``, ``temperature``, ``cfg``. For tests a pre-built
``model`` object may be stored under ``model`` instead of ``state``.

Batch contract (shared with ``interpretation/attribution.py``): the model is called with
one dict of tensors: ``centre`` (B, D_enc), ``grid`` (B, g*g, D_grid), ``nb`` (B, k, D_enc),
``nb_offset`` (B, k, 2) microns, ``nb_mask`` (B, k), ``sid`` (B,), ``morpho`` (B, M) and
``features`` (B, F) when the checkpoint carries a feature space. It returns a dict with
``mean`` (B, T), optionally ``log_var`` (B, T) and ``comp_logits`` (B, L).
"""
from __future__ import annotations

import copy

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KDTree

from ..logging_utils import say, timed, warn
from ..models.neighbours import knn_table
from ..preprocessing.stain import harmonise_stain


class _SingleImageStore:
    """ImageStore look-alike for one in-memory image."""

    def __init__(self, img: np.ndarray):
        self.img = img

    def get(self, sample: str) -> np.ndarray:
        return self.img

    def offset(self, sample: str) -> Tuple[int, int]:
        return (0, 0)


# ---------------------------------------------------------------------------------------- checkpoint
def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    """Load ``final_model.pt`` and rebuild the network; prints what the checkpoint carries."""
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    spec = ck.get("spec") or {}
    if ck.get("model") is not None:
        model = ck["model"]                       # test path: a ready model object
    else:
        from ..components.unified_model import build_model
        model = build_model(spec, device)
        model.load_state_dict(ck["state"])
        model.temperature = ck.get("temperature", 1.0)
    model = model.to(device) if hasattr(model, "to") else model
    model.eval()
    if ck.get("fs") is None and ck.get("feature_space") is not None:
        ck["fs"] = ck["feature_space"]            # training/cv.py stores the feature space under this key
    say(f"checkpoint {os.path.basename(path)}: {len(ck.get('targets', []))} targets, "
        f"{len(ck.get('lin_names', []))} lineages, temperature {ck.get('temperature', 1.0)}, "
        f"spec keys {sorted(spec)[:8]}{'...' if len(spec) > 8 else ''}")
    ck["model"] = model
    return ck


# ---------------------------------------------------------------------------------------- features
def _default_encoder_fn():
    from ..features.encoder import get_encoder
    return get_encoder


def _grid_tokens_fallback(obs: pd.DataFrame, store: Any, cfg: Any, grid: int) -> np.ndarray:
    """(N, g*g, 2) block mean and std of intensity, the pixel-grid tokens of ``features/patch_grid.py``."""
    from ..features.patch_grid import patch_grid_features
    G = patch_grid_features(obs, store, cfg, grid=grid)          # (N, 2*g*g): means then stds
    return np.stack([G[:, : grid * grid], G[:, grid * grid:]], axis=-1).astype(np.float32)   # (N, g*g, 2)


def extract_inference_features(obs: pd.DataFrame, store: Any, cfg: Any, device: str, spec: Dict[str, Any],
                               encoder: Any = None, embed_fn: Optional[Callable] = None,
                               grid_fn: Optional[Callable] = None, morpho_fn: Optional[Callable] = None,
                               encoder_fn: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """EMB, GRID and MOR for detected nuclei with the same extractor as training.

    Training (``features/embed.py``) pools the encoder's feature map to a g x g grid of
    D_grid-dimensional tokens, so inference must do the same; the ``*_fn`` hooks replace the
    encoder path in tests (``grid_fn(obs, store, cfg, grid)`` must then return (N, g*g, D_grid)).
    """
    # The checkpoint, not the current configuration, decides the encoder and the grid: a run deployed with
    # a different cfg.grid or cfg.encoder would otherwise produce tokens the network was never trained on.
    grid = int(spec.get("grid", getattr(cfg, "grid", 8)))
    encoder_name = str(spec.get("encoder") or getattr(cfg, "encoder", "resnet50"))
    cfg = copy.copy(cfg)
    if getattr(cfg, "grid", grid) != grid or getattr(cfg, "encoder", encoder_name) != encoder_name:
        say(f"inference uses the checkpoint's encoder '{encoder_name}' and grid {grid} "
            f"(config had '{getattr(cfg, 'encoder', None)}' and grid {getattr(cfg, 'grid', None)})")
    cfg.grid = grid
    cfg.encoder = encoder_name
    if embed_fn is None and grid_fn is None:
        from ..features.embed import extract_features
        from ..features.encoders import get_encoder
        if encoder is not None and spec.get("encoder_card") is not None:
            enc, card = encoder, spec["encoder_card"]
        else:
            enc, card = get_encoder(encoder_name, device, cfg)
        out_dir = tempfile.mkdtemp(prefix="mmprog_infer_")
        try:
            with timed("EMB and GRID extraction on detected nuclei"):
                emb_mm, grid_mm = extract_features(obs, store, enc, card, cfg, device, out_dir)
            emb = np.asarray(emb_mm, dtype=np.float32).copy()
            grid_tokens = np.asarray(grid_mm, dtype=np.float32).copy()
            del emb_mm, grid_mm
        finally:
            for f in os.listdir(out_dir):
                try:
                    os.remove(os.path.join(out_dir, f))
                except OSError:
                    pass
        say(f"features: EMB {emb.shape}, GRID {grid_tokens.shape}")
    else:
        if embed_fn is None:
            from ..features.encoder import embed_cells as embed_fn
        if encoder_fn is None:
            encoder_fn = _default_encoder_fn()
        enc = encoder if encoder is not None else encoder_fn(encoder_name, device)
        tmp = tempfile.NamedTemporaryFile(suffix="_infer_emb.npy", delete=False)
        tmp.close()
        try:
            with timed("embedding detected nuclei (hook)"):
                emb = np.asarray(embed_fn(obs, store, enc, tmp.name, cfg, device), dtype=np.float32).copy()
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
        grid_tokens = (grid_fn(obs, store, cfg, grid) if grid_fn is not None
                       else _grid_tokens_fallback(obs, store, cfg, grid))
        say(f"features (hooks): EMB {emb.shape}, GRID {grid_tokens.shape}")
    if morpho_fn is None:
        from ..features.morphometrics import morphometrics as morpho_fn
    mor = np.asarray(morpho_fn(obs, store, cfg), dtype=np.float32)
    # Shape checks against the checkpoint: a token mismatch here would otherwise surface as an opaque
    # broadcasting error inside the position-embedding add, or worse, run silently with wrong features.
    if grid_tokens.shape[1] != grid * grid:
        raise ValueError(f"GRID has {grid_tokens.shape[1]} tokens per cell but the checkpoint expects grid {grid} "
                         f"({grid * grid} tokens); the extractor and the checkpoint disagree")
    d_raw = spec.get("d_emb_raw")
    if d_raw is not None and emb.shape[1] != int(d_raw):
        raise ValueError(f"EMB has {emb.shape[1]} dims but the checkpoint was trained on {d_raw}-dim embeddings "
                         f"(encoder '{encoder_name}'); refusing to predict on mismatched features")
    d_grid = spec.get("d_grid")
    if d_grid is not None and grid_tokens.shape[2] != int(d_grid):
        raise ValueError(f"GRID token dim {grid_tokens.shape[2]} != checkpoint d_grid {d_grid}")
    say(f"feature shapes match the checkpoint: EMB {emb.shape}, GRID {grid_tokens.shape}, grid {grid}, encoder {encoder_name}")
    return dict(EMB=emb, GRID=grid_tokens, MOR=mor)


def build_inference_batch_arrays(obs: pd.DataFrame, feats: Dict[str, np.ndarray], cfg: Any, spec: Dict[str, Any],
                                 fs: Any = None, selector: Any = None, um_per_px: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Assemble the batch arrays for the unified model from features and the detected-nucleus graph.

    ``um_per_px`` is the slide's own pixel size at the working level. Training used each slide's OME
    pixel size for the neighbour offsets (``Context.um_per_px_per_cell``), so inference must do the
    same; the configured default is only the fallback, and the caller prints when it falls back.
    """
    k = int(spec.get("k", spec.get("knn_k", getattr(cfg, "knn_k", 6))))
    n_samples = int(spec.get("n_samples", 0))
    NB = knn_table(obs, k)
    xy = obs[["x", "y"]].to_numpy(dtype=np.float32)
    scale = float(um_per_px) if um_per_px else float(cfg.um_per_px)
    nb_offset = (xy[NB] - xy[:, None, :]) * scale                             # microns, relative to the cell
    say(f"neighbour offsets in microns at {scale:.4f} um/px" + ("" if um_per_px else " (configured default; no slide value given)"))
    emb = feats["EMB"]
    # The model's centre and neighbour inputs are the feature-space vectors (standardised PCA of the
    # embedding plus standardised morphometrics), exactly as in training; raw embeddings only when no
    # feature space was saved.
    centre = emb
    if fs is not None:
        keep_e = getattr(selector, "keep_emb_dims", None) if selector is not None else None
        try:
            centre = np.asarray(fs.transform(emb, feats["MOR"]), dtype=np.float32)
            say(f"feature space applied: centre features {centre.shape}"
                + (f" (selector kept {len(keep_e)} embedding dims)" if keep_e is not None else ""))
        except Exception as e:
            warn(f"feature space transform failed ({type(e).__name__}); raw embeddings used as centre input")
    arrays: Dict[str, np.ndarray] = dict(
        centre=centre, grid=feats["GRID"], nb=centre[NB], nb_offset=nb_offset.astype(np.float32),
        nb_mask=np.ones(NB.shape, dtype=bool), sid=np.full(len(obs), n_samples, dtype=np.int64),
        morpho=feats["MOR"],
    )
    flags = spec.get("flags") or {}
    if flags:
        arrays["flags"] = flags
    say(f"inference batch arrays: centre {arrays['centre'].shape}, grid {arrays['grid'].shape}, "
        f"nb {arrays['nb'].shape}, sid slot {n_samples} (unseen slide), morpho {arrays['morpho'].shape}")
    return arrays


# ---------------------------------------------------------------------------------------- prediction
def predict_unified(model: Any, arrays: Dict[str, np.ndarray], device: str, temperature: float = 1.0,
                    batch_size: int = 2048) -> Dict[str, np.ndarray]:
    """Fixed-size batches (the last one padded, never shrunk), bf16 autocast on CUDA, inference_mode."""
    import torch
    from ..interpretation.attribution import batch_slice, call_model

    n = len(arrays["centre"])
    means, stds, comps = [], [], []
    use_amp = device.startswith("cuda")
    with torch.inference_mode(), timed(f"unified model inference on {n} nuclei"):
        for b in range(0, n, batch_size):
            idx = np.arange(b, min(b + batch_size, n))
            pad = batch_size - len(idx)
            idx_p = np.concatenate([idx, np.repeat(idx[-1], pad)]) if pad > 0 else idx
            batch = batch_slice(arrays, idx_p, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = call_model(model, batch)
            mean = out["mean"].float().cpu().numpy()[: len(idx)]
            means.append(mean)
            if out.get("log_var") is not None:
                lv = out["log_var"].float().clamp(-10, 10).cpu().numpy()[: len(idx)]   # same clamp as training predict
                stds.append(np.exp(0.5 * lv) * float(temperature))
            if out.get("comp_logits") is not None:
                cl = out["comp_logits"].float()
                comps.append(torch.softmax(cl, dim=-1).cpu().numpy()[: len(idx)])
    result = dict(mean=np.vstack(means))
    if stds:
        result["std"] = np.vstack(stds)
    if comps:
        result["comp"] = np.vstack(comps)
    say(f"predictions: mean {result['mean'].shape}" + (f", std present (temperature {temperature:.3f})" if stds else "")
        + (", lineage probabilities present" if comps else ""))
    return result


def virtual_program_map_v3(img: np.ndarray, checkpoint: Dict[str, Any], cfg: Any, device: str,
                           encoder: Any = None, detect_fn: Optional[Callable] = None,
                           embed_fn: Optional[Callable] = None, grid_fn: Optional[Callable] = None,
                           morpho_fn: Optional[Callable] = None, encoder_fn: Optional[Callable] = None,
                           sample_name: str = "NEW", batch_size: int = 2048, um_per_px: Optional[float] = None) -> pd.DataFrame:
    """Detect nuclei, extract features, build the neighbour graph, run the unified model.

    ``um_per_px`` is the slide's pixel size at the working level (from the OME metadata); it sets
    the neighbour offsets in microns exactly as training did for that slide. When None the
    configured default is used and said so.

    Columns: x, y, sample, ``<target>_pred``, ``<target>_std`` (when the model has an
    uncertainty head) and ``lineage_prob_<name>`` (when it has a composition head).
    """
    if detect_fn is None:
        from ..preprocessing.nuclei import detect_nuclei as detect_fn
    targets: List[str] = list(checkpoint["targets"])
    lin_names: List[str] = list(checkpoint.get("lin_names") or [])
    spec = checkpoint.get("spec") or {}
    img = harmonise_stain(img, getattr(cfg, "stain_mode", "native"))
    with timed("nucleus detection"):
        xy = detect_fn(img)
    say(f"detected {len(xy)} nuclei on a {img.shape[1]}x{img.shape[0]} image ({'RGB' if img.ndim == 3 else 'single channel'})")
    pred_cols = [t.replace("_z", "_pred") for t in targets]
    if len(xy) == 0:
        return pd.DataFrame(columns=["x", "y", "sample"] + pred_cols)
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sample": sample_name})
    store = _SingleImageStore(img)
    feats = extract_inference_features(obs, store, cfg, device, spec, encoder=encoder, embed_fn=embed_fn,
                                       grid_fn=grid_fn, morpho_fn=morpho_fn, encoder_fn=encoder_fn)
    spec = dict(spec)
    spec.setdefault("flags", checkpoint.get("flags"))
    spec.setdefault("n_samples", checkpoint.get("n_samples", spec.get("n_samples", 0)))
    arrays = build_inference_batch_arrays(obs, feats, cfg, spec, fs=checkpoint.get("fs") or checkpoint.get("feature_space"),
                                          selector=checkpoint.get("selector"), um_per_px=um_per_px)
    out = predict_unified(checkpoint["model"], arrays, device, temperature=float(checkpoint.get("temperature", 1.0) or 1.0),
                          batch_size=batch_size)
    vmap = obs.copy()
    for j, c in enumerate(pred_cols):
        vmap[c] = out["mean"][:, j]
    if "std" in out:
        for j, t in enumerate(targets):
            vmap[t.replace("_z", "_std")] = out["std"][:, j]
    if "comp" in out and lin_names:
        for j, name in enumerate(lin_names[: out["comp"].shape[1]]):
            vmap[f"lineage_prob_{name}"] = out["comp"][:, j]
    say(f"virtual program map: {len(vmap)} nuclei x {len(targets)} targets; "
        + ", ".join(f"{c} mean {vmap[c].mean():+.3f}" for c in pred_cols[:4]))
    return vmap


# ---------------------------------------------------------------------------------------- checks
def sanity_check(vmap: pd.DataFrame, obs_slide: pd.DataFrame, offset: Tuple[int, int], main_target: str, cfg: Any,
                 match_um: float = 8.0, um_per_px: Optional[float] = None) -> pd.DataFrame:
    """Matched-nucleus Pearson between predictions and measured scores on a training slide, plus calibration offset.

    Returns a one-row table so the deployment component can write it as an artefact.
    """
    ox, oy = offset
    ref = obs_slide[["x", "y"]].to_numpy(dtype=np.float32) - np.array([ox, oy], dtype=np.float32)
    d, nn = KDTree(ref).query(vmap[["x", "y"]].to_numpy(dtype=np.float32), k=1)
    ok = d[:, 0] < match_um / float(um_per_px or cfg.um_per_px)
    col = main_target.replace("_z", "_pred")
    row = dict(main_target=main_target, n_detected=int(len(vmap)), n_segmented=int(len(obs_slide)),
               n_matched=int(ok.sum()), match_um=match_um, pearson=np.nan, pred_mean=np.nan, measured_mean=np.nan,
               coverage_90=np.nan)
    if ok.sum() < 3 or col not in vmap:
        say(f"sanity check: only {int(ok.sum())} nuclei matched within {match_um} um; no correlation computed")
        return pd.DataFrame([row])
    pred = vmap.loc[ok, col].to_numpy()
    meas = obs_slide[main_target].to_numpy()[nn[ok, 0]]
    row["pearson"] = float(stats.pearsonr(pred, meas)[0])
    row["pred_mean"], row["measured_mean"] = float(pred.mean()), float(meas.mean())
    std_col = main_target.replace("_z", "_std")
    if std_col in vmap:
        s = vmap.loc[ok, std_col].to_numpy()
        row["coverage_90"] = float((np.abs(pred - meas) <= 1.645 * s).mean())
    say(f"sanity check: matched {int(ok.sum())}/{len(vmap)} detected nuclei to annotated cells; "
        f"pearson(pred, measured) = {row['pearson']:.3f}")
    say(f"sanity check: detected {len(vmap)} nuclei vs {len(obs_slide)} segmented cells "
        f"({len(vmap) / max(len(obs_slide), 1):.2f}x); predicted mean {row['pred_mean']:+.3f} vs measured mean "
        f"{row['measured_mean']:+.3f} (a large offset means the density features shifted between training and inference)")
    if std_col in vmap:
        say(f"sanity check: empirical 90% coverage of the predictive interval on matched nuclei = {row['coverage_90']:.3f}")
    return pd.DataFrame([row])
