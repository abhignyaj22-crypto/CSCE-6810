"""Patient-level cross-validation of the unified model, every variant, seeds and the final fit.

Outer loop: GroupKFold over patients (cfg.n_folds). Inner: one training patient is held
out for early stopping and temperature scaling. The fold's feature space and selector
are fitted on the inner-training cells only (``evaluation.cross_validation.build_features``),
so no test statistic enters the features. Out-of-fold predictions are the only
predictions Evaluation sees.

Variants (DESIGN_V3.md D29): every ablation is a switch on the same network.
    full                all tokens and losses
    no_neighbours       neighbour tokens removed (k = 0 in the model)
    no_grid             centre token only (no patch-block tokens)
    no_program_tokens   raw centre features into a v2-shaped MLP head, no tokens or attention
    no_comp             no lineage head
    no_smooth           no neighbourhood consistency term
    no_align            same as full (alignment is off by default); kept for configuration compatibility
    align               full model plus the contrastive alignment term (lambda_align_on)
    shuffled            targets permuted within slide (chance bound)
    oracle              true lineage one-hot appended to the centre features (identity upper bound)
    anchor_only_targets full model trained on the fixed-set (anchor_only) target family only
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from ..core.context import Context, Fold
from ..evaluation.cross_validation import CVData, build_features
from ..logging_utils import say, section, show_table
from .data import ALL_VARIANTS, CellBatcher, make_loader, variant_flags
from .loop import configure_backend, predict, train_one


# ----------------------------------------------------------------------------- helpers
def _cv_data(ctx: Context, Y: np.ndarray, targets: List[str]) -> CVData:
    n_lin = len(ctx.lin_names)
    onehot = np.zeros((len(ctx.obs), n_lin), dtype=np.float32)
    onehot[np.arange(len(ctx.obs)), ctx.lineage] = 1.0
    return CVData(obs=ctx.obs, Y=Y, XY=ctx.XY, SID=ctx.SID, LIN=ctx.lineage, NB=ctx.NB, GROUPS=ctx.GROUPS,
                  ONEHOT=onehot, targets=list(targets), lin_names=list(ctx.lin_names), n_samples=int(ctx.n_samples))


def make_folds(groups: np.ndarray, n_folds: int) -> List[Fold]:
    """Outer GroupKFold over patients; inner validation patient rotates with the fold index."""
    pats = np.unique(groups)
    n_splits = min(n_folds, len(pats))
    folds = []
    dummy = np.zeros(len(groups))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(dummy, dummy, groups)):
        tr_pats = np.unique(groups[tr])
        va_pat = tr_pats[k % len(tr_pats)]
        train_pats = [str(p) for p in tr_pats if p != va_pat]
        if not train_pats:
            raise ValueError(f"fold {k}: no training patient is left after holding out {va_pat} for early stopping; "
                             f"{len(pats)} patients are too few for n_folds={n_folds} (need at least 3 patients)")
        folds.append(Fold(fold_id=k, train_patients=train_pats, val_patient=str(va_pat),
                          test_patients=[str(p) for p in np.unique(groups[te])]))
    say(f"{n_splits} outer folds over {len(pats)} patients; inner validation patient per fold: "
        + ", ".join(f"f{f.fold_id}:{f.val_patient}" for f in folds))
    return folds


def _rows(groups: np.ndarray, patients: List[str]) -> np.ndarray:
    return np.where(np.isin(groups.astype(str), patients))[0]


def _xy_um(ctx: Context) -> np.ndarray:
    """Cell coordinates in microns using each slide's own pixel size (offsets are only taken within a slide)."""
    return ctx.obs[["x", "y"]].to_numpy(dtype=np.float32) * ctx.um_per_px_per_cell()[:, None]


def _model_builder() -> Callable:
    from ..components.unified_model import build_model
    return build_model


def head_layout(ctx: Context, variant: str) -> Tuple[List[int], List[str]]:
    """Which ``ctx.Y`` columns the network's heads emit, in head order (program-major, scale-minor).

    The network predicts the learned program family; ``anchor_only_targets`` trains the same
    network on the fixed-set family instead, by pointing the same head layout at the
    ``anchor_<p>_<scale>_z`` columns.
    """
    spec = ctx.model_spec or {}
    targets = list(ctx.targets)
    if "target_index" in spec and variant != "anchor_only_targets":
        idx = [int(i) for i in spec["target_index"]]
        return idx, [targets[i] for i in idx]
    programs = spec.get("programs")
    scales = spec.get("scales")
    if not programs or not scales:
        # fall back: every target the family allows, in ctx order
        fam = ctx.target_family or {}
        want = "anchor_only" if variant == "anchor_only_targets" else "learned"
        idx = [j for j, t in enumerate(targets) if fam.get(t, "learned") == want]
        say(f"head layout ({variant}): model_spec lists no programs/scales; using the {len(idx)} '{want}' targets in context order", indent=1)
        return idx, [targets[j] for j in idx]
    prefix = "anchor_" if variant == "anchor_only_targets" else "prog_"
    names = [f"{prefix}{p}_{sc}_z" for p in programs for sc in scales]
    idx = [targets.index(n) for n in names if n in targets]
    missing = [n for n in names if n not in targets]
    if missing:
        say(f"head layout ({variant}): {len(missing)} of {len(names)} program x scale targets are absent "
            f"(e.g. {missing[:3]}); heads are built for the {len(idx)} present", indent=1)
    return idx, [targets[i] for i in idx]


def _spec_for(ctx: Context, d_centre: int, n_targets: int, flags: Dict[str, bool],
              targets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Copy the stored model specification and overwrite what the fold decides (feature dim, flags, head layout).

    ``anchor_only_targets`` points the same network at the fixed-set family, which exists only for
    the curated programs (no ``extra_<i>``): the program tokens are then sliced to those programs
    so the token initialisation and the head count agree.
    """
    spec = dict(ctx.model_spec or {})
    spec.update(d_centre=int(d_centre), d_emb_raw=int(ctx.EMB.shape[1]), n_lineages=len(ctx.lin_names), n_lin=len(ctx.lin_names),
                n_samples=int(ctx.n_samples), k=int(ctx.NB.shape[1]), knn_k=int(ctx.NB.shape[1]),
                d_grid=int(ctx.GRID.shape[-1]) if ctx.GRID is not None else 0,
                n_grid_tokens=int(ctx.GRID.shape[1]) if ctx.GRID is not None else 0, n_targets=int(n_targets))
    spec.setdefault("n_programs", int(n_targets))
    spec.setdefault("n_scales", 1)
    programs = list(spec.get("programs") or [])
    scales = list(spec.get("scales") or [])
    if targets and programs and scales:
        # programs that actually have a head in this variant, in head (program-major) order
        present = []
        for p in programs:
            if any(t.endswith(f"_{p}_{sc}_z") or t.endswith(f"{p}_{sc}_z") for t in targets for sc in scales):
                present.append(p)
        n_sc = max(1, len(targets) // max(1, len(present)))
        if present and len(present) * n_sc == n_targets:
            if len(present) != len(programs):
                init = spec.get("program_init_weights")
                if init is not None:
                    rows = [programs.index(p) for p in present]
                    spec["program_init_weights"] = np.asarray(init, dtype=np.float32)[rows].tolist()
                say(f"head layout: {len(present)} of {len(programs)} programs have targets in this variant "
                    f"({present}); program tokens sliced to match", indent=1)
            spec["n_programs"], spec["n_scales"], spec["programs"] = len(present), int(n_sc), present
    if spec["n_programs"] * spec["n_scales"] != n_targets:
        say(f"head layout: {n_targets} targets do not factor as programs x scales "
            f"({spec['n_programs']} x {spec['n_scales']}); using one head per target", indent=1)
        spec["n_programs"], spec["n_scales"] = int(n_targets), 1
        spec["program_init_weights"] = None
    if ctx.expression_embedding is not None:
        spec["d_align"] = int(np.asarray(ctx.expression_embedding).shape[1])
    spec.update({f"flag_{k}": v for k, v in flags.items()})
    return spec


# ----------------------------------------------------------------------------- one variant
def run_variant(variant: str, ctx: Context, cfg: Any, seed: int = 0, builder: Optional[Callable] = None,
                folds: Optional[List[Fold]] = None, verbose: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    """Out-of-fold predictions for one variant. Returns (preds, unc, comp, fold_log, histories, targets_used)."""
    assert variant in ALL_VARIANTS, f"unknown variant {variant}; choose from {ALL_VARIANTS}"
    builder = builder or _model_builder()
    flags = variant_flags(variant)
    device = ctx.device
    Y_all = np.asarray(ctx.Y, dtype=np.float32)
    keep, targets = head_layout(ctx, variant)
    if not keep:
        say(f"{variant}: no targets of the required family in this run; variant skipped")
        nan = np.full_like(Y_all, np.nan)
        return nan, nan.copy(), None, pd.DataFrame(), {}, []
    Y = Y_all[:, keep].copy()
    say(f"{variant}: the network predicts {len(keep)} targets (ctx.Y columns {keep})")
    expr = ctx.expression_embedding
    lineage = np.asarray(ctx.lineage, dtype=np.int64)
    if variant == "shuffled":
        # Chance bound: every per-cell label (targets, expression embedding, lineage) is permuted within slide with ONE
        # permutation, so nothing true about a cell reaches the network, not even through the composition head; the
        # permuted targets are returned and saved so Evaluation scores this variant against them (v2 protocol).
        rng = np.random.default_rng(seed)
        expr = None if expr is None else np.asarray(expr, dtype=np.float32).copy()
        lineage = lineage.copy()
        for s in np.unique(ctx.SID):
            m = np.where(ctx.SID == s)[0]
            perm = rng.permutation(len(m))
            Y[m] = Y[m][perm]
            lineage[m] = lineage[m][perm]
            if expr is not None:
                expr[m] = expr[m][perm]
        say("shuffled: targets, expression embedding and lineage labels permuted within slide with one permutation; "
            "predictions are scored against the permuted targets", indent=1)
    data = _cv_data(ctx, Y, targets)
    if variant == "shuffled":
        data.LIN = lineage
        onehot = np.zeros_like(data.ONEHOT)
        onehot[np.arange(len(lineage)), lineage] = 1.0
        data.ONEHOT = onehot
    run_variant.last_targets_used = None
    folds = folds or make_folds(ctx.GROUPS, cfg.n_folds)
    xy_um = _xy_um(ctx)
    N, T = Y.shape
    preds = np.full((N, Y_all.shape[1]), np.nan, dtype=np.float32)     # full width: columns not predicted stay NaN
    unc = np.full((N, Y_all.shape[1]), np.nan, dtype=np.float32)
    keep_arr = np.asarray(keep)
    comp = np.full((N, len(ctx.lin_names)), np.nan, dtype=np.float32) if flags["use_comp"] else None
    log_rows, histories = [], {}
    val_preds: Dict[int, Dict[str, np.ndarray]] = {}
    use_morpho = bool(cfg.use_morpho)
    say(f"variant '{variant}': flags {flags}, seed {seed}, {T} targets")
    for f in folds:
        t0 = time.time()
        tr = _rows(ctx.GROUPS, f.train_patients)
        va = _rows(ctx.GROUPS, [f.val_patient])
        te = _rows(ctx.GROUPS, f.test_patients)
        say(f"fold {f.fold_id}: train {len(f.train_patients)} patients ({len(tr)} cells) | early-stop on {f.val_patient} "
            f"({len(va)} cells) | test {f.test_patients} ({len(te)} cells)", indent=1)
        X, fs, selector = build_features(ctx.EMB, ctx.MOR, data, tr, cfg, seed, use_morpho)
        if variant == "oracle":
            X = np.hstack([X, data.ONEHOT])
            say(f"oracle: true lineage one-hot ({data.ONEHOT.shape[1]} columns) appended to the centre features of every "
                f"cell, test cells included; this is the identity upper bound by construction", indent=1)
        train_mask = np.zeros(N, dtype=bool)
        train_mask[tr] = True
        grid = ctx.GRID if flags["use_grid"] else None
        common = dict(X_centre=X, GRID=grid, NB=ctx.NB, NB_DIST=ctx.NB_DIST, XY_um=xy_um, SID=ctx.SID, Y=Y,
                      LIN=lineage, expression_embedding=expr, batch_cells=cfg.batch_cells,
                      train_mask=train_mask, pin_memory=(device == "cuda"))
        b_tr = CellBatcher(cell_idx=tr, shuffle=True, seed=seed, name=f"train f{f.fold_id}", **common)
        b_va = CellBatcher(cell_idx=va, name=f"val f{f.fold_id}", **common)
        b_te = CellBatcher(cell_idx=te, name=f"test f{f.fold_id}", **common)
        spec = _spec_for(ctx, X.shape[1], T, flags, targets=targets)
        torch.manual_seed(seed)                      # weight initialisation depends on the seed, not on how many models came before
        model = builder(spec, device)
        loader = (lambda b: make_loader(b, cfg, device)) if device == "cuda" else None
        model, best, epochs, hist = train_one(model, b_tr, b_va, cfg, device, flags, seed=seed, loader_factory=loader,
                                              verbose=verbose)
        m, s, c, _ = predict(model, b_te, cfg, device, flags, loader)
        preds[np.ix_(te, keep_arr)], unc[np.ix_(te, keep_arr)] = m, s
        if comp is not None and c is not None:
            comp[te] = c
        # predictions on the inner validation patient with THIS fold's model: the split-conformal quantile is
        # taken from these (same model, same temperature as the test predictions), never from out-of-fold rows
        mv, sv, _, _ = predict(model, b_va, cfg, device, flags, loader)
        val_preds[f.fold_id] = dict(rows=va, mean=mv, std=sv)
        histories[f"{variant}_s{seed}_f{f.fold_id}"] = hist
        log_rows.append(dict(variant=variant, seed=seed, fold=f.fold_id, val_patient=f.val_patient,
                             test_patients=",".join(f.test_patients), best_val_fit=round(float(best), 4),
                             epochs=int(epochs), temperature=round(float(getattr(model, "temperature", 1.0)), 3),
                             seconds=round(time.time() - t0, 1)))
        say(f"fold {f.fold_id} done: val fit {best:.4f}, {epochs} epochs, T={getattr(model, 'temperature', 1.0):.2f}, "
            f"{time.time() - t0:.0f}s", indent=1)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    run_variant.last_val_preds = val_preds           # picked up by run_all for the calibration artefact
    if variant == "shuffled":
        Y_used = np.full((N, Y_all.shape[1]), np.nan, dtype=np.float32)
        Y_used[:, keep_arr] = Y
        run_variant.last_targets_used = Y_used       # the permuted targets, full width, for Evaluation
    return preds, unc, comp, pd.DataFrame(log_rows), histories, targets


# ----------------------------------------------------------------------------- everything
def run_all(ctx: Context, cfg: Any, out_dir: str, builder: Optional[Callable] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run every configured variant, the multi-seed repeat of ``full`` and the final fit; write artefacts."""
    section("training: cross-validation of every variant")
    hardware = configure_backend(cfg, ctx.device)
    folds = make_folds(ctx.GROUPS, cfg.n_folds)
    with open(os.path.join(out_dir, "folds.json"), "w") as fh:
        json.dump([f.__dict__ for f in folds], fh, indent=2)
    cv_preds, cv_unc, cv_comp, histories, logs, targets_used = {}, {}, {}, {}, [], {}
    for variant in cfg.model_variants:
        t0 = time.time()
        p, u, c, lg, hist, tg = run_variant(variant, ctx, cfg, seed=cfg.seed, builder=builder, folds=folds, verbose=verbose)
        cv_preds[variant], cv_unc[variant] = p, u
        if c is not None:
            cv_comp[variant] = c
        histories.update(hist)
        logs.append(lg)
        targets_used[variant] = tg
        np.save(os.path.join(out_dir, f"cv_preds_{variant}.npy"), p)
        np.save(os.path.join(out_dir, f"cv_unc_{variant}.npy"), u)
        if c is not None:
            np.save(os.path.join(out_dir, f"cv_comp_{variant}.npy"), c)
        if variant == "shuffled" and getattr(run_variant, "last_targets_used", None) is not None:
            np.save(os.path.join(out_dir, "cv_targets_shuffled.npy"), run_variant.last_targets_used)
            say("permuted targets of the shuffled variant saved (cv_targets_shuffled.npy); Evaluation scores it against these")
        vp = getattr(run_variant, "last_val_preds", {})
        if variant == "full" and vp:
            np.savez(os.path.join(out_dir, "cv_val_preds_full.npz"),
                     **{f"rows_f{k}": v["rows"] for k, v in vp.items()},
                     **{f"mean_f{k}": v["mean"] for k, v in vp.items()},
                     **{f"std_f{k}": v["std"] for k, v in vp.items()},
                     target_index=np.asarray(head_layout(ctx, "full")[0]))
            say("in-fold validation predictions of the full model saved (cv_val_preds_full.npz) for the conformal quantile")
        say(f"variant '{variant}' finished in {time.time() - t0:.0f}s")
    # multi-seed stability of the full model
    seeds = [s for s in cfg.seeds if s != cfg.seed]
    multiseed: Dict[int, np.ndarray] = {cfg.seed: cv_preds.get("full")} if "full" in cv_preds else {}
    for s in seeds:
        p, _, _, lg, _, _ = run_variant("full", ctx, cfg, seed=s, builder=builder, folds=folds)
        multiseed[s] = p
        logs.append(lg)
        np.save(os.path.join(out_dir, f"cv_preds_full_seed{s}.npy"), p)
    fold_log = pd.concat([lg for lg in logs if len(lg)], ignore_index=True) if logs else pd.DataFrame()
    fold_log.to_csv(os.path.join(out_dir, "cv_fold_log.csv"), index=False)
    show_table(fold_log, title="fold log:")
    final_path, cps = fit_final(ctx, cfg, out_dir, builder=builder)
    hardware.update(cells_per_second_final_fit=round(cps, 1), n_cells=int(len(ctx.obs)), batch_cells=int(cfg.batch_cells),
                    compile_model=bool(getattr(cfg, "compile_model", False)), cuda_graphs=bool(getattr(cfg, "cuda_graphs", False)))
    with open(os.path.join(out_dir, "hardware.json"), "w") as fh:
        json.dump(hardware, fh, indent=2, default=str)
    say(f"hardware.json: {hardware}")
    return dict(folds=folds, cv_preds=cv_preds, cv_unc=cv_unc, cv_comp=cv_comp, histories=histories,
                final_model_path=final_path, hardware=hardware, multiseed=multiseed, targets_used=targets_used)


def fit_final(ctx: Context, cfg: Any, out_dir: str, builder: Optional[Callable] = None) -> Tuple[str, float]:
    """Train on every patient but the last (early-stop patient) and save the deployable checkpoint."""
    section("training: final fit on all patients")
    builder = builder or _model_builder()
    flags = variant_flags("full")
    pats = [str(p) for p in np.unique(ctx.GROUPS)]
    va_pat = pats[-1]
    say(f"final fit: {len(pats) - 1} training patients; patient '{va_pat}' (last in sorted order) is held out for "
        f"early stopping and temperature scaling and is NOT in the training set of the deployed model")
    tr = _rows(ctx.GROUPS, [p for p in pats if p != va_pat])
    va = _rows(ctx.GROUPS, [va_pat])
    keep, targets = head_layout(ctx, "full")
    Y = np.asarray(ctx.Y, dtype=np.float32)[:, keep]
    data = _cv_data(ctx, Y, targets)
    X, fs, selector = build_features(ctx.EMB, ctx.MOR, data, tr, cfg, cfg.seed, bool(cfg.use_morpho))
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[tr] = True
    xy_um = _xy_um(ctx)
    common = dict(X_centre=X, GRID=ctx.GRID, NB=ctx.NB, NB_DIST=ctx.NB_DIST, XY_um=xy_um, SID=ctx.SID, Y=Y,
                  LIN=ctx.lineage, expression_embedding=ctx.expression_embedding, batch_cells=cfg.batch_cells,
                  train_mask=train_mask, pin_memory=(ctx.device == "cuda"))
    b_tr = CellBatcher(cell_idx=tr, shuffle=True, seed=cfg.seed, name="final train", **common)
    b_va = CellBatcher(cell_idx=va, name="final val", **common)
    spec = _spec_for(ctx, X.shape[1], Y.shape[1], flags, targets=targets)
    torch.manual_seed(cfg.seed)
    model = builder(spec, ctx.device)
    loader = (lambda b: make_loader(b, cfg, ctx.device)) if ctx.device == "cuda" else None
    model, best, epochs, hist = train_one(model, b_tr, b_va, cfg, ctx.device, flags, seed=cfg.seed, loader_factory=loader)
    hist.to_csv(os.path.join(out_dir, "final_fit_history.csv"), index=False)
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    ckpt = dict(state=state, spec=spec, feature_space=fs, selector=selector, targets=targets,
                target_index=keep, all_targets=list(ctx.targets), target_family=ctx.target_family, lin_names=list(ctx.lin_names), temperature=float(getattr(model, "temperature", 1.0)),
                cfg=ctx.cfg.to_dict(), flags=flags, n_samples=int(ctx.n_samples), val_patient=va_pat,
                best_val_fit=float(best), epochs=int(epochs))
    path = os.path.join(out_dir, "final_model.pt")
    torch.save(ckpt, path)
    say(f"final model saved to {path}: val fit {best:.4f}, {epochs} epochs, d_centre {X.shape[1]}, T={ckpt['temperature']:.2f}")
    return path, float(getattr(model, "cells_per_second", 0.0))


def load_final(path: str, device: str, builder: Optional[Callable] = None):
    """Rebuild the model from a checkpoint; returns (model, checkpoint dict)."""
    builder = builder or _model_builder()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = builder(ckpt["spec"], device)
    model.load_state_dict(ckpt["state"])
    model.temperature = ckpt.get("temperature", 1.0)
    model.eval()
    say(f"final model loaded from {path}: {len(ckpt['targets'])} targets, T={model.temperature:.2f}")
    return model, ckpt
