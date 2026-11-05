"""Nested patient-level cross-validation with baselines and ablations.

Outer loop: GroupKFold over patients, so every cell of a patient is either
entirely in training or entirely held out. Inner loop: one training patient is
held out for early stopping. The feature space (scaler, PCA) and the feature
selector are fit on the inner-training cells only, so no held-out information
reaches the model through preprocessing.

Variants
- full:              the proposed model
- no_spatial:        drop the spatial prior g
- no_comp:           drop the lineage auxiliary head
- no_morpho:         drop the morphometric branch
- shuffled:          targets permuted within each slide (chance level, same marginals)
- oracle:            true lineage one-hot appended to the features (what identity alone would buy)
- composition_only:  ridge regression from one-hot lineage, no image at all
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from ..logging_utils import say, section, show_table
from ..models.neighbours import knn_table, norm_xy
from ..models.train import predict, train_model
from .metrics import bootstrap_ci, score

VARIANT_ORDER = ["shuffled", "composition_only", "no_morpho", "no_spatial", "no_comp", "full", "oracle"]


@dataclass
class CVData:
    """Everything the CV loop needs, in the row order of ``obs``."""
    obs: pd.DataFrame
    Y: np.ndarray
    XY: np.ndarray
    SID: np.ndarray
    LIN: np.ndarray
    NB: np.ndarray
    GROUPS: np.ndarray
    ONEHOT: np.ndarray
    targets: List[str]
    lin_names: List[str]
    n_samples: int

    @property
    def n_patients(self) -> int:
        return int(len(np.unique(self.GROUPS)))


def build_cv_data(obs: pd.DataFrame, targets: List[str], lin_names: List[str], cfg: Any) -> CVData:
    """Assemble the arrays once; every variant and seed reuses them."""
    Y = obs[targets].values.astype(np.float32)
    XY = norm_xy(obs)
    SID = pd.Categorical(obs["sample"]).codes.astype(np.int64)
    LIN = obs["lineage"].values.astype(np.int64)
    NB = knn_table(obs, cfg.knn_k)
    GROUPS = obs["patient"].values
    onehot = np.zeros((len(obs), len(lin_names)), dtype=np.float32)
    onehot[np.arange(len(obs)), LIN] = 1.0
    data = CVData(obs=obs, Y=Y, XY=XY, SID=SID, LIN=LIN, NB=NB, GROUPS=GROUPS, ONEHOT=onehot,
                  targets=list(targets), lin_names=list(lin_names), n_samples=int(SID.max()) + 1)
    say(f"cv data: {len(obs)} cells, {len(targets)} targets, {data.n_patients} patients, {data.n_samples} slides")
    return data


def _default_feature_space(pca_dim: int, use_morpho: bool):
    from ..features.feature_space import FeatureSpace
    return FeatureSpace(pca_dim, use_morpho)


def _default_selector(cfg: Any):
    from ..features.selection import FeatureSelector
    return FeatureSelector(cfg)


def build_features(
    EMB: np.ndarray, MOR: np.ndarray, data: CVData, fit_idx: np.ndarray, cfg: Any, seed: int,
    use_morpho: bool, feature_space_factory: Optional[Callable] = None,
    selector_factory: Optional[Callable] = None, selection_enabled: Optional[bool] = None,
) -> Tuple[np.ndarray, Any, Any]:
    """Fit selector + feature space on ``fit_idx`` cells and transform every cell.

    Returns (X for all cells, fitted feature space, fitted selector or None).
    """
    fs_factory = feature_space_factory or _default_feature_space
    sel_factory = selector_factory or _default_selector
    enabled = cfg.selection_enabled if selection_enabled is None else selection_enabled

    selector = None
    keep_emb, keep_mor = None, None
    if enabled:
        selector = sel_factory(cfg).fit(EMB, MOR, data.Y, data.obs, fit_idx)
        keep_emb = getattr(selector, "keep_emb_dims", None)
        keep_mor = getattr(selector, "keep_mor_dims", None) if use_morpho else None
    fs = fs_factory(cfg.pca_dim, use_morpho).fit(EMB, MOR, idx=fit_idx, seed=seed,
                                                   keep_emb_dims=keep_emb, keep_mor_dims=keep_mor)
    X = fs.transform(EMB, MOR)
    return X, fs, selector


def run_cv(
    variant: str, data: CVData, EMB: np.ndarray, MOR: np.ndarray, cfg: Any, device: str,
    seed: int = 0, selector_enabled: Optional[bool] = None, verbose: bool = False,
    feature_space_factory: Optional[Callable] = None, selector_factory: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Held-out predictions for every cell under one variant.

    Returns (preds, Y used for scoring, fold log, per-fold training histories keyed
    ``"<variant>_s<seed>_f<fold>"``). The histories feed the training-curve figure.
    """
    use_spatial = variant != "no_spatial"
    use_comp = variant != "no_comp"
    use_morpho = bool(cfg.use_morpho) and variant != "no_morpho"

    Y = data.Y.copy()
    if variant == "shuffled":
        # permute within slide: destroys the cell-to-target link but keeps every slide's marginal
        rng = np.random.default_rng(seed)
        for s in np.unique(data.SID):
            m = data.SID == s
            Y[m] = Y[m][rng.permutation(int(m.sum()))]

    preds = np.zeros_like(Y)
    n_splits = min(cfg.n_folds, data.n_patients)
    gkf = GroupKFold(n_splits=n_splits)
    log = []
    histories: Dict[str, pd.DataFrame] = {}
    say(f"variant '{variant}': spatial={use_spatial} comp={use_comp} morpho={use_morpho} | "
        f"{n_splits} folds over {data.n_patients} patients, seed {seed}")
    for fold, (tr, te) in enumerate(gkf.split(Y, Y, data.GROUPS)):
        t0 = time.time()
        tr_pats = np.unique(data.GROUPS[tr])
        va_pat = tr_pats[fold % len(tr_pats)]
        va = tr[data.GROUPS[tr] == va_pat]
        tr2 = tr[data.GROUPS[tr] != va_pat]
        te_pats = [str(p) for p in np.unique(data.GROUPS[te])]
        say(f"fold {fold}: train {len(tr_pats) - 1} patients ({len(tr2)} cells) | "
            f"early-stop on {va_pat} ({len(va)} cells) | test {te_pats} ({len(te)} cells)", indent=1)

        X, _, _ = build_features(EMB, MOR, data, tr2, cfg, seed, use_morpho,
                                 feature_space_factory, selector_factory, selector_enabled)
        if variant == "oracle":
            X = np.hstack([X, data.ONEHOT])

        model, val_mse, epochs, history = train_model(
            X, Y, data.XY, data.SID, data.LIN, data.NB, tr2, va, data.n_samples, len(data.lin_names),
            cfg, device, use_spatial=use_spatial, use_comp=use_comp, seed=seed, verbose=verbose,
        )
        histories[f"{variant}_s{seed}_f{fold}"] = history
        preds[te] = predict(model, X[te], data.XY[te], data.SID[te], device)
        seconds = time.time() - t0
        log.append(dict(fold=fold, val_patient=str(va_pat), test_patients=",".join(te_pats),
                        val_mse=round(float(val_mse), 4), epochs=int(epochs), seconds=round(seconds, 1)))
        say(f"fold {fold} done: val mse {val_mse:.4f}, {epochs} epochs, {seconds:.0f}s", indent=1)
    return preds, Y, pd.DataFrame(log), histories


def _sample_ids(data: CVData) -> np.ndarray:
    """Sample (slide) id of every cell as strings; the reliability table is keyed by sample."""
    return data.obs["sample"].astype(str).values


def composition_only(data: CVData, cfg: Any) -> np.ndarray:
    """Ridge regression from one-hot lineage alone: the identity baseline."""
    preds = np.zeros_like(data.Y)
    n_splits = min(cfg.n_folds, data.n_patients)
    for tr, te in GroupKFold(n_splits=n_splits).split(data.ONEHOT, data.Y, data.GROUPS):
        preds[te] = Ridge(alpha=1.0).fit(data.ONEHOT[tr], data.Y[tr]).predict(data.ONEHOT[te])
    say(f"composition_only baseline: ridge on {data.ONEHOT.shape[1]} lineage indicators, {n_splits} folds")
    return preds


def run_all_variants(
    data: CVData, EMB: np.ndarray, MOR: np.ndarray, cfg: Any, device: str, run: Any,
    rel_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    feature_space_factory: Optional[Callable] = None, selector_factory: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Run every configured variant plus the composition baseline and write the artefacts.

    Returns (RES, RES_ADJ, per_patient_df, fold_logs, pred_full, histories, full_scores).
    ``histories`` holds the full model's per-fold training curves; ``full_scores`` is its
    score table at ``cfg.seed`` so ``run_multiseed`` does not train that seed again.
    """
    section("S5 patient-level cross-validation")
    results: Dict[str, pd.DataFrame] = {}
    logs: Dict[str, pd.DataFrame] = {}
    histories: Dict[str, pd.DataFrame] = {}
    sids = _sample_ids(data)
    pred_full = None
    for variant in cfg.cv_variants:
        t0 = time.time()
        p, Y_used, lg, hist = run_cv(variant, data, EMB, MOR, cfg, device, seed=cfg.seed,
                                     feature_space_factory=feature_space_factory, selector_factory=selector_factory)
        results[variant] = score(p, Y_used, data.targets, data.GROUPS, rel_lookup,
                                 cfg.min_cells_per_patient_metric, sids=sids).assign(model=variant)
        logs[variant] = lg
        say(f"{variant}: {time.time() - t0:.0f}s | epochs per fold {lg['epochs'].tolist()}")
        if variant == "full":
            pred_full = p
            histories.update(hist)
            np.save(run.art("pred_full"), p)
    results["composition_only"] = score(composition_only(data, cfg), data.Y, data.targets, data.GROUPS,
                                        rel_lookup, cfg.min_cells_per_patient_metric, sids=sids).assign(model="composition_only")
    if pred_full is None:
        # 'full' not among the variants: keep the composition baseline so later stages have something
        pred_full = composition_only(data, cfg)
        np.save(run.art("pred_full"), pred_full)
        say("note: 'full' variant not run; pred_full.npy holds the composition baseline")

    allres = pd.concat(results.values(), ignore_index=True)
    RES = allres.pivot(index="target", columns="model", values="pearson_patient_median").round(3)
    RES = RES[[c for c in VARIANT_ORDER if c in RES.columns]]
    RES_ADJ = allres.pivot(index="target", columns="model", values="pearson_patient_median_adj").round(3)
    RES_ADJ = RES_ADJ[[c for c in VARIANT_ORDER if c in RES_ADJ.columns]]

    per_rows = []
    for r in allres.itertuples():
        for p, v in r.per_patient.items():
            per_rows.append(dict(target=r.target, model=r.model, patient=p, r=v, r_adj=r.per_patient_adj.get(p)))
    per_patient_df = pd.DataFrame(per_rows)

    ci_source = results.get("full")
    if ci_source is not None:
        by_target = ci_source.set_index("target")
        RES["full_95ci"] = [str(bootstrap_ci(by_target.loc[t, "per_patient"], cfg.n_boot, cfg.seed)) for t in RES.index]

    RES.to_csv(run.art("cv_results"))
    RES_ADJ.to_csv(run.art("cv_results_adj"))
    per_patient_df.to_csv(run.art("cv_per_patient"), index=False)
    fold_logs = pd.concat([v.assign(model=k) for k, v in logs.items()], ignore_index=True) if logs else pd.DataFrame()
    fold_logs.to_csv(run.art("cv_fold_log"), index=False)

    show_table(RES, title="per-patient median Pearson (held-out), by target and model:")
    show_table(RES_ADJ, title="reliability-adjusted (r / sqrt(split-half ceiling)):")
    return RES, RES_ADJ, per_patient_df, fold_logs, pred_full, histories, results.get("full")


def run_multiseed(
    data: CVData, EMB: np.ndarray, MOR: np.ndarray, cfg: Any, device: str, run: Any,
    rel_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    feature_space_factory: Optional[Callable] = None, selector_factory: Optional[Callable] = None,
    precomputed: Optional[Dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Repeat the full model over cfg.seeds and report mean and sd of the headline metric.

    ``precomputed`` maps a seed to a score table that ``run_all_variants`` already produced
    (the full model at ``cfg.seed``), so that seed is reused instead of trained a second time.
    """
    section("S5 multi-seed stability of the full model")
    precomputed = precomputed or {}
    sids = _sample_ids(data)
    rows = []
    for s in cfg.seeds:
        if precomputed.get(s) is not None:
            say(f"seed {s}: reusing the full-model scores from the main cross-validation run")
            rows.append(precomputed[s].drop(columns=["model"], errors="ignore").assign(seed=s))
            continue
        p, Y_used, _, _ = run_cv("full", data, EMB, MOR, cfg, device, seed=s,
                                 feature_space_factory=feature_space_factory, selector_factory=selector_factory)
        rows.append(score(p, Y_used, data.targets, data.GROUPS, rel_lookup,
                          cfg.min_cells_per_patient_metric, sids=sids).assign(seed=s))
    tab = pd.concat(rows).groupby("target")["pearson_patient_median"].agg(["mean", "std"]).round(3)
    tab.to_csv(run.art("cv_multiseed"))
    show_table(tab, title=f"full model over seeds {list(cfg.seeds)}:")
    return tab
