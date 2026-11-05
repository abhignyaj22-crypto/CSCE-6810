"""Training component: patient-level cross-validation of every variant, seeds, final fit."""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context, Fold
from ..logging_utils import say
from ..training.cv import run_all


class Training(BaseComponent):
    name = "training"
    requires: Tuple[str, ...] = ("EMB", "GRID", "MOR", "NB", "NB_DIST", "XY", "SID", "GROUPS", "Y", "LIN", "targets",
                                 "model_spec", "expression_embedding")
    provides: Tuple[str, ...] = ("folds", "cv_preds", "cv_unc", "cv_comp", "histories", "final_model_path", "hardware")
    # only files every run writes are listed: cv_comp_full.npy (needs a composition head), cv_val_preds_full.npz
    # (needs in-fold validation rows) and cv_targets_shuffled.npy (needs the shuffled variant) are conditional and
    # are read with a printed fallback by Training.load and Evaluation
    artefacts: Tuple[str, ...] = ("folds.json", "cv_preds_full.npy", "cv_unc_full.npy", "final_model.pt",
                                  "hardware.json", "cv_fold_log.csv")
    depends_on: Tuple[str, ...] = ("unified_model", "feature_selection")
    config_fields: Tuple[str, ...] = ("n_folds", "seeds", "model_variants", "batch_cells", "max_epochs", "patience", "lr",
                                      "weight_decay", "warmup_steps", "amp_enabled", "amp_dtype", "tf32", "compile_model",
                                      "compile_mode", "cuda_graphs", "grad_clip", "lambda_comp", "lambda_smooth",
                                      "lambda_align", "lambda_align_on", "uncertainty", "nll_beta", "temperature_scaling", "hidden", "dropout", "knn_k",
                                      "pca_dim", "use_morpho", "selection_enabled")

    def _run(self, ctx: Context) -> None:
        cfg = ctx.cfg
        if "full" not in cfg.model_variants:
            raise ValueError("model_variants must include 'full'; the evaluation and deployment read its predictions")
        say(f"variants: {list(cfg.model_variants)}; seeds: {list(cfg.seeds)}; folds: {cfg.n_folds}; "
            f"batch {cfg.batch_cells}; epochs <= {cfg.max_epochs}; device {ctx.device}")
        out = run_all(ctx, cfg, self.dir(ctx))
        ctx.set("folds", out["folds"])
        ctx.set("cv_preds", out["cv_preds"])
        ctx.set("cv_unc", out["cv_unc"])
        ctx.set("cv_comp", out["cv_comp"])
        ctx.set("histories", out["histories"])
        ctx.set("final_model_path", out["final_model_path"])
        ctx.set("hardware", out["hardware"])
        # multi-seed predictions and the target list of every variant are needed by Evaluation
        ms_dir = self.path(ctx, "multiseed")
        os.makedirs(ms_dir, exist_ok=True)
        for s, p in out["multiseed"].items():
            if p is not None:
                np.save(os.path.join(ms_dir, f"seed{s}.npy"), p)
        with open(self.path(ctx, "targets_used.json"), "w") as fh:
            json.dump(out["targets_used"], fh, indent=2)
        for name, h in out["histories"].items():
            h.to_csv(self.path(ctx, f"history_{name}.csv"), index=False)
        from ..observations import observe
        examples = []
        for name, h in list(out["histories"].items())[:5]:
            if len(h):
                last = h.iloc[-1]
                examples.append(f"{name} last epoch { {c: round(float(last[c]), 4) for c in list(h.columns)[:4]} }")
        observe(cfg.run_dir, "training",
                before=f"{len(ctx.obs)} cells, variants {list(cfg.model_variants)}",
                after=f"{len(out['folds'])} folds, final weights {out['final_model_path']}",
                note="patient-level folds; a cell and its patient stay together",
                examples=examples)

    def load(self, ctx: Context) -> None:
        d = self.dir(ctx)
        with open(os.path.join(d, "folds.json")) as fh:
            ctx.set("folds", [Fold(**f) for f in json.load(fh)])
        preds, unc, comp = {}, {}, {}
        for f in sorted(os.listdir(d)):
            if f.startswith("cv_preds_") and f.endswith(".npy") and "seed" not in f:
                v = f[len("cv_preds_"):-4]
                preds[v] = np.load(os.path.join(d, f))
                u = os.path.join(d, f"cv_unc_{v}.npy")
                c = os.path.join(d, f"cv_comp_{v}.npy")
                if os.path.exists(u):
                    unc[v] = np.load(u)
                if os.path.exists(c):
                    comp[v] = np.load(c)
        ctx.set("cv_preds", preds)
        ctx.set("cv_unc", unc)
        ctx.set("cv_comp", comp)
        hist = {}
        for f in sorted(os.listdir(d)):
            if f.startswith("history_") and f.endswith(".csv"):
                hist[f[len("history_"):-4]] = pd.read_csv(os.path.join(d, f))
        ctx.set("histories", hist)
        ctx.set("final_model_path", os.path.join(d, "final_model.pt"))
        with open(os.path.join(d, "hardware.json")) as fh:
            ctx.set("hardware", json.load(fh))
        say(f"training loaded: variants {list(preds)}, {len(hist)} histories")


def train_unified_model(ctx: Context) -> Context:
    """Component 7: patient-level cross-validation of every variant, temperature scaling, the final fit.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Training().run_in(ctx)
