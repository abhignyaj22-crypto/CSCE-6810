"""FeatureSelection component: the fold-free report of what is dropped and why.

Policy unchanged from v2 (D14): only dead embedding dimensions and redundant
morphometrics are dropped; the univariate screen, PCA variance and permutation importance
are reports. The fold-time selector is fitted again inside each training fold on training
cells only; this component fits it once on all cells for the record and checks that the
selector is deterministic for a given training index.
"""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..features.selection import FeatureSelector, save_selection_report
from ..logging_utils import say, warn


class FeatureSelection(BaseComponent):
    name = "feature_selection"
    requires: Tuple[str, ...] = ("EMB", "MOR", "Y", "obs")
    provides: Tuple[str, ...] = ("selection",)
    artefacts: Tuple[str, ...] = ("feature_selection.json", "selection_report.csv")
    depends_on: Tuple[str, ...] = ("feature_extraction", "program_inference")
    config_fields: Tuple[str, ...] = ("selection_enabled", "selection_variance_floor", "selection_corr_threshold",
                                      "selection_report_top", "pca_dim")

    def _run(self, ctx: Context) -> None:
        n = len(ctx.obs)
        all_idx = np.arange(n)
        say(f"inputs: EMB {ctx.EMB.shape}, MOR {ctx.MOR.shape}, Y {ctx.Y.shape}; selection_enabled={ctx.cfg.selection_enabled}")
        sel = FeatureSelector(ctx.cfg).fit(ctx.EMB, ctx.MOR, ctx.Y, ctx.obs, all_idx)
        # determinism check: the same training index must give the same kept dimensions
        sel2 = FeatureSelector(ctx.cfg).fit(ctx.EMB, ctx.MOR, ctx.Y, ctx.obs, all_idx)
        same = np.array_equal(sel.keep_emb_dims, sel2.keep_emb_dims) and np.array_equal(sel.keep_mor_dims, sel2.keep_mor_dims)
        if not same:
            warn("feature selector is not deterministic for the same training index; fold results may not be reproducible")
        say(f"selector deterministic for a fixed training index: {same}")
        save_selection_report(sel, self.path(ctx, "feature_selection.json"))
        rows = []
        for key, val in sel.report.items():
            if isinstance(val, (int, float, str)):
                rows.append(dict(item=key, value=val))
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    rows.append(dict(item=f"{key}.{k2}", value=v2 if isinstance(v2, (int, float, str)) else json.dumps(v2, default=float)[:200]))
            elif isinstance(val, list):
                rows.append(dict(item=key, value=json.dumps(val, default=float)[:500]))
        pd.DataFrame(rows).to_csv(self.path(ctx, "selection_report.csv"), index=False)
        selection = dict(keep_emb_dims=None if sel.keep_emb_dims is None else [int(i) for i in sel.keep_emb_dims],
                         keep_mor_dims=None if sel.keep_mor_dims is None else [int(i) for i in sel.keep_mor_dims],
                         report=sel.report, deterministic=bool(same))
        ctx.set("selection", selection)
        say(f"decision: keep {len(selection['keep_emb_dims'] or [])}/{ctx.EMB.shape[1]} embedding dims and "
            f"{len(selection['keep_mor_dims'] or [])}/{ctx.MOR.shape[1]} morphometrics; everything else is report only")
        from ..observations import observe
        observe(ctx.cfg.run_dir, "feature_selection",
                before=f"EMB {ctx.EMB.shape[1]} dims, MOR {ctx.MOR.shape[1]} morphometrics",
                after=f"kept {len(selection['keep_emb_dims'] or [])} embedding dims and "
                      f"{len(selection['keep_mor_dims'] or [])} morphometrics",
                note="only dead embedding dimensions and redundant morphometrics are dropped",
                examples=[f"{k}={v}" for k, v in list(sel.report.items())[:5] if isinstance(v, (int, float, str))])

    def load(self, ctx: Context) -> None:
        with open(self.path(ctx, "feature_selection.json")) as fh:
            payload = json.load(fh)
        ctx.set("selection", dict(keep_emb_dims=payload.get("keep_emb_dims"), keep_mor_dims=payload.get("keep_mor_dims"),
                                  report=payload.get("report", {}), deterministic=True))
        say(f"loaded selection: {len(payload.get('keep_emb_dims') or [])} embedding dims, "
            f"{len(payload.get('keep_mor_dims') or [])} morphometrics kept")


def select_features(ctx: Context) -> Context:
    """Component 5: dead and redundant features dropped, importance reported.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return FeatureSelection().run_in(ctx)
