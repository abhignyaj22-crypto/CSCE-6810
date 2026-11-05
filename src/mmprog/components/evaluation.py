"""Evaluation component: scores every variant, calibration, decision rules, burden, stage association.

The stage association is PERMANOVA over the patient-level program vectors plus a
Jonckheere-Terpstra trend test per program (``evaluation/stage_association.py``); the
classical MANOVA stays available through ``stage_test: manova`` or ``both``.

Nothing here trains anything. Every number comes from the out-of-fold predictions that
Training wrote, so a result can always be traced to ``training/cv_preds_<variant>.npy``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from ..core.component import BaseComponent
from ..core.context import Context
from ..evaluation.calibration import calibration_from_folds
from ..evaluation.clinical import cox_pfs, program_burden, stage_trend
from ..evaluation.decisions import apply_decision_rules
from ..evaluation.metrics import bootstrap_ci, score
from ..logging_utils import say, section, show_table, warn
from ..resources import load_stage_vocabulary

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z", "anchor_wnt_camdr_nb_z"]
COLUMN_ORDER = ["shuffled", "composition_only", "anchor_only_targets", "no_align", "align", "no_smooth", "no_comp", "no_grid",
                "no_neighbours", "no_program_tokens", "full", "oracle"]


class _ArtefactDir:
    """Minimal RunDir look-alike so v2 helpers that call ``run.art(name)`` write into this component's folder."""

    NAMES = {"manova": "manova_results.csv", "burden": "program_burden.csv", "cox_pfs": "cox_pfs.csv"}

    def __init__(self, directory: str):
        self.path = directory

    def art(self, name: str) -> str:
        return os.path.join(self.path, self.NAMES.get(name, name))

    def done(self, name: str) -> bool:
        return os.path.exists(self.art(name))


def main_target_of(targets: List[str]) -> str:
    for t in MAIN_TARGET_PREFERENCE:
        if t in targets:
            return t
    return targets[0]


def composition_only_predictions(LIN: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_folds: int) -> np.ndarray:
    """Ridge from the lineage one-hot alone, out-of-fold: the cell-type-detector baseline."""
    preds = np.zeros_like(Y)
    n_splits = min(n_folds, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(LIN, Y, groups):
        preds[te] = Ridge(alpha=1.0).fit(LIN[tr], Y[tr]).predict(LIN[te])
    say(f"composition_only baseline: ridge on {LIN.shape[1]} lineage indicators, {n_splits} patient folds")
    return preds


def _valid_columns(preds: np.ndarray) -> np.ndarray:
    return np.array([np.isfinite(preds[:, j]).any() for j in range(preds.shape[1])])


def _load_val_preds(ctx: Context, n_targets: int):
    """In-fold validation predictions of the full model (training/cv_val_preds_full.npz), widened to ctx.Y columns."""
    path = os.path.join(ctx.component_dir("training"), "cv_val_preds_full.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    tidx = np.asarray(z["target_index"])
    out = {}
    for key in z.files:
        if key.startswith("rows_f"):
            k = int(key[len("rows_f"):])
            mean = np.full((len(z[key]), n_targets), np.nan, dtype=np.float32)
            std = np.full_like(mean, np.nan)
            mean[:, tidx] = z[f"mean_f{k}"]
            std[:, tidx] = z[f"std_f{k}"]
            out[k] = dict(rows=z[key], mean=mean, std=std)
    say(f"loaded in-fold validation predictions for {len(out)} folds from {path}")
    return out


class Evaluation(BaseComponent):
    name = "evaluation"
    # cv_unc, folds and LIN are read with guards (uncertainty may be off; LIN falls back to lineage + lin_names)
    requires: Tuple[str, ...] = ("cv_preds", "Y", "targets", "GROUPS", "obs", "rel_lookup", "reliability_summary",
                                 "lineage", "lin_names")
    provides: Tuple[str, ...] = ("results", "results_adj", "calibration", "decisions", "verdict", "stage_assoc", "burden",
                                 "burden_stats", "multiseed")
    artefacts: Tuple[str, ...] = ("cv_results.csv", "cv_results_reliability_adjusted.csv", "calibration.csv",
                                  "decisions.txt", "program_burden.csv", "stage_association.csv", "per_patient.csv")
    depends_on: Tuple[str, ...] = ("training",)
    config_fields: Tuple[str, ...] = ("n_boot", "min_cells_per_patient_metric", "coverage_levels", "conformal_alpha",
                                      "stage_test", "stage_n_perm", "manova_n_perm", "clinical_csv",
                                      "min_plasma_for_burden", "n_folds")

    # ------------------------------------------------------------------
    def _run(self, ctx: Context) -> None:
        cfg = ctx.cfg
        d = self.dir(ctx)
        targets = list(ctx.targets)
        Y = np.asarray(ctx.Y, dtype=np.float32)
        groups = ctx.GROUPS
        sids = ctx.obs["sample"].astype(str).to_numpy()
        main_target = main_target_of(targets)
        say(f"targets: {len(targets)}; main target {main_target}; variants with predictions: {list(ctx.cv_preds)}")

        # ---- score every variant ---------------------------------------------------------
        section("evaluation: per-variant scores")
        tables: Dict[str, pd.DataFrame] = {}
        for variant, preds in ctx.cv_preds.items():
            preds = np.asarray(preds, dtype=np.float32)
            valid = _valid_columns(preds)
            if not valid.any():
                warn(f"variant {variant}: no finite predictions; skipped")
                continue
            t_use = [t for t, v in zip(targets, valid) if v]
            j_use = np.where(valid)[0]
            dropped = [t for t, v in zip(targets, valid) if not v]
            if dropped:
                say(f"{variant}: {len(dropped)} target columns have no predictions from this variant and are not scored "
                    f"(e.g. {dropped[:3]}); the variant's head family does not cover them", indent=1)
            Y_score = Y
            if variant == "shuffled":
                shuf_path = os.path.join(ctx.component_dir("training"), "cv_targets_shuffled.npy")
                if os.path.exists(shuf_path):
                    Y_score = np.load(shuf_path)
                    say("shuffled: scored against the permuted targets it was trained on (chance level of the metric)", indent=1)
                else:
                    warn("shuffled: permuted targets not found; scoring against the true targets (not the chance bound)")
            tab = score(preds[:, j_use], Y_score[:, j_use], t_use, groups, ctx.rel_lookup, cfg.min_cells_per_patient_metric, sids)
            tables[variant] = tab.assign(model=variant)
            say(f"{variant}: median over targets of the per-patient median r = {tab['pearson_patient_median'].median():.3f}", indent=1)
        LIN = ctx.LIN if ctx.LIN is not None else np.eye(len(ctx.lin_names))[ctx.lineage]
        comp_pred = composition_only_predictions(np.asarray(LIN, dtype=np.float32), Y, groups, cfg.n_folds)
        tables["composition_only"] = score(comp_pred, Y, targets, groups, ctx.rel_lookup,
                                           cfg.min_cells_per_patient_metric, sids).assign(model="composition_only")
        allres = pd.concat(tables.values(), ignore_index=True)
        res = allres.pivot(index="target", columns="model", values="pearson_patient_median").round(3)
        res = res[[c for c in COLUMN_ORDER if c in res.columns] + [c for c in res.columns if c not in COLUMN_ORDER]]
        res_adj = allres.pivot(index="target", columns="model", values="pearson_patient_median_adj").round(3)
        res_adj = res_adj[[c for c in res.columns if c in res_adj.columns]]
        if "full" in tables:
            by_t = tables["full"].set_index("target")
            res["full_95ci"] = [str(bootstrap_ci(by_t.loc[t, "per_patient"], cfg.n_boot, cfg.seed)) if t in by_t.index else ""
                                for t in res.index]
        per_rows = [dict(target=r.target, model=r.model, patient=p, r=v, r_adj=r.per_patient_adj.get(p))
                    for r in allres.itertuples() for p, v in r.per_patient.items()]
        pd.DataFrame(per_rows).to_csv(os.path.join(d, "per_patient.csv"), index=False)
        res.columns.name = None                 # the pivot's axis label is not part of the table
        res_adj.columns.name = None
        res.to_csv(os.path.join(d, "cv_results.csv"))
        from ..observations import observe
        if main_target in res.index:
            row = res.loc[main_target]
            examples = [f"{main_target} {col}={row[col]}" for col in list(res.columns)[:5]]
        else:
            examples = [f"{idx}" for idx in list(res.index)[:5]]
        observe(cfg.run_dir, "evaluation",
                before="out-of-fold predictions and the measured program scores",
                after=f"per-patient Pearson table {res.shape[0]} targets x {res.shape[1]} models",
                note="read the per-patient median, and the reliability-adjusted table next to it",
                examples=examples)
        res_adj.to_csv(os.path.join(d, "cv_results_reliability_adjusted.csv"))
        show_table(res, title="per-patient median Pearson r (rows: targets, columns: variants):")
        show_table(res_adj, title="reliability-adjusted r / sqrt(ceiling):")
        ctx.set("results", res)
        ctx.set("results_adj", res_adj)

        # ---- multi-seed stability -------------------------------------------------------
        ms_dir = os.path.join(os.path.dirname(d), "training", "multiseed")
        multiseed = None
        if os.path.isdir(ms_dir):
            rows = []
            for f in sorted(os.listdir(ms_dir)):
                s = int(f[len("seed"):-4])
                p = np.load(os.path.join(ms_dir, f))
                rows.append(score(p, Y, targets, groups, ctx.rel_lookup, cfg.min_cells_per_patient_metric, sids).assign(seed=s))
            if rows:
                multiseed = pd.concat(rows).groupby("target")["pearson_patient_median"].agg(["mean", "std"]).round(3)
                multiseed.to_csv(os.path.join(d, "cv_multiseed.csv"))
                show_table(multiseed, title="full model over seeds (mean, sd of the per-patient median r):")
        ctx.set("multiseed", multiseed if multiseed is not None else pd.DataFrame())

        # ---- calibration -----------------------------------------------------------------
        section("evaluation: calibration of the predictive uncertainty (full model)")
        unc = (ctx.cv_unc or {}).get("full")
        if unc is not None and np.isfinite(unc).any() and ctx.folds is None:
            say("no fold definitions on the context; calibration from folds not possible")
            unc = None
        if unc is not None and np.isfinite(unc).any():
            val_preds = _load_val_preds(ctx, len(targets))
            cal = calibration_from_folds(np.asarray(ctx.cv_preds["full"], dtype=np.float32), np.asarray(unc, dtype=np.float32),
                                         Y, groups, targets, ctx.folds, alpha=cfg.conformal_alpha, levels=cfg.coverage_levels,
                                         val_preds=val_preds)
        else:
            say("no predictive uncertainty available (uncertainty = none); calibration table empty")
            cal = pd.DataFrame(columns=["target", "level", "coverage_pooled", "kind"])
        cal.to_csv(os.path.join(d, "calibration.csv"), index=False)
        ctx.set("calibration", cal)

        # ---- decision rules (v3 ablation names mapped onto the v2 rule inputs) ---------------
        section("evaluation: decision rules")
        res_rules = res.copy()
        if "no_neighbours" in res_rules.columns and "no_spatial" not in res_rules.columns:
            res_rules["no_spatial"] = res_rules["no_neighbours"]      # the rule asks whether spatial context earns anything
        decisions, verdict = apply_decision_rules(res_rules, res_adj, main_target)
        decisions += self._v3_statements(res, main_target)
        with open(os.path.join(d, "decisions.txt"), "w") as fh:
            fh.write("\n".join(decisions) + "\n")
            fh.write("\nverdict: " + json.dumps(verdict) + "\n")
        ctx.set("decisions", decisions)
        ctx.set("verdict", verdict)

        # ---- patient-level endpoints ------------------------------------------------------
        section("evaluation: patient-level burden, stage trend, PERMANOVA")
        _, stage_order = load_stage_vocabulary()
        pred_full = np.asarray(ctx.cv_preds["full"], dtype=np.float32)
        burden = program_burden(ctx.obs, pred_full, targets.index(main_target), cfg, stage_order)
        burden.to_csv(os.path.join(d, "program_burden.csv"), index=False)
        burden_stats = stage_trend(burden)
        if cfg.clinical_csv:
            cox_pfs(burden, cfg.clinical_csv, os.path.join(d, "cox_pfs.csv"))
        ctx.set("burden", burden)
        ctx.set("burden_stats", burden_stats)
        stage_test = str(getattr(cfg, "stage_test", "permanova")).lower()
        if stage_test not in ("permanova", "manova", "both"):
            warn(f"stage_test={stage_test!r} not recognised; using permanova")
            stage_test = "permanova"
        say(f"stage test: {stage_test} (PERMANOVA and the trend test always run; MANOVA only when asked for)")
        ctx.set("stage_assoc", self._stage_association(ctx, pred_full, targets, d))
        if stage_test in ("manova", "both"):
            ctx.set("manova", self._manova(ctx, pred_full, targets, d))
        else:
            say("classical MANOVA not run (set evaluation.stage_test to manova or both to add it)")

    # ------------------------------------------------------------------
    @staticmethod
    def _v3_statements(res: pd.DataFrame, main_target: str) -> List[str]:
        """Plain sentences for the v3-specific ablations on the main target."""
        out = []
        if main_target not in res.index or "full" not in res.columns:
            return out
        full = float(res.loc[main_target, "full"])
        for col, what in (("no_program_tokens", "program-token cross-attention"), ("no_grid", "patch-block tokens"),
                          ("no_align", "contrastive alignment to expression"), ("no_smooth", "neighbourhood consistency term")):
            if col in res.columns and np.isfinite(res.loc[main_target, col]):
                v = float(res.loc[main_target, col])
                gain = full - v
                verdict = "earns" if gain > 0.02 else ("costs" if gain < -0.02 else "neither helps nor hurts")
                out.append(f"ABLATION {col}: full ({full:.3f}) vs without ({v:.3f}); the {what} {verdict} "
                           f"{abs(gain):.3f} on {main_target}.")
        # the alignment loss is off in `full` and switched on by the `align` variant, so its sentence reads the other way
        if "align" in res.columns and np.isfinite(res.loc[main_target, "align"]):
            v = float(res.loc[main_target, "align"])
            diff = v - full
            verdict = "improves" if diff > 0.02 else ("worsens" if diff < -0.02 else "neither helps nor hurts")
            out.append(f"ABLATION align: full ({full:.3f}) vs with the contrastive alignment loss switched on ({v:.3f}); "
                       f"switching the alignment loss on {verdict} {main_target} by {abs(diff):.3f}.")
        if "anchor_only_targets" in res.columns:
            anchor_t = main_target.replace("prog_", "anchor_")
            if anchor_t in res.index and np.isfinite(res.loc[anchor_t, "anchor_only_targets"]):
                v = float(res.loc[anchor_t, "anchor_only_targets"])
                out.append(f"LEARNED VS FIXED TARGETS: learned {main_target} r = {full:.3f}; fixed-set {anchor_t} r = {v:.3f}; "
                           f"learned programs are {'more' if full > v + 0.02 else 'not more'} predictable from morphology.")
        for s in out:
            say(s, indent=1)
        return out

    @staticmethod
    def _stage_variables(targets: List[str]) -> List[str]:
        """Neighbourhood-scale learned programs; the composite wnt_camdr set is omitted when its subsets are present."""
        use = [t for t in targets if t.startswith("prog_") and "wnt_camdr" not in t and t.endswith("_nb_z")]
        if not use:
            use = [t for t in targets if t.endswith("_nb_z")][:5]
        return use

    STAGE_COLLAPSE = {"control": "control", "MGUS": "precursor", "SMM": "precursor", "MM": "myeloma", "RM": "myeloma"}
    COLLAPSED_ORDER = {"control": 0, "precursor": 1, "myeloma": 2}

    def _stage_association(self, ctx: Context, pred_full: np.ndarray, targets: List[str], d: str) -> pd.DataFrame:
        """PERMANOVA + Jonckheere-Terpstra on the patient-level predicted (and measured) program vectors."""
        from ..evaluation.manova import patient_level_table
        from ..evaluation.stage_association import RESULT_COLUMNS, stage_association_table

        _, full_order = load_stage_vocabulary()
        use = self._stage_variables(targets)
        say("stage association variables: " + ", ".join(use))
        obs2 = ctx.obs.copy()
        obs2["stage"] = obs2["stage"].astype(str)
        Y = np.asarray(ctx.Y, dtype=np.float32)
        cols = [targets.index(t) for t in use]
        for t, j in zip(use, cols):
            if t not in obs2.columns:
                obs2[t] = Y[:, j]
        table = patient_level_table(obs2, pred_full[:, cols], use)
        n_perm = int(getattr(ctx.cfg, "stage_n_perm", 999))
        # the full stage order is used when every stage has at least 2 patients; otherwise the
        # control / precursor / myeloma collapse the MANOVA used, so the trend test keeps 3 ordered groups
        counts = table["stage"].value_counts()
        collapse = None if (counts >= 2).all() and len(counts) >= 3 else self.STAGE_COLLAPSE
        order = full_order if collapse is None else self.COLLAPSED_ORDER
        frames = []
        for label, columns in (("predicted", [f"{t}_pred" for t in use]), ("measured", use)):
            say(f"--- {label} program vectors ---")
            res = stage_association_table(table, columns, stage_col="stage", stage_order=order, n_perm=n_perm,
                                          seed=ctx.seed, collapse_rule=collapse)
            res.insert(0, "vectors", label)
            res["program"] = res["program"].str.replace("_pred$", "", regex=True)
            frames.append(res)
        out = pd.concat(frames, ignore_index=True)[["vectors"] + RESULT_COLUMNS]
        out.to_csv(os.path.join(d, "stage_association.csv"), index=False)
        table.to_csv(os.path.join(d, "stage_patient_table.csv"), index=False)
        say(f"stage association -> {os.path.join(d, 'stage_association.csv')}")
        return out

    def _manova(self, ctx: Context, pred_full: np.ndarray, targets: List[str], d: str) -> pd.DataFrame:
        try:
            from ..evaluation.manova import run_manova_stage
        except Exception as e:  # statsmodels missing
            warn(f"MANOVA skipped: {e}")
            tab = pd.DataFrame([dict(test="skipped", reason=str(e))])
            tab.to_csv(os.path.join(d, "manova_results.csv"), index=False)
            return tab
        use = self._stage_variables(targets)
        say("MANOVA variables: " + ", ".join(use) + " (composite wnt_camdr omitted when its subsets are present)")
        collapse = {"control": "control", "MGUS": "precursor", "SMM": "precursor", "MM": "myeloma", "RM": "myeloma"}
        obs2 = ctx.obs.copy()
        obs2["stage"] = obs2["stage"].astype(str).map(lambda s: collapse.get(s, s))
        say("stages collapsed for MANOVA: " + ", ".join(f"{k}->{v}" for k, v in collapse.items()))
        cols = [targets.index(t) for t in use]
        # the MANOVA helper reads the measured targets from obs; make sure the columns exist (ctx.Y is the source)
        Y = np.asarray(ctx.Y, dtype=np.float32)
        for t, j in zip(use, cols):
            if t not in obs2.columns:
                obs2[t] = Y[:, j]
        order = {"control": 0, "precursor": 1, "myeloma": 2}
        return run_manova_stage(obs2, pred_full[:, cols], use, _ArtefactDir(d), ctx.cfg, order)

    # ------------------------------------------------------------------
    def load(self, ctx: Context) -> None:
        d = self.dir(ctx)
        res = pd.read_csv(os.path.join(d, "cv_results.csv"), index_col=0)
        if "full_95ci" in res.columns:
            res["full_95ci"] = res["full_95ci"].fillna("").astype(str)   # empty CI cells round-trip as NaN
        ctx.set("results", res)
        ctx.set("results_adj", pd.read_csv(os.path.join(d, "cv_results_reliability_adjusted.csv"), index_col=0))
        ctx.set("calibration", pd.read_csv(os.path.join(d, "calibration.csv")))
        with open(os.path.join(d, "decisions.txt")) as fh:
            lines = [ln.rstrip("\n") for ln in fh]
        verdict = {}
        statements = []
        for ln in lines:
            if ln.startswith("verdict: "):
                verdict = json.loads(ln[len("verdict: "):])
            elif ln.strip():
                statements.append(ln)
        ctx.set("decisions", statements)
        ctx.set("verdict", verdict)
        ctx.set("stage_assoc", pd.read_csv(os.path.join(d, "stage_association.csv")))
        manova_path = os.path.join(d, "manova_results.csv")
        if os.path.exists(manova_path):
            ctx.set("manova", pd.read_csv(manova_path))
        burden = pd.read_csv(os.path.join(d, "program_burden.csv"))
        ctx.set("burden", burden)
        ctx.set("burden_stats", stage_trend(burden) if len(burden) else (np.nan, np.nan, 0))
        ms = os.path.join(d, "cv_multiseed.csv")
        ctx.set("multiseed", pd.read_csv(ms, index_col=0) if os.path.exists(ms) else pd.DataFrame())
        say("evaluation loaded from artefacts")


def evaluate(ctx: Context) -> Context:
    """Component 8: per-patient metrics against ceilings, baselines, calibration, decision rules, stage association.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Evaluation().run_in(ctx)
