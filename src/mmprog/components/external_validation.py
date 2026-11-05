"""ExternalValidation component: modality-agnostic validation beyond the cross-validation metrics.

Why this component exists
-------------------------
A cross-validated correlation on held-out patients is necessary but not sufficient. It
does not say whether the prediction tracks the program rather than any gene set of the
same expression level, whether the predicted map has the tissue's spatial organisation,
whether a different scanner would give the same map, or how the model behaves on a
cohort it never saw. The original design reserved the orthogonal check for a CODEX
bridge, but no public CODEX myeloma dataset pairs with Xenium, so that gate never opened.
This component runs every validator that the available data allow and records, for each
one, whether it ran, was skipped (and why) or failed (and how), so a run never silently
loses a check.

Validators (``src/mmprog/validation/``)
  1. program_specificity   always; expression-matched random gene sets and split-gene halves
  2. spatial_fidelity      always; Moran's I and plasma-cell-dense niche contrast, predicted vs measured
  3. perturbation_robustness  when a checkpoint and a slide image exist; gamma, blur, rotation, noise, scale
  4. independent_cohort    when ``validation_cohort_manifest`` is set (Xenium or Visium HD rows, or a synthetic demo)
  5. protein_bridge        when ``codex_table`` and ``codex_dapi`` are set (CODEX / PhenoCycler)

Artefacts: ``validation_summary.json`` and one CSV per validator that ran.
"""
from __future__ import annotations

import json
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, section, warn

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z", "anchor_wnt_camdr_nb_z"]
VALIDATORS = ("program_specificity", "spatial_fidelity", "perturbation_robustness", "independent_cohort", "protein_bridge")


def _judge(*conditions: Any) -> bool:
    """A validator's verdict from its stated bar: every condition must hold (NaN never holds)."""
    return bool(all(bool(c) for c in conditions))


def _main_target(targets: List[str]) -> str:
    for t in MAIN_TARGET_PREFERENCE:
        if t in targets:
            return t
    return targets[0]


def _learned_nb_targets(targets: List[str], main: str) -> List[str]:
    """The main target first, then every other learned neighbourhood-scale target."""
    rest = [t for t in targets if t.startswith("prog_") and t.endswith("_nb_z") and t != main]
    return ([main] if main in targets else []) + rest


class ExternalValidation(BaseComponent):
    name = "external_validation"
    requires = ("cv_preds", "Y", "targets", "obs", "adata", "gene_mask", "program_weights", "programs", "GROUPS", "NB",
                "lineage", "lin_names")
    provides = ("validation", "validation_tables")
    artefacts = ("validation_summary.json",)
    depends_on = ("training",)
    config_fields = ("validation_n_random_sets", "validation_perturbations", "validation_cohort_manifest",
                     "codex_table", "codex_dapi", "codex_um_per_px", "codex_dapi_um_per_px", "nb_radius_um", "knn_k",
                     "encoder", "patch_px", "img_level", "z_project", "stain_mode", "target_spaces", "target_norm")

    # hooks a test can override (same names as virtual_program_map_v3); strict re-raises validator errors
    vmap_hooks: Dict[str, Any] = {}
    strict: bool = False

    # ------------------------------------------------------------------ shared pieces
    def _vmap_fn(self, ctx: Context) -> Callable:
        from ..inference.virtual_map_v3 import virtual_program_map_v3

        hooks = dict(self.vmap_hooks)

        def run(img: np.ndarray, ck: Dict[str, Any], cfg, device: str, sample: str, detect_fn: Optional[Callable],
                um_per_px: Optional[float] = None) -> pd.DataFrame:
            kw = dict(hooks)
            if detect_fn is not None:
                kw["detect_fn"] = detect_fn
            return virtual_program_map_v3(img, ck, cfg, device, sample_name=sample, um_per_px=um_per_px, **kw)

        return run

    def _checkpoint(self, ctx: Context) -> Optional[Dict[str, Any]]:
        if not ctx.final_model_path or not os.path.exists(str(ctx.final_model_path)):
            return None
        from ..inference.virtual_map_v3 import load_checkpoint
        return load_checkpoint(ctx.final_model_path, ctx.device)

    def _guard(self, name: str, fn: Callable[[], Tuple[dict, Optional[pd.DataFrame]]], summary: dict,
               tables: Dict[str, pd.DataFrame], ctx: Context) -> None:
        section(f"external validation: {name}")
        try:
            result, table = fn()
        except Exception as e:                                   # one broken validator must not lose the others
            if self.strict or os.environ.get("MMPROG_STRICT_VALIDATION"):
                raise
            warn(f"{name} failed: {type(e).__name__}: {e}")
            summary[name] = dict(status="failed", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc().splitlines()[-3:])
            return
        summary[name] = result
        if table is not None and len(table):
            path = self.path(ctx, f"{name}.csv")
            table.to_csv(path, index=False)
            tables[name] = table
            summary[name]["table"] = os.path.basename(path)
        say(f"{name}: {result.get('status')}" + (f" ({result.get('reason')})" if result.get("reason") else ""))

    # ------------------------------------------------------------------ validators
    def _program_specificity(self, ctx: Context):
        from ..validation.program_specificity import program_specificity, specificity_headline
        cfg = ctx.cfg
        preds = np.asarray(ctx.cv_preds["full"], dtype=np.float32) if "full" in ctx.cv_preds else None
        if preds is None:
            return dict(status="skipped", reason="no out-of-fold predictions of the full model"), None
        main = _main_target(list(ctx.targets))
        tests = _learned_nb_targets(list(ctx.targets), main)
        if not tests:
            return dict(status="skipped", reason="no learned targets to test (programs_n_extra and anchored programs absent)"), None
        table, summ = program_specificity(ctx.adata, ctx.gene_mask, ctx.program_weights, ctx.obs, cfg, ctx.um_per_px_by_sample(),
                                          preds, list(ctx.targets), np.asarray(ctx.Y, dtype=np.float32), ctx.GROUPS, tests,
                                          n_sets=int(cfg.validation_n_random_sets), seed=ctx.seed)
        if table.empty:
            return dict(status="skipped", reason="no patient had enough cells with finite predictions"), None
        summ.to_csv(self.path(ctx, "program_specificity_summary.csv"), index=False)
        head = specificity_headline(summ, main)
        head["passed"] = _judge(head.get("median_z", np.nan) > 2,
                                head.get("median_r_half_a", np.nan) > 0, head.get("median_r_half_b", np.nan) > 0)
        head["pass_bar"] = "median_z > 2 on the main target and both split-gene halves correlate positively"
        head.update(status="ran", reason=None, n_random_sets=int(cfg.validation_n_random_sets),
                    summary_table="program_specificity_summary.csv",
                    reading="median_z above 2 (null spread floored at 1/sqrt(n cells)), a positive margin over the null 95th percentile and both half-correlations positive mean the prediction tracks the program, "
                            "not any gene set of the same expression level")
        return head, table

    def _spatial_fidelity(self, ctx: Context):
        from ..validation.spatial_fidelity import spatial_fidelity
        preds = np.asarray(ctx.cv_preds["full"], dtype=np.float32) if "full" in ctx.cv_preds else None
        if preds is None:
            return dict(status="skipped", reason="no out-of-fold predictions of the full model"), None
        main = _main_target(list(ctx.targets))
        tests = _learned_nb_targets(list(ctx.targets), main) or [main]
        table, head = spatial_fidelity(preds, np.asarray(ctx.Y, dtype=np.float32), list(ctx.targets), tests, ctx.GROUPS, ctx.NB,
                                       ctx.obs, ctx.lineage, ctx.lin_names, ctx.cfg, ctx.um_per_px_by_sample())
        if table.empty:
            return dict(status="skipped", reason="no patient had enough cells"), None
        ratio = head.get("median_moran_ratio", np.nan)
        sign = head.get("niche_sign_agreement", np.nan)
        head["passed"] = _judge(0.5 <= ratio <= 2.0 if np.isfinite(ratio) else False,
                                sign if isinstance(sign, (bool, np.bool_)) else False)
        head["pass_bar"] = "0.5 <= median Moran ratio <= 2 and the plasma-dense niche contrast has the measured sign, on the main target"
        head.update(status="ran", reason=None,
                    reading="moran_ratio near 1 means the predicted map is as spatially organised as the tissue; "
                            "niche sign agreement means the plasma-dense contrast has the measured direction")
        return head, table

    def _perturbation_robustness(self, ctx: Context, ck: Optional[Dict[str, Any]]):
        from ..preprocessing.nuclei import detect_nuclei
        from ..validation.perturbations import parse_perturbation
        from ..validation.robustness import perturbation_robustness
        tokens = list(ctx.cfg.validation_perturbations or [])
        for t in tokens:
            parse_perturbation(t)
        if ck is None:
            return dict(status="skipped", reason="no final checkpoint (final_model_path unset or missing)"), None
        if ctx.images is None:
            return dict(status="skipped", reason="no image store on the context"), None
        if not tokens:
            return dict(status="skipped", reason="validation_perturbations is empty"), None
        s0 = str(ctx.obs["sample"].iloc[0])
        img = ctx.images.get(s0)
        xy = detect_nuclei(img)
        if len(xy) < 10:
            return dict(status="skipped", reason=f"only {len(xy)} nuclei detected on {s0}"), None
        say(f"slide {s0}: {len(xy)} nuclei kept fixed across {len(tokens)} perturbations")
        targets = [t for t in ck["targets"]]
        main = _main_target(list(targets))
        ordered = [main] + [t for t in targets if t != main]
        um_px = ctx.um_per_px_by_sample().get(s0, float(ctx.cfg.um_per_px))
        table, head = perturbation_robustness(img, xy, ck, ctx.cfg, ctx.device, tokens, ordered, self._vmap_fn(ctx), sample=s0,
                                              seed=ctx.seed, um_per_px=um_px)
        if table.empty:
            return dict(status="skipped", reason="no comparable predictions"), None
        head["passed"] = _judge(head.get("min_spearman_main", np.nan) > 0.9)
        head["pass_bar"] = "Spearman between perturbed and clean predictions > 0.9 on the main target under every perturbation"
        head.update(status="ran", reason=None, slide=s0, n_nuclei=int(len(xy)), perturbations=tokens,
                    reading="spearman_vs_clean above 0.9 on the main target under every perturbation is the working bar; "
                            "a proxy for scanner shift, not a multi-site study")
        return head, table

    def _independent_cohort(self, ctx: Context, ck: Optional[Dict[str, Any]]):
        from ..validation.independent_cohort import cohort_status_note, load_cohort, read_cohort_manifest, score_cohort
        path = ctx.cfg.validation_cohort_manifest
        if not path:
            return dict(status="skipped", reason=cohort_status_note()), None
        if not os.path.exists(path):
            return dict(status="skipped", reason=f"validation_cohort_manifest not found: {path}"), None
        if ck is None:
            return dict(status="skipped", reason="no final checkpoint to score the cohort with"), None
        from ..resources import load_lineages
        manifest = read_cohort_manifest(path)
        adata, store, rows = load_cohort(manifest, ctx.cfg, ctx.programs, load_lineages(), ctx.seed)
        usable = [g for g, m in zip(ctx.adata.var_names, np.asarray(ctx.gene_mask, dtype=bool)) if m]
        table, head = score_cohort(adata, store, rows, ck, ctx.cfg, ctx.device, ctx.program_weights, usable, ctx.programs,
                                   self._vmap_fn(ctx), ctx.seed)
        if table.empty:
            head.update(status="skipped", reason="no cohort slide had enough matched nuclei")
            return head, None
        head["passed"] = _judge(bool(head.get("above_chance", False)))
        head["pass_bar"] = "the main target is above chance on the cohort (median r minus shuffled > 0.05)"
        head.update(status="ran", reason=None, manifest=path,
                    reading="above_chance uses the same 0.05 margin over a within-slide permutation as the main cross-validation; "
                            "learned targets are measured as a linear proxy on the cohort's genes")
        return head, table

    def _protein_bridge(self, ctx: Context, ck: Optional[Dict[str, Any]]):
        from ..validation.protein_bridge import protein_bridge
        cfg = ctx.cfg
        if not cfg.codex_table or not cfg.codex_dapi:
            return dict(status="skipped",
                        reason="set codex_table (cell-by-marker CSV/parquet with x, y) and codex_dapi (DAPI image of the same section); "
                               "no public dataset pairs CODEX with Xenium on the same myeloma trephine"), None
        if ck is None:
            return dict(status="skipped", reason="no final checkpoint"), None
        result, contrast = protein_bridge(cfg, ck, ctx.device, list(ctx.targets), list(ctx.lin_names), self._vmap_fn(ctx), self.dir(ctx))
        result.setdefault("reason", None)
        status = str(result.get("status", ""))
        if status in ("aborted", "insufficient_matches"):          # neither ran nor broke: a skip with its reason kept
            result["reason"] = f"{status}: {result.get('reason') or ''}".strip(": ")
            result["status"] = "skipped"
        elif status == "ran":
            comp = result.get("composition") or {}
            acc = comp.get("cell_agreement", np.nan)
            result["passed"] = _judge(float(acc) > 0.5) if isinstance(acc, (int, float, np.floating)) and np.isfinite(acc) else None
            result["pass_bar"] = ("composition-head lineage agrees with the protein lineage on more than half of the matched nuclei; "
                                  "unjudged when the model has no composition head")
        return result, contrast

    # ------------------------------------------------------------------ component API
    def _run(self, ctx: Context) -> None:
        summary: Dict[str, dict] = {}
        tables: Dict[str, pd.DataFrame] = {}
        say(f"targets: {len(ctx.targets)}; main target {_main_target(list(ctx.targets))}; "
            f"validators: {', '.join(VALIDATORS)}")
        ck = self._checkpoint(ctx)
        if ck is None:
            say("no final checkpoint on the context; image-based validators will be skipped")
        self._guard("program_specificity", lambda: self._program_specificity(ctx), summary, tables, ctx)
        self._guard("spatial_fidelity", lambda: self._spatial_fidelity(ctx), summary, tables, ctx)
        self._guard("perturbation_robustness", lambda: self._perturbation_robustness(ctx, ck), summary, tables, ctx)
        self._guard("independent_cohort", lambda: self._independent_cohort(ctx, ck), summary, tables, ctx)
        self._guard("protein_bridge", lambda: self._protein_bridge(ctx, ck), summary, tables, ctx)
        self._finish(ctx, summary, tables)

    def _finish(self, ctx: Context, summary: Dict[str, dict], tables: Dict[str, pd.DataFrame]) -> None:
        counts = {s: sum(1 for v in summary.values() if v.get("status") == s) for s in ("ran", "skipped", "failed")}
        summary["_overall"] = dict(ran=counts["ran"], skipped=counts["skipped"], failed=counts["failed"],
                                   validators=[v for v in VALIDATORS if v in summary])
        with open(self.path(ctx, "validation_summary.json"), "w") as fh:
            json.dump(_sanitize(summary), fh, indent=2, default=_json_default)
        section("external validation: summary")
        for name in VALIDATORS:
            v = summary.get(name, {})
            verdict = "" if v.get("passed") is None else (" PASS" if v.get("passed") else " FAIL")
            say(f"{name:24s} {v.get('status', 'missing'):8s}{verdict} " + (v.get("reason") or self._one_line(v)))
        n_pass = sum(1 for v in summary.values() if v.get("passed") is True)
        n_fail = sum(1 for v in summary.values() if v.get("passed") is False)
        say(f"verdicts: {n_pass} passed their stated bar, {n_fail} did not (the release gate reads these)")
        say(f"{counts['ran']} ran, {counts['skipped']} skipped, {counts['failed']} failed -> validation_summary.json")
        ctx.set("validation", summary)
        ctx.set("validation_tables", tables)
        # the reporting component reads ctx.codex in the v2 shape; the bridge now lives here
        ctx.set("codex", summary.get("protein_bridge", {"status": "skipped", "reason": "not run"}))

    @staticmethod
    def _one_line(v: dict) -> str:
        keys = [k for k in ("median_z", "median_moran_ratio", "min_spearman_main", "median_r", "n_matched") if k in v]
        return ", ".join(f"{k}={v[k]:.3f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in keys)

    def load(self, ctx: Context) -> None:
        with open(self.path(ctx, "validation_summary.json")) as fh:
            summary = json.load(fh)
        tables: Dict[str, pd.DataFrame] = {}
        for name in VALIDATORS:
            p = self.path(ctx, f"{name}.csv")
            if os.path.exists(p):
                tables[name] = pd.read_csv(p)
        ctx.set("validation", summary)
        ctx.set("validation_tables", tables)
        ctx.set("codex", summary.get("protein_bridge", {"status": "skipped", "reason": "not run"}))
        ov = summary.get("_overall", {})
        say(f"external validation loaded: {ov.get('ran', '?')} ran, {ov.get('skipped', '?')} skipped, "
            f"{ov.get('failed', '?')} failed; tables: {list(tables)}")


def _sanitize(o: Any) -> Any:
    """NaN and numpy scalars to JSON-safe values (NaN becomes null, which every reader accepts)."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def validate_externally(ctx: Context) -> Context:
    """Component 10: modality-agnostic validation; every validator records ran, skipped or failed with its reason.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return ExternalValidation().run_in(ctx)
