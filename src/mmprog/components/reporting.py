"""Reporting: model card, figures, QA tiers and CONTRIBUTIONS.md.

Nothing is computed here. Every number is read from a context field that an earlier
component wrote to disk, and the v2 writers are reused unchanged through a small adapter
that maps their artefact names onto this component's directory (D23).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, warn

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z", "anchor_wnt_camdr_nb_z"]


class _ArtefactDir:
    """Adapter with the ``.art(name)`` interface the v2 reporting writers expect."""

    NAMES: Dict[str, str] = {
        "model_card": "model_card.json",
        "contributions": "CONTRIBUTIONS.md",
        "tier2": "qa_tier2.csv",
        "tier3": "qa_tier3.csv",
        "overlays": os.path.join("figures", "overlays.png"),
    }

    def __init__(self, root: str):
        self.path = root
        os.makedirs(os.path.join(root, "figures"), exist_ok=True)

    def art(self, name: str) -> str:
        return os.path.join(self.path, self.NAMES.get(name, name))

    def done(self, name: str) -> bool:
        return os.path.exists(self.art(name))


def main_target_of(targets) -> str:
    for t in MAIN_TARGET_PREFERENCE:
        if t in targets:
            return t
    return targets[0]


class Reporting(BaseComponent):
    name = "reporting"
    requires = ("results", "decisions", "reliability_summary", "obs", "targets", "cv_preds")
    provides = ("model_card", "contributions_path", "gates")
    artefacts = ("model_card.json", "CONTRIBUTIONS.md", "release_gates.csv", os.path.join("figures", "cv_results.png"),
                 os.path.join("figures", "reliability.png"), "qa_tier2.csv", "qa_tier3.csv")
    config_fields = ("run_name", "n_folds", "encoder", "patch_px", "pca_dim", "target_norm", "nb_radius_um",
                     "srb_grid", "demo_mode", "clinical_csv", "commpass_csv", "codex_table", "codex_dapi", "slide_dir",
                     "stain_mode", "um_per_px_level0", "img_level")
    depends_on = ("evaluation", "interpretation", "external_validation", "deployment")

    # ------------------------------------------------------------------ pieces
    def _phase_status(self, ctx: Context) -> pd.DataFrame:
        cfg = ctx.cfg
        cohorts = sorted(ctx.obs["cohort"].unique()) if "cohort" in ctx.obs else []
        return pd.DataFrame([
            dict(phase=1, name="Xenium within-cohort", data_present=True, ran=ctx.results is not None),
            dict(phase=2, name="Visium HD + transfer", data_present=len(cohorts) >= 2, ran=False),
            dict(phase="2b", name="CODEX proteomics bridge", data_present=bool(cfg.codex_table and cfg.codex_dapi),
                 ran=bool(ctx.codex) and ctx.codex.get("status", "") not in ("", "skipped")),
            dict(phase=3, name="Clinical endpoints", data_present=bool(cfg.clinical_csv or cfg.commpass_csv), ran=False),
            dict(phase=4, name="Encoder / stain harmonisation", data_present=True,
                 ran=cfg.encoder != "resnet50" or cfg.stain_mode == "hematoxylin"),
            dict(phase=5, name="Batch deployment", data_present=bool(cfg.slide_dir), ran=bool(ctx.exports)),
        ])

    def _main_predictions(self, ctx: Context) -> Optional[np.ndarray]:
        if ctx.cv_preds and "full" in ctx.cv_preds:
            return np.asarray(ctx.cv_preds["full"])
        if ctx.cv_preds:
            first = next(iter(ctx.cv_preds))
            warn(f"no 'full' variant among cross-validated predictions; overlays and QA tiers use '{first}'")
            return np.asarray(ctx.cv_preds[first])
        return None

    def _figures(self, ctx: Context, art: _ArtefactDir, pred: Optional[np.ndarray], main_target: str, main_idx: int) -> None:
        from ..reporting.figures import importance_map_figure, overlay_figure, reliability_figure, results_figure
        results_figure(ctx.results, art.art(os.path.join("figures", "cv_results.png")))
        reliability_figure(ctx.reliability_summary, ctx.results, art.art(os.path.join("figures", "reliability.png")))
        if pred is not None and ctx.images is not None:
            overlay_figure(ctx.obs, pred, main_idx, main_target, ctx.images, art.art("overlays"))
        else:
            say("overlay figure skipped: no held-out predictions or no image store in the context")
        if ctx.srb_maps:
            block_um = ctx.cfg.patch_px / ctx.cfg.srb_grid * ctx.cfg.um_per_px
            importance_map_figure(ctx.srb_maps, ctx.cfg.srb_grid, block_um, art.art(os.path.join("figures", "srb_importance_maps.png")))

    # ------------------------------------------------------------------ component API
    def _run(self, ctx: Context) -> None:
        from ..reporting.contributions import write_contributions
        from ..reporting.model_card import write_model_card
        from ..reporting.qa_tiers import flag_tiers

        art = _ArtefactDir(self.dir(ctx))
        targets = list(ctx.targets) if ctx.targets else list(ctx.results.index)
        main_target = main_target_of(targets)
        main_idx = targets.index(main_target)
        say(f"main target for figures and QA tiers: {main_target}")
        pred = self._main_predictions(ctx)

        if pred is not None:
            obs = ctx.obs
            if "lineage_name" not in obs:
                warn("obs has no lineage_name column (program inference did not run); QA tiers use 'unknown'")
                obs = obs.assign(lineage_name="unknown")
            flag_tiers(obs, pred, main_idx, main_target, art)
        else:
            warn("QA tiers skipped: no held-out predictions in the context; writing empty tier files")
            pd.DataFrame().to_csv(art.art("tier2"), index=False)
            pd.DataFrame().to_csv(art.art("tier3"), index=False)
        self._figures(ctx, art, pred, main_target, main_idx)

        phase_status = self._phase_status(ctx)
        phase_status.to_csv(os.path.join(self.dir(ctx), "phase_status.csv"), index=False)
        programs = ctx.programs or {}
        card = write_model_card(art, ctx.cfg, ctx.obs, targets, programs, ctx.morpho_names or [],
                                ctx.reliability_summary, ctx.results, ctx.results_adj, ctx.decisions,
                                ctx.verdict or {}, phase_status, selection_report=ctx.selection)
        card["encoder_card"] = ctx.encoder_card
        card["model_spec"] = ctx.model_spec
        card["hardware"] = ctx.hardware
        card["sources"] = {k: v.get("hash") for k, v in ctx.component_manifests.items()}
        # release gates: every check read from an artefact on disk, written before the model card and CONTRIBUTIONS.md quote them
        from ..reporting.gates import evaluate_gates, print_gates, write_gates
        gates = evaluate_gates(ctx.run_dir, main_target)
        print_gates(gates, demo_mode=bool(ctx.cfg.demo_mode))
        results = write_gates(ctx.run_dir, main_target, gates,
                              extra=dict(run_name=ctx.cfg.run_name, demo_mode=bool(ctx.cfg.demo_mode),
                                         decisions=list(ctx.decisions or []), verdict=ctx.verdict or {}))
        card["release_gates"] = results["gates"]
        with open(art.art("model_card"), "w") as fh:
            json.dump(card, fh, indent=2, default=str)
        text_path = art.art("contributions")
        write_contributions(art, ctx.cfg, ctx.results, ctx.results_adj, ctx.reliability_summary, ctx.decisions,
                            ctx.verdict or {}, main_target, burden_stats=ctx.burden_stats, multiseed=ctx.multiseed,
                            n_cells=len(ctx.obs), n_patients=int(ctx.obs["patient"].nunique()),
                            srb_results=ctx.srb_results, srb_rings=ctx.srb_rings, manova=ctx.manova,
                            motifs=ctx.motifs, codex=ctx.codex, calibration=ctx.calibration,
                            gene_backbone=ctx.gene_backbone, enrichment=ctx.enrichment, pathway_attr=ctx.pathway_attr,
                            gates=gates, stage_assoc=ctx.stage_assoc, validation=ctx.validation,
                            validation_tables=ctx.validation_tables)
        ctx.set("model_card", card)
        ctx.set("contributions_path", text_path)
        ctx.set("gates", results["gates"])

    def load(self, ctx: Context) -> None:
        with open(self.path(ctx, "model_card.json")) as fh:
            card = json.load(fh)
        say(f"loaded model_card.json ({card.get('name')}) and CONTRIBUTIONS.md")
        ctx.set("model_card", card)
        ctx.set("contributions_path", self.path(ctx, "CONTRIBUTIONS.md"))
        results_path = os.path.join(ctx.run_dir, "RESULTS.json")
        gates: Dict = {}
        if os.path.exists(results_path):
            with open(results_path) as fh:
                gates = json.load(fh).get("gates", {})
        else:
            warn("RESULTS.json not found next to this run; release gates unavailable on load")
        ctx.set("gates", gates)


def report(ctx: Context) -> Context:
    """Component 12: model card, figures, QA tiers, release gates, CONTRIBUTIONS.md; every number read from an artefact.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Reporting().run_in(ctx)
