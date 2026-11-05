"""Model card: what the model is, what it was trained on, what the numbers are, what it must not be used for."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import say
from ..paths import RunDir


def write_model_card(run: RunDir, cfg: Config, obs: pd.DataFrame, targets: List[str],
                     kept_programs: Dict[str, List[str]], morpho_names: List[str],
                     reliability_summary: pd.DataFrame, res: pd.DataFrame, res_adj: Optional[pd.DataFrame],
                     decisions: List[str], verdict: Dict[str, bool], phase_status: pd.DataFrame,
                     selection_report: Optional[dict] = None) -> Dict[str, Any]:
    groups = obs["patient"].values
    card: Dict[str, Any] = dict(
        name="mmprog: morphology to investigator-defined gene-program scores, multiple myeloma",
        version="1.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M"),
        demo_mode=cfg.demo_mode,
        task=("Predict per-cell and 30 um neighbourhood gene-program scores from DAPI nuclear morphology patches; "
              "morphology-only inference via classical nucleus detection"),
        training_data=dict(
            source=cfg.geo_accession if not cfg.demo_mode else "synthetic fixture",
            platform=cfg.platform,
            morphology="synthetic" if cfg.demo_mode else "DAPI (Xenium morphology.ome.tif, level %d, %s projection)" % (cfg.img_level, cfg.z_project),
            n_cells=int(len(obs)),
            n_patients=int(len(np.unique(groups))),
            samples=list(map(str, obs["sample"].unique())),
            stages=obs.groupby("sample")["stage"].first().value_counts().to_dict(),
        ),
        features=dict(encoder=cfg.encoder, patch_px=cfg.patch_px, pca_dim=cfg.pca_dim,
                      morphometrics=morpho_names if cfg.use_morpho else [],
                      feature_selection=selection_report or "disabled"),
        targets=targets,
        program_genes=kept_programs,
        target_norm=cfg.target_norm,
        nb_radius_um=cfg.nb_radius_um,
        target_reliability=json.loads(reliability_summary.to_json()),
        evaluation=("patient-level GroupKFold (%d folds), inner validation patient for early stopping; "
                    "per-patient median Pearson with bootstrap 95%% CI; reliability-adjusted r = r / sqrt(r_sb)" % cfg.n_folds),
        results=json.loads(res.drop(columns=[c for c in ["full_95ci"] if c in res.columns]).to_json()),
        results_ci={t: str(c) for t, c in res["full_95ci"].items()} if "full_95ci" in res.columns else {},
        results_reliability_adjusted=json.loads(res_adj.to_json()) if res_adj is not None else None,
        decision_statements=decisions,
        verdict=verdict,
        phase_status=phase_status.to_dict("records"),
        not_for="clinical decision making; any cohort, stain or platform outside the training distribution without re-validation",
        program_caveats=("Gene sets are investigator-defined abundances associated with drug resistance in the literature, not "
                         "pathway activity; WNT10A, NCAM1 and BCL2 carry evidence pointing toward drug sensitivity or better outcome; "
                         "no drug-response data exist in the training cohort. See docs/research/PATHWAY_INTERPRETATION.md."),
        config=cfg.to_dict(),
    )
    with open(run.art("model_card"), "w") as fh:
        json.dump(card, fh, indent=2, default=str)
    say(f"model card -> {run.art('model_card')}")
    say(json.dumps({k: card[k] for k in ("name", "version", "training_data", "verdict")}, indent=2, default=str))
    return card
