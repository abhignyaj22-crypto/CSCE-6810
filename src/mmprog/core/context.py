"""The run context: every array, table and path that components pass to each other.

Fields are assigned once by the component that provides them. A component that must
replace a field goes through ``ctx.replace``, which logs the change, so the run log
shows where every object came from. Arrays are numpy or memmap, never torch
tensors, so the context pickles and the CPU-only test path works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config, resolve_device
from ..logging_utils import attach_log_file, say
from ..resources import seed_everything


@dataclass
class Fold:
    fold_id: int
    train_patients: List[str]
    val_patient: str
    test_patients: List[str]


@dataclass
class Context:
    # run
    cfg: Config
    run_dir: str
    device: str
    seed: int

    # Ingest
    manifest: Optional[List[dict]] = None
    adata_raw: Any = None
    images: Any = None
    full_xy: Optional[Dict[str, np.ndarray]] = None

    # Preprocess
    adata: Any = None
    obs: Optional[pd.DataFrame] = None
    gene_mask: Optional[np.ndarray] = None
    nuclei: Optional[Dict[str, np.ndarray]] = None

    # ProgramInference
    programs: Optional[Dict[str, List[str]]] = None
    program_weights: Optional[pd.DataFrame] = None
    program_scores: Optional[np.ndarray] = None
    targets: Optional[List[str]] = None
    target_family: Optional[Dict[str, str]] = None      # target -> "learned" | "anchor_only"
    Y: Optional[np.ndarray] = None
    lineage: Optional[np.ndarray] = None
    lin_names: Optional[List[str]] = None
    LIN: Optional[np.ndarray] = None
    reliability: Optional[pd.DataFrame] = None
    reliability_summary: Optional[pd.DataFrame] = None
    rel_lookup: Optional[Dict] = None
    expression_embedding: Optional[np.ndarray] = None   # (N, P) per-cell program-space vector for alignment

    # FeatureExtraction
    EMB: Any = None
    GRID: Any = None
    MOR: Optional[np.ndarray] = None
    morpho_names: Optional[List[str]] = None
    NB: Optional[np.ndarray] = None
    NB_DIST: Optional[np.ndarray] = None
    XY: Optional[np.ndarray] = None
    SID: Optional[np.ndarray] = None
    GROUPS: Optional[np.ndarray] = None
    n_samples: int = 0
    encoder_card: Optional[dict] = None

    # FeatureSelection
    selection: Optional[dict] = None

    # UnifiedModel
    model_spec: Optional[dict] = None

    # Training
    folds: Optional[List[Fold]] = None
    cv_preds: Optional[Dict[str, np.ndarray]] = None
    cv_unc: Optional[Dict[str, np.ndarray]] = None
    cv_comp: Optional[Dict[str, np.ndarray]] = None
    histories: Optional[Dict[str, pd.DataFrame]] = None
    final_model_path: Optional[str] = None
    hardware: Optional[dict] = None

    # Evaluation
    results: Optional[pd.DataFrame] = None
    results_adj: Optional[pd.DataFrame] = None
    calibration: Optional[pd.DataFrame] = None
    decisions: Optional[List[str]] = None
    verdict: Optional[Dict[str, bool]] = None
    manova: Optional[pd.DataFrame] = None
    burden: Optional[pd.DataFrame] = None
    burden_stats: Any = None
    multiseed: Optional[pd.DataFrame] = None

    stage_assoc: Optional[pd.DataFrame] = None            # PERMANOVA + Jonckheere-Terpstra trend by stage (replaces MANOVA by default)

    # Interpretation
    srb_results: Optional[pd.DataFrame] = None
    srb_maps: Optional[Dict[str, np.ndarray]] = None
    srb_rings: Optional[pd.DataFrame] = None
    pathway_attr: Optional[pd.DataFrame] = None
    gene_backbone: Optional[pd.DataFrame] = None
    enrichment: Optional[pd.DataFrame] = None
    motifs: Optional[pd.DataFrame] = None

    # ExternalValidation (modality-agnostic; each validator records status ran|skipped and its reason)
    validation: Optional[Dict[str, dict]] = None           # validator name -> {status, reason, headline numbers}
    validation_tables: Optional[Dict[str, pd.DataFrame]] = None   # validator name -> its table (also written as CSV)

    # Deployment
    vmaps: Optional[Dict[str, pd.DataFrame]] = None
    sanity: Optional[pd.DataFrame] = None
    codex: Optional[dict] = None
    exports: Optional[List[str]] = None

    # Reporting
    model_card: Optional[dict] = None
    contributions_path: Optional[str] = None
    gates: Optional[Dict[str, dict]] = None                # release gates: name -> {passed, value, threshold, source artefact}

    # bookkeeping
    plan: Any = None                                        # the RunPlan behind --only / --force (set by main.py)
    component_manifests: Dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def new(cls, cfg: Config) -> "Context":
        cfg.validate()
        os.makedirs(cfg.run_dir, exist_ok=True)
        os.makedirs(cfg.crop_dir, exist_ok=True)
        os.makedirs(cfg.local_data, exist_ok=True)
        attach_log_file(os.path.join(cfg.run_dir, "run.log"))
        device = resolve_device(cfg)
        seed_everything(cfg.seed)
        say(f"context: run_dir={cfg.run_dir} device={device} seed={cfg.seed}")
        return cls(cfg=cfg, run_dir=cfg.run_dir, device=device, seed=cfg.seed)

    def set(self, name: str, value: Any) -> None:
        """Assign a field once; a second assignment of a non-None field is logged as a replacement."""
        if getattr(self, name) is not None and name not in ("n_samples",):
            say(f"context: field '{name}' replaced")
        setattr(self, name, value)

    def replace(self, **kw: Any) -> "Context":
        for k, v in kw.items():
            self.set(k, v)
        return self

    def has(self, *names: str) -> bool:
        return all(getattr(self, n) is not None for n in names)

    def component_dir(self, name: str) -> str:
        d = os.path.join(self.run_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def um_per_px_by_sample(self) -> Dict[str, float]:
        """Microns per working-level pixel for every slide, from the OME metadata recorded at ingest.

        Falls back to the configured default for slides without a recorded value, so the same
        physical radius is used wherever a neighbourhood is built (summaries at ingest, learned
        neighbourhood targets, neighbour distances, attribution offsets).
        """
        default = float(self.cfg.um_per_px)
        out: Dict[str, float] = {}
        for row in self.manifest or []:
            l0 = row.get("um_per_px_l0")
            out[str(row["sample"])] = float(l0) * (2 ** self.cfg.img_level) if l0 else default
        return out

    def um_per_px_per_cell(self) -> np.ndarray:
        """(N,) microns per pixel of each cell's slide, aligned with ``obs`` rows."""
        by_sample = self.um_per_px_by_sample()
        default = float(self.cfg.um_per_px)
        samples = self.obs["sample"].astype(str).to_numpy()
        return np.array([by_sample.get(s, default) for s in samples], dtype=np.float32)

    def describe(self) -> None:
        say("context fields set: " + ", ".join(f.name for f in fields(self)
                                               if getattr(self, f.name) is not None and f.name not in ("cfg", "run_dir", "device", "seed")))
