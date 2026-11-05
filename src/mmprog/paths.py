"""Run directory and artefact registry.

Every stage writes its result to the run directory under a fixed name and skips
itself on re-run when that file exists. This is what makes a Colab run
resumable after a disconnect, and it is what lets the stage order be read from
the file listing alone.
"""
from __future__ import annotations

import os
from typing import Dict, List

from .config import Config
from .logging_utils import say


class RunDir:
    """Thin wrapper around the run directory with named artefacts."""

    ARTEFACTS: Dict[str, str] = {
        "config": "config.json",
        "manifest": "manifest.json",
        "adata_raw": "adata_raw.h5ad",
        "full_xy": "full_xy.npz",
        "demo_images": "images.npz",
        "adata_proc": "adata_proc.h5ad",
        "targets": "targets.json",
        "reliability": "target_reliability.csv",
        "feature_selection": "feature_selection.json",
        "cv_results": "cv_results.csv",
        "cv_results_adj": "cv_results_reliability_adjusted.csv",
        "cv_per_patient": "cv_per_patient.csv",
        "cv_fold_log": "cv_fold_log.csv",
        "cv_multiseed": "cv_multiseed.csv",
        "pred_full": "pred_full.npy",
        "transfer": "transfer_results.csv",
        "burden": "program_burden.csv",
        "cox_pfs": "cox_pfs.csv",
        "cox_signature": "cox_signature_bulk.csv",
        "final_model": "final_model.pt",
        "tier2": "tier2_review_cells.csv",
        "tier3": "tier3_reimage_requests.csv",
        "overlays": "overlays.png",
        "model_card": "model_card.json",
        "phase_status": "phase_status.csv",
        "contributions": "CONTRIBUTIONS.md",
        "batch_burden": "batch_burden.csv",
        "codex": "codex_validation.json",
        "manova": "manova_results.csv",
        "srb_results": "srb_results.csv",
        "srb_maps": "srb_importance_maps.npz",
        "srb_rings": "srb_ring_summary.csv",
        "motifs": "motif_enrichment.csv",
        "motif_vocab": "motif_vocabulary.json",
        "run_log": "run.log",
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.run_dir
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(cfg.crop_dir, exist_ok=True)
        os.makedirs(cfg.local_data, exist_ok=True)

    def art(self, name: str) -> str:
        """Absolute path of a named artefact (or any file name inside the run dir)."""
        return os.path.join(self.path, self.ARTEFACTS.get(name, name))

    def done(self, name: str) -> bool:
        return os.path.exists(self.art(name))

    def remove(self, *names: str) -> None:
        for n in names:
            p = self.art(n)
            if os.path.exists(p):
                os.remove(p)
                say(f"removed stale artefact {os.path.basename(p)}")

    def listing(self) -> List[str]:
        return sorted(os.listdir(self.path))

    def report(self) -> None:
        say(f"run directory: {self.path}")
        for f in self.listing():
            size = os.path.getsize(os.path.join(self.path, f))
            say(f"{f:<48} {size / 1e6:8.1f} MB", indent=1)
