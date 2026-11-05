"""Shared test fixture: a tiny synthetic run context every component test can start from."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mmprog.config import load_config  # noqa: E402
from mmprog.core.context import Context  # noqa: E402


def small_cfg(tmp_path, n_patients: int = 3, cells_per: int = 300, **overrides):
    """Demo-mode config small enough for CPU unit tests (seconds, not minutes)."""
    root = str(tmp_path)
    sets = [
        "demo_mode=true", "run_name=unit", f"root={root}", f"local_data={root}/raw",
        "encoder=tiny_test", "pca_dim=8", "grid=4", "srb_grid=4", "patch_px=32", "n_folds=2", "seeds=[7]",
        "max_epochs=2", "srb_rounds=10", "model_variants=[full,shuffled]", "cv_variants=[full,shuffled]",
        "max_cells_per_sample=200", "n_boot=10", "programs_max_iter=20", "programs_fit_cells=2000",
        "attribution_n_cells=200", "manova_n_perm=19", "num_workers=0", "persistent_workers=false",
    ] + [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(overrides=sets)
    # fixture size for the synthetic cohort (read by components.ingest through getattr)
    cfg.demo_n_patients = n_patients
    cfg.demo_cells_per = cells_per
    cfg.demo_img_size = 256
    return cfg


def small_ctx(tmp_path, n_patients: int = 3, cells_per: int = 300, **overrides) -> Context:
    return Context.new(small_cfg(tmp_path, n_patients, cells_per, **overrides))
