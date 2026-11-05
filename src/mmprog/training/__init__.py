"""Batching, the training loop and patient-level cross-validation of the unified model."""
from .data import ALL_VARIANTS, CellBatcher, make_loader, variant_flags
from .loop import batch_loss, configure_backend, fit_temperature, predict, train_one
from .cv import fit_final, load_final, make_folds, run_all, run_variant

__all__ = ["ALL_VARIANTS", "CellBatcher", "make_loader", "variant_flags", "batch_loss", "configure_backend",
           "fit_temperature", "predict", "train_one", "fit_final", "load_final", "make_folds", "run_all", "run_variant"]
