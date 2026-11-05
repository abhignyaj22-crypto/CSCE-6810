"""Evaluation: patient-level cross-validation, baselines, ablations, decision rules, clinical association."""
from .cross_validation import CVData, build_cv_data, run_cv, composition_only, run_all_variants, run_multiseed
from .metrics import score, bootstrap_ci
from .decisions import apply_decision_rules
from .clinical import program_burden, stage_trend, cox_pfs, signature_concordance
from .transfer import run_transfer

__all__ = [
    "CVData", "build_cv_data", "run_cv", "composition_only", "run_all_variants", "run_multiseed",
    "score", "bootstrap_ci", "apply_decision_rules",
    "program_burden", "stage_trend", "cox_pfs", "signature_concordance", "run_transfer",
]
