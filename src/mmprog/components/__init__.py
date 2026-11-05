"""The twelve pipeline components as plain functions, in the order ``main.py`` calls them.

    ingest                GEO manifest, downloads, Xenium loading, count summaries on all cells, image crops
    preprocess            cell QC, normalisation, missing-gene mask, nucleus detection
    infer_gene_programs   anchored factorisation into gene programs, targets, lineage labels, reliability
    extract_features      encoder embeddings, patch-grid tokens, morphometrics, neighbour graph
    select_features       dead and redundant feature removal, importance report
    build_unified_model   the multimodal token network specification
    train_unified_model   patient-level cross-validation of every variant and the final fit
    evaluate              metrics, baselines, calibration, decision rules, stage association
    interpret_programs    spatially regularized boosting, pathway attribution, gene backbone
    validate_externally   modality-agnostic validation: program specificity, spatial fidelity, robustness, bridges
    deploy                morphology-only inference, parity check, batch slides, exports
    report                model card, figures, QA tiers, release gates, CONTRIBUTIONS.md

Each function takes the run context and returns it with the component's fields filled in.
The function is defined at the bottom of its own component module, so "go to definition"
lands on the component; this package only re-exports them. The class behind each function
(``Ingest``, ``Preprocess``, ...) carries the skip rule, the manifest and the artefact
contract, and ``BaseComponent.run_in`` applies the run plan stored on ``ctx.plan``.
Imports are local so a missing optional dependency (torch, statsmodels) only fails the
component that needs it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.context import Context


def ingest(ctx: "Context") -> "Context":
    from .ingest import ingest as _fn
    return _fn(ctx)


def preprocess(ctx: "Context") -> "Context":
    from .preprocess import preprocess as _fn
    return _fn(ctx)


def infer_gene_programs(ctx: "Context") -> "Context":
    from .program_inference import infer_gene_programs as _fn
    return _fn(ctx)


def extract_features(ctx: "Context") -> "Context":
    from .feature_extraction import extract_features as _fn
    return _fn(ctx)


def select_features(ctx: "Context") -> "Context":
    from .feature_selection import select_features as _fn
    return _fn(ctx)


def build_unified_model(ctx: "Context") -> "Context":
    from .unified_model import build_unified_model as _fn
    return _fn(ctx)


def train_unified_model(ctx: "Context") -> "Context":
    from .training import train_unified_model as _fn
    return _fn(ctx)


def evaluate(ctx: "Context") -> "Context":
    from .evaluation import evaluate as _fn
    return _fn(ctx)


def interpret_programs(ctx: "Context") -> "Context":
    from .interpretation import interpret_programs as _fn
    return _fn(ctx)


def validate_externally(ctx: "Context") -> "Context":
    from .external_validation import validate_externally as _fn
    return _fn(ctx)


def deploy(ctx: "Context") -> "Context":
    from .deployment import deploy as _fn
    return _fn(ctx)


def report(ctx: "Context") -> "Context":
    from .reporting import report as _fn
    return _fn(ctx)


__all__ = ["ingest", "preprocess", "infer_gene_programs", "extract_features", "select_features", "build_unified_model",
           "train_unified_model", "evaluate", "interpret_programs", "validate_externally", "deploy", "report"]
