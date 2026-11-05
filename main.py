#!/usr/bin/env python
"""mmprog: inferring investigator-defined gene programs from bone marrow morphology.

Input:   Xenium 5K trephines (DAPI morphology image, cell-by-gene counts, cell coordinates).
         The public cohort has no H&E; an H&E path exists as ``stain_mode: hematoxylin`` on a
         slide the manifest marks and is not exercised here. Optionally, an independent cohort
         manifest or a protein-marker table for the data-gated validators.
Output:  per-cell program scores predicted from morphology alone with calibrated uncertainty,
         the comparison of the unified model against its baselines and ablations, the gene
         backbone of each program (genes whose weight survives
         stability selection, graded by literature evidence), and the release gates that say
         which of these findings the run supports, written to <run_dir>/ (RESULTS.json).

``main`` is the whole pipeline: it calls the twelve components in order, one line each.
Each component prints what it does, writes its artefacts under <run_dir>/<component>/ and
reloads them on re-run when its configuration and its inputs are unchanged.

    python main.py --config configs/synthetic.yaml
    python main.py --config configs/default.yaml --set training.max_epochs=40
    python main.py --config configs/default.yaml --only evaluation,reporting
    python main.py --config configs/default.yaml --force training
    python main.py --config configs/default.yaml --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Each name is the function at the bottom of that component module, so a jump from
# main lands on the component itself rather than the re-export in components/__init__.py.
from mmprog.components.deployment import deploy  # noqa: E402
from mmprog.components.evaluation import evaluate  # noqa: E402
from mmprog.components.external_validation import validate_externally  # noqa: E402
from mmprog.components.feature_extraction import extract_features  # noqa: E402
from mmprog.components.feature_selection import select_features  # noqa: E402
from mmprog.components.ingest import ingest  # noqa: E402
from mmprog.components.interpretation import interpret_programs  # noqa: E402
from mmprog.components.preprocess import preprocess  # noqa: E402
from mmprog.components.program_inference import infer_gene_programs  # noqa: E402
from mmprog.components.reporting import report  # noqa: E402
from mmprog.components.training import train_unified_model  # noqa: E402
from mmprog.components.unified_model import build_unified_model  # noqa: E402
from mmprog.config import load_config  # noqa: E402
from mmprog.core.context import Context  # noqa: E402
from mmprog.core.pipeline import COMPONENT_ORDER, RunPlan, describe_dry_run, start_run, write_run_manifest  # noqa: E402
from mmprog.logging_utils import banner, say  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mmprog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", default="configs/default.yaml", help="YAML configuration, grouped by component section")
    p.add_argument("--set", "-s", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a configuration value; repeatable")
    p.add_argument("--only", default=None, help="comma-separated component names to run; earlier ones are loaded from artefacts")
    p.add_argument("--force", default=None, help="comma-separated component names to recompute together with everything after them")
    p.add_argument("--dry-run", action="store_true", help="print the resolved configuration and what each component would do, then exit")
    p.add_argument("--list-components", action="store_true")
    return p.parse_args(argv)


def print_run_summary(ctx: Context) -> None:
    """The last thing a run prints: the release gates and where the key artefacts are."""
    from mmprog.reporting.gates import evaluate_gates, print_gates

    banner("run summary")
    if ctx.gates is None:
        say("reporting did not run under this plan; no release gates to show")
    else:
        from mmprog.components.reporting import main_target_of
        targets = list(ctx.targets) if ctx.targets else (list(ctx.results.index) if ctx.results is not None else [])
        main_target = main_target_of(targets) if targets else "unknown"
        print_gates(evaluate_gates(ctx.run_dir, main_target), demo_mode=bool(ctx.cfg.demo_mode))
    say("key artefacts:")
    for label, rel in [("release gates and verdict", "RESULTS.json"),
                       ("held-out performance of every variant", "evaluation/cv_results.csv"),
                       ("decision rules in plain language", "evaluation/decisions.txt"),
                       ("stage association (patient level)", "evaluation/stage_association.csv"),
                       ("gene backbone of each program", "interpretation/gene_backbone.csv"),
                       ("pathway enrichment of the backbones", "interpretation/enrichment.csv"),
                       ("external validation summary", "external_validation/validation_summary.json"),
                       ("morphology-only program maps", "deployment/"),
                       ("human-readable report", "reporting/CONTRIBUTIONS.md"),
                       ("reproducibility record", "environment.json")]:
        path = os.path.join(ctx.run_dir, rel)
        say(f"  {label:<40} {path}{'' if os.path.exists(path) else '   (not written under this plan)'}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_components:
        print("\n".join(COMPONENT_ORDER))
        return 0

    cfg = load_config(args.config, args.set)
    plan = RunPlan.from_args(args.only, args.force)
    if args.dry_run:
        print(cfg.describe())
        print(plan.describe())
        print(describe_dry_run(plan))
        return 0

    t0 = time.time()
    ctx = start_run(cfg, plan, argv=sys.argv if argv is None else ["main.py", *argv])

    # 1. data in: GEO manifest, Xenium matrices and coordinates, DAPI / H&E images, count summaries on every cell
    ctx = ingest(ctx)
    # 2. preprocessing: cell QC, log normalisation, missing-gene mask across slides, nucleus detection
    ctx = preprocess(ctx)
    # 3. gene programs: anchored factorisation of expression, cell- and neighbourhood-scale targets, reliability ceilings, lineage labels
    ctx = infer_gene_programs(ctx)
    # 4. morphology features: frozen encoder embeddings, patch-grid tokens, morphometrics, neighbour graph
    ctx = extract_features(ctx)
    # 5. feature selection: dead and redundant features dropped, importance reported
    ctx = select_features(ctx)
    # 6. the unified model: token network (centre, grid blocks, neighbours, program tokens) and its heads
    ctx = build_unified_model(ctx)
    # 7. training: patient-level cross-validation of every variant, temperature scaling, the final fit
    ctx = train_unified_model(ctx)
    # 8. evaluation: per-patient metrics against ceilings, baselines, calibration, decision rules, stage association
    ctx = evaluate(ctx)
    # 9. interpretation: spatially regularized boosting, attribution, the gene backbone of each program
    ctx = interpret_programs(ctx)
    # 10. external validation: program specificity, spatial fidelity, perturbation robustness, cohort and protein bridges when data are given
    ctx = validate_externally(ctx)
    # 11. deployment: morphology-only inference, parity check, batch slides, exports
    ctx = deploy(ctx)
    # 12. reporting: model card, figures, QA tiers, release gates, CONTRIBUTIONS.md with every number read from an artefact
    ctx = report(ctx)

    write_run_manifest(ctx, t0)
    print_run_summary(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
