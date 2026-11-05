"""The component order, the run plan behind ``--only`` / ``--force``, and the run manifest.

``main.py`` calls the components directly, one after the other. This module only holds
what those calls share: the canonical order (so ``--only`` and ``--force`` can be
validated and resolved), ``RunPlan`` (which decides for every component whether it is
run, loaded from artefacts or skipped) and ``write_run_manifest``. ``run_pipeline`` is
kept for the tests and for callers that want the whole sequence as one function; it is
the same sequence ``main.py`` writes out.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from ..config import Config
from ..logging_utils import banner, say
from .component import _git_commit
from .context import Context

COMPONENT_ORDER = ["ingest", "preprocess", "program_inference", "feature_extraction", "feature_selection",
                   "unified_model", "training", "evaluation", "interpretation", "external_validation", "deployment", "reporting"]


@dataclass
class RunPlan:
    """For each component: ``mode`` in {run, load, skip} and whether it is forced.

    ``only``: run exactly these components; components before the last requested one are
    loaded from their artefacts (computed if the artefacts are missing or stale); components
    after it are skipped.
    ``force``: recompute these components and everything after them, even when artefacts
    are valid; a forced component that is not in ``only`` is still recomputed (force wins
    over load), and the components after it that ``only`` requests are recomputed too.
    """
    only: set = field(default_factory=set)
    force: set = field(default_factory=set)
    _forcing: bool = False

    @classmethod
    def from_args(cls, only: Optional[str], force: Optional[str]) -> "RunPlan":
        parse = lambda s: {x.strip() for x in s.split(",") if x.strip()} if s else set()  # noqa: E731
        plan = cls(only=parse(only), force=parse(force))
        unknown = (plan.only | plan.force) - set(COMPONENT_ORDER)
        if unknown:
            raise SystemExit(f"unknown component(s) {sorted(unknown)}; choose from {COMPONENT_ORDER}")
        return plan

    def mode(self, name: str) -> str:
        if not self.only or name in self.only:
            return "run"
        last_requested = max(COMPONENT_ORDER.index(n) for n in self.only)
        return "skip" if COMPONENT_ORDER.index(name) > last_requested else "load"

    def forced(self, name: str) -> bool:
        if name in self.force:
            self._forcing = True
        return self._forcing

    def describe(self) -> str:
        if not self.only and not self.force:
            return "run plan: every component in order (valid artefacts are reloaded, not recomputed)"
        parts = []
        if self.only:
            parts.append(f"only {sorted(self.only, key=COMPONENT_ORDER.index)}")
        if self.force:
            parts.append(f"force {sorted(self.force, key=COMPONENT_ORDER.index)} and everything after")
        return "run plan: " + "; ".join(parts)


def environment_record(cfg: Config, argv: Optional[Iterable[str]] = None) -> Dict:
    """What is needed to reproduce the run: interpreter, libraries, accelerator, commit, config, command line."""
    import platform

    def _version(module: str) -> Optional[str]:
        try:
            return __import__(module).__version__
        except Exception:
            return None

    gpu = None
    cuda = None
    torch_version = _version("torch")
    if torch_version:
        try:
            import torch
            cuda = torch.version.cuda
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
    try:
        commit = _git_commit(short=False)
        commit = None if commit == "unknown" else commit
    except Exception:
        commit = None
    return dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                torch=torch_version, cuda=cuda, gpu=gpu, numpy=_version("numpy"), pandas=_version("pandas"),
                sklearn=_version("sklearn"), git_commit=commit, command_line=list(argv) if argv is not None else sys.argv,
                started=time.strftime("%Y-%m-%d %H:%M:%S"), config=cfg.to_dict())


def describe_dry_run(plan: RunPlan) -> str:
    """One line per component saying whether the plan would run, load or skip it."""
    lines = ["dry run: what each component would do under this plan"]
    first_forced = min((COMPONENT_ORDER.index(f) for f in plan.force), default=None)
    for i, name in enumerate(COMPONENT_ORDER):
        mode = plan.mode(name)
        forced = first_forced is not None and i >= first_forced and mode == "run"
        lines.append(f"  {name:<20} {mode}{' (forced recompute)' if forced else ''}")
    return "\n".join(lines)


def start_run(cfg: Config, plan: RunPlan, argv: Optional[Iterable[str]] = None) -> Context:
    """Create the run context, store the plan on it and write environment.json next to run.log."""
    banner(f"mmprog v3: run '{cfg.run_name}'")
    say(cfg.describe())
    say(plan.describe())
    ctx = Context.new(cfg)
    ctx.plan = plan
    record = environment_record(cfg, argv)
    with open(os.path.join(ctx.run_dir, "environment.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    say(f"environment: python {record['python']}, torch {record['torch']}, cuda {record['cuda']}, gpu {record['gpu']}, "
        f"commit {record['git_commit']}; written to environment.json")
    return ctx


def write_run_manifest(ctx: Context, t0: float) -> Dict:
    summary = dict(run_name=ctx.cfg.run_name, minutes=round((time.time() - t0) / 60, 2),
                   components={k: dict(hash=v["hash"], seconds=v["seconds"], seconds_compute=v.get("seconds_compute"),
                                       recomputed=v["recomputed"]) for k, v in ctx.component_manifests.items()})
    with open(os.path.join(ctx.run_dir, "run_manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    say(f"pipeline finished in {summary['minutes']} min; run manifest written")
    return summary


def run_pipeline(cfg: Config, only: Optional[Iterable[str]] = None, force: Optional[Iterable[str]] = None) -> Context:
    """The same sequence as ``main.py``, as one function, for tests and notebooks."""
    from ..components import (build_unified_model, deploy, evaluate, extract_features, infer_gene_programs, ingest,
                              interpret_programs, preprocess, report, select_features, train_unified_model,
                              validate_externally)
    plan = RunPlan(only=set(only or []), force=set(force or []))
    t0 = time.time()
    ctx = start_run(cfg, plan)
    for step in (ingest, preprocess, infer_gene_programs, extract_features, select_features, build_unified_model,
                 train_unified_model, evaluate, interpret_programs, validate_externally, deploy, report):
        ctx = step(ctx)
    write_run_manifest(ctx, t0)
    return ctx
