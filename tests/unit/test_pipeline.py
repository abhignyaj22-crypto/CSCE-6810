"""The run plan, the component functions main.py calls, --dry-run and --list-components."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import small_ctx  # noqa: E402
from mmprog.core.component import BaseComponent  # noqa: E402
from mmprog.core.pipeline import COMPONENT_ORDER, RunPlan, describe_dry_run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = os.path.join(ROOT, "main.py")

TWELVE = ["ingest", "preprocess", "program_inference", "feature_extraction", "feature_selection", "unified_model",
          "training", "evaluation", "interpretation", "external_validation", "deployment", "reporting"]


def test_component_order_has_twelve_names():
    assert COMPONENT_ORDER == TWELVE


def test_run_plan_modes():
    plan = RunPlan.from_args("evaluation,reporting", None)
    assert plan.mode("ingest") == "load" and plan.mode("evaluation") == "run"
    assert plan.mode("external_validation") == "load" and plan.mode("reporting") == "run"
    plan = RunPlan.from_args("evaluation", None)
    assert plan.mode("deployment") == "skip" and plan.mode("external_validation") == "skip"
    plan = RunPlan.from_args(None, "training")
    assert plan.mode("ingest") == "run"
    assert not plan.forced("ingest") and not plan.forced("unified_model")
    assert plan.forced("training") and plan.forced("evaluation") and plan.forced("reporting")
    text = describe_dry_run(RunPlan.from_args(None, "interpretation"))
    assert "interpretation       run (forced recompute)" in text and "training             run\n" in text
    try:
        RunPlan.from_args("nope", None)
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown component name accepted")


class _Probe(BaseComponent):
    name = "ingest"
    seen = None

    def execute(self, ctx, mode="run", force=False):
        _Probe.seen = (mode, force)
        return ctx


def test_run_in_defaults_without_a_plan(tmp_path):
    ctx = small_ctx(tmp_path)
    ctx.plan = None
    _Probe().run_in(ctx)
    assert _Probe.seen == ("run", False)
    ctx.plan = RunPlan.from_args("preprocess", "ingest")
    _Probe().run_in(ctx)
    assert _Probe.seen == ("load", True)        # the plan says load, the flag says forced; execute() lets force win
    ctx.plan = RunPlan.from_args(None, "ingest")
    _Probe().run_in(ctx)
    assert _Probe.seen == ("run", True)


def test_component_functions_are_defined_in_their_modules():
    from mmprog.components import feature_extraction, ingest, interpretation, preprocess, program_inference, reporting, training
    for module, fn in [(ingest, "ingest"), (preprocess, "preprocess"), (program_inference, "infer_gene_programs"),
                       (feature_extraction, "extract_features"), (training, "train_unified_model"),
                       (interpretation, "interpret_programs"), (reporting, "report")]:
        assert callable(getattr(module, fn)), fn
        assert getattr(module, fn).__module__ == module.__name__


def _main(*args):
    return subprocess.run([sys.executable, MAIN, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_list_components_prints_twelve():
    proc = _main("--list-components")
    assert proc.returncode == 0
    assert proc.stdout.split() == TWELVE


def test_dry_run_exits_without_artefacts(tmp_path):
    ws = str(tmp_path / "ws")
    proc = _main("--config", "configs/synthetic.yaml", "--set", f"run.root={ws}", "--set", f"run.local_data={ws}/raw",
                 "--force", "training", "--dry-run")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "dry run" in proc.stdout and "training             run (forced recompute)" in proc.stdout
    assert "Resolved configuration" in proc.stdout
    run_dir = os.path.join(ws, "runs", "synthetic_v3")
    assert not os.path.exists(run_dir) or set(os.listdir(run_dir)) <= {"environment.json", "run.log"}


def test_force_wins_over_load_in_execute(tmp_path, monkeypatch):
    """--force X --only Y (Y after X): X must be recomputed, not quietly reloaded because it was not in --only."""
    calls = []

    class _Comp(BaseComponent):
        name = "ingest"
        requires = ()
        provides = ()
        artefacts = ()

        def is_done(self, ctx):
            return True                      # valid artefacts exist

        def load(self, ctx):
            calls.append("load")

        def run(self, ctx, force=False):
            calls.append(("run", force))
            return ctx

    ctx = small_ctx(tmp_path)
    comp = _Comp()
    os.makedirs(os.path.dirname(comp.manifest_path(ctx)), exist_ok=True)
    with open(comp.manifest_path(ctx), "w") as fh:
        json.dump({"hash": "x"}, fh)
    comp.execute(ctx, mode="load", force=False)
    assert calls == ["load"]
    calls.clear()
    comp.execute(ctx, mode="load", force=True)
    assert calls == [("run", True)]
