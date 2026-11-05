"""Component base class: the one place where skipping, logging and manifests are implemented.

A component declares what it needs (``requires``), what it sets (``provides``), the files
it writes (``artefacts``) and which configuration fields influence it (``config_fields``).
``run`` checks the inputs, decides whether the stored artefacts are still valid, and either
reloads them or recomputes. Validity is a hash of the component's own configuration
fields plus the manifests of the components it depends on, so a change upstream
invalidates everything downstream without any marker files.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, Sequence, Tuple

from ..logging_utils import banner, say, timed
from .context import Context


REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _git_commit(short: bool = True) -> str:
    """The commit of this source tree (not of the process working directory, which may be anywhere)."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.run(args, capture_output=True, text=True, timeout=5, cwd=REPO_DIR).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class BaseComponent:
    name: str = "base"
    requires: Tuple[str, ...] = ()
    provides: Tuple[str, ...] = ()
    artefacts: Tuple[str, ...] = ()
    config_fields: Tuple[str, ...] = ()
    depends_on: Tuple[str, ...] = ()       # names of upstream components whose manifests enter the hash

    # ------------------------------------------------------------------ to implement
    def _run(self, ctx: Context) -> None:
        raise NotImplementedError

    def load(self, ctx: Context) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared behaviour
    def dir(self, ctx: Context) -> str:
        return ctx.component_dir(self.name)

    def path(self, ctx: Context, artefact: str) -> str:
        return os.path.join(self.dir(ctx), artefact)

    def config_section(self, ctx: Context) -> Dict[str, Any]:
        cfg = dataclasses.asdict(ctx.cfg)
        section = {k: cfg[k] for k in self.config_fields if k in cfg}
        section["seed"] = cfg["seed"]
        return section

    def expected_hash(self, ctx: Context) -> str:
        upstream = {d: ctx.component_manifests.get(d, {}).get("hash", "missing") for d in self.depends_on}
        return _hash({"config": self.config_section(ctx), "upstream": upstream})

    def manifest_path(self, ctx: Context) -> str:
        return self.path(ctx, "manifest.json")

    def is_done(self, ctx: Context) -> bool:
        if not all(os.path.exists(self.path(ctx, a)) for a in self.artefacts):
            return False
        if not os.path.exists(self.manifest_path(ctx)):
            return False
        with open(self.manifest_path(ctx)) as fh:
            stored = json.load(fh)
        return stored.get("hash") == self.expected_hash(ctx)

    def _write_manifest(self, ctx: Context, seconds: float, recomputed: bool) -> None:
        # on a load, keep the compute time of the run that produced the artefacts; record the load time separately
        previous = {}
        if not recomputed and os.path.exists(self.manifest_path(ctx)):
            with open(self.manifest_path(ctx)) as fh:
                previous = json.load(fh)
        seconds_compute = round(seconds, 1) if recomputed else previous.get("seconds_compute", previous.get("seconds"))
        manifest = dict(component=self.name, hash=self.expected_hash(ctx), config=self.config_section(ctx),
                        upstream={d: ctx.component_manifests.get(d, {}).get("hash") for d in self.depends_on},
                        artefacts=list(self.artefacts), git_commit=_git_commit(), device=ctx.device,
                        python=platform.python_version(), seconds=round(seconds, 1), seconds_compute=seconds_compute,
                        seconds_load=None if recomputed else round(seconds, 1), recomputed=recomputed,
                        computed_at=previous.get("computed_at") if not recomputed else time.strftime("%Y-%m-%d %H:%M:%S"),
                        finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.manifest_path(ctx), "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        ctx.component_manifests[self.name] = manifest

    def run(self, ctx: Context, force: bool = False) -> Context:
        banner(f"{self.name}")
        missing = [r for r in self.requires if getattr(ctx, r) is None]
        if missing:
            raise RuntimeError(f"component '{self.name}' requires context fields {missing}; run the upstream components first")
        t0 = time.time()
        if not force and self.is_done(ctx):
            say(f"{self.name}: artefacts are valid for the current configuration and upstream manifests; loading")
            with timed(f"{self.name}: load"):
                self.load(ctx)
            recomputed = False
        else:
            reason = "forced" if force else ("no artefacts yet" if not os.path.exists(self.manifest_path(ctx)) else "configuration or upstream changed")
            say(f"{self.name}: recomputing ({reason})")
            with timed(f"{self.name}: run"):
                self._run(ctx)
            recomputed = True
        not_set = [p for p in self.provides if getattr(ctx, p) is None]
        if not_set:
            raise RuntimeError(f"component '{self.name}' finished without setting {not_set}")
        absent = [a for a in self.artefacts if not os.path.exists(self.path(ctx, a))]
        if absent:
            raise RuntimeError(f"component '{self.name}' finished without writing {absent}")
        self._write_manifest(ctx, time.time() - t0, recomputed)
        say(f"{self.name}: provides {list(self.provides)}; artefacts in {self.dir(ctx)}")
        ctx.describe()
        return ctx

    def run_in(self, ctx: Context) -> Context:
        """Execute under the run plan stored on the context (``ctx.plan`` from ``--only`` / ``--force``).

        This is what the one-line component functions in ``mmprog.components`` call. A context
        without a plan (a notebook, a test that built the context by hand) runs the component
        normally: mode "run", not forced.
        """
        plan = getattr(ctx, "plan", None)
        if plan is None:
            return self.execute(ctx, mode="run", force=False)
        return self.execute(ctx, mode=plan.mode(self.name), force=plan.forced(self.name))

    def execute(self, ctx: Context, mode: str = "run", force: bool = False) -> Context:
        """What ``main.py`` calls for every component.

        ``mode`` is decided by the run plan from ``--only`` / ``--force``:
          run   compute, or reload valid artefacts (the normal path)
          load  not requested, but a later component needs its outputs: reload artefacts
                if they are valid, otherwise compute after all
          skip  after the last requested component: do nothing
        """
        if mode == "skip":
            say(f"{self.name}: after the last requested component; not run")
            return ctx
        if mode == "load" and force:
            # --force X with --only Y (Y after X): the forced component is recomputed, as the flag promises,
            # instead of being quietly reloaded because it was not in --only
            say(f"{self.name}: not requested but forced; recomputing")
            mode = "run"
        if mode == "load":
            missing = [r for r in self.requires if getattr(ctx, r) is None]
            if missing:
                raise RuntimeError(f"component '{self.name}' requires context fields {missing}; run the upstream components first")
            if self.is_done(ctx):
                banner(f"{self.name}")
                say(f"{self.name}: not requested; loading artefacts")
                with timed(f"{self.name}: load"):
                    self.load(ctx)
                with open(self.manifest_path(ctx)) as fh:
                    ctx.component_manifests[self.name] = json.load(fh)
                return ctx
            say(f"{self.name}: not requested and no valid artefacts; running it because a requested component needs it")
        return self.run(ctx, force=force)


def hash_arrays(*arrays: Sequence[Any]) -> str:
    """Short content hash of numpy arrays or lists, for cache keys inside components."""
    h = hashlib.md5()
    for a in arrays:
        try:
            import numpy as np
            h.update(np.ascontiguousarray(a).tobytes()[:10_000_000])
        except Exception:
            h.update(json.dumps(a, default=str).encode())
    return h.hexdigest()[:10]
