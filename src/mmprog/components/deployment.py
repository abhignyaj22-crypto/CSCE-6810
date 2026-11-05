"""Deployment component: morphology-only inference, exports and batch slides.

1. Check on a training slide: run the morphology-only path on the first training slide's
   image, match detected nuclei to annotated cells and report the Pearson and the
   calibration offset (``sanity_check.csv``).
2. Batch inference over every slide in ``cfg.slide_dir`` when set.
3. Exports of every map for QuPath and Xenium Explorer (D22), plus a README that says what
   the columns mean.

The protein-imaging (CODEX) bridge that used to live here is now the ``protein_bridge``
validator of the ExternalValidation component, next to the other orthogonal checks.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, section, show_table, timed, warn

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "anchor_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z"]

README_TEXT = """Exports of the morphology-only program maps.

<slide>_qupath.tsv          tab separated; x_um, y_um and one 'Measurement: <target>_pred' column per target;
                            import in QuPath as detections (File > Import objects).
<slide>_xenium_explorer.csv cell_id, x_centroid, y_centroid (microns) and one column per target;
                            load in Xenium Explorer as a cell group or feature table.
<target>_pred               predicted standardised program score (cohort z units of the training run)
<target>_std                predictive standard deviation (temperature-scaled heteroscedastic head), when present
lineage_prob_<name>         composition-head probability of each lineage, when present

Scores are gene-set abundances associated with drug resistance in the literature, not pathway
activity and not a drug-response measurement (docs/research/PATHWAY_INTERPRETATION.md).
"""


def _main_target(targets: List[str]) -> str:
    for t in MAIN_TARGET_PREFERENCE:
        if t in targets:
            return t
    return targets[0]


class Deployment(BaseComponent):
    name = "deployment"
    # what _run touches unconditionally; the parity check also reads EMB, GRID, MOR, NB, NB_DIST, SID, lineage and
    # the manifest, and says "not computed" when a hand-built context lacks them instead of failing late
    requires = ("final_model_path", "images", "obs", "targets", "lin_names")
    parity_fields = ("Y", "EMB", "GRID", "MOR", "NB", "NB_DIST", "SID", "lineage")
    provides = ("vmaps", "sanity", "exports")
    artefacts = ("vmap_check.csv", "sanity_check.csv", "exports/README.txt")
    config_fields = ("slide_dir", "encoder", "patch_px", "knn_k", "img_level", "z_project", "stain_mode", "nb_radius_um",
                     "um_per_px_level0")
    depends_on = ("training",)

    # hooks a test can override (same names as virtual_program_map_v3)
    vmap_hooks: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _um_per_px(self, ctx: Context, slide: str, image_path: Optional[str] = None) -> float:
        """The slide's pixel size at the working level: the manifest value for a training slide, the OME
        metadata for a new slide, the configured default (printed) when neither is available."""
        by_sample = ctx.um_per_px_by_sample() if ctx.manifest else {}
        if slide in by_sample:
            return float(by_sample[slide])
        if image_path:
            from ..data.images import um_per_px_level0
            l0 = um_per_px_level0(image_path, default=float(ctx.cfg.um_per_px_level0))
            return float(l0) * (2 ** ctx.cfg.img_level)
        warn(f"slide {slide}: no pixel size on the manifest or in the image; using the configured {ctx.cfg.um_per_px:.4f} um/px")
        return float(ctx.cfg.um_per_px)

    def _vmap(self, ctx: Context, img: np.ndarray, ck: Dict[str, Any], sample: str, um_per_px: Optional[float] = None) -> pd.DataFrame:
        from ..inference.virtual_map_v3 import virtual_program_map_v3
        return virtual_program_map_v3(img, ck, ctx.cfg, ctx.device, sample_name=sample, um_per_px=um_per_px, **self.vmap_hooks)

    def _export(self, ctx: Context, vmap: pd.DataFrame, slide: str, um_per_px: Optional[float] = None) -> List[str]:
        from ..inference.export import export_qupath, export_xenium_explorer
        out_dir = os.path.join(self.dir(ctx), "exports")
        os.makedirs(out_dir, exist_ok=True)
        q = os.path.join(out_dir, f"{slide}_qupath.tsv")
        x = os.path.join(out_dir, f"{slide}_xenium_explorer.csv")
        scale = float(um_per_px or ctx.cfg.um_per_px)
        export_qupath(vmap, q, scale)
        export_xenium_explorer(vmap, x, scale)
        say(f"exports for {slide}: {os.path.basename(q)}, {os.path.basename(x)}")
        return [q, x]

    def _parity(self, ctx: Context, ck: Dict[str, Any], s0: str, target: str, um_per_px: float) -> Tuple[float, str]:
        """Inference parity: the morphology-only path fed the slide's own annotated centroids versus the
        training path (precomputed features, saved feature space, selector and temperature) on the same cells.

        The two paths differ only in the unseen-slide embedding slot and in the neighbour graph being rebuilt
        from the same centroids, so a Pearson well below 1 means the inference code drifted from training.
        Returns (pearson, error): a crash in the inference path is returned as the error text, never hidden
        as a missing number, because that crash is exactly what the ``inference_parity`` release gate must catch.
        """
        from ..inference.virtual_map_v3 import virtual_program_map_v3
        from ..training.cv import head_layout, load_final
        from ..training.data import CellBatcher, variant_flags
        from ..training.loop import predict
        rows = np.where(ctx.obs["sample"].astype(str).to_numpy() == s0)[0]
        ox, oy = ctx.images.offset(s0)
        xy_crop = ctx.obs[["x", "y"]].to_numpy(np.float32)[rows] - np.array([ox, oy], dtype=np.float32)
        try:
            with timed(f"parity: morphology-only path on the {len(rows)} annotated centroids of {s0}"):
                vmap_c = virtual_program_map_v3(ctx.images.get(s0), ck, ctx.cfg, ctx.device, sample_name=s0,
                                                detect_fn=lambda img: xy_crop.copy(), um_per_px=um_per_px, **self.vmap_hooks)
            model, raw = load_final(ctx.final_model_path, ctx.device)
            keep, head_targets = head_layout(ctx, "full")
            if target not in head_targets:
                say(f"parity: {target} is not a head of the final model; not computed")
                return float("nan"), ""
            missing = [f for f in self.parity_fields if getattr(ctx, f) is None]
            if missing:
                say(f"parity: context lacks {missing} (hand-built context); not computed")
                return float("nan"), ""
            fs = raw.get("feature_space") or raw.get("fs")
            if fs is None:
                say("parity: the checkpoint carries no feature space (fake or v2 checkpoint); not computed")
                return float("nan"), ""
            X = fs.transform(ctx.EMB, ctx.MOR)
            Y = np.asarray(ctx.Y, dtype=np.float32)[:, keep]
            xy_um = ctx.obs[["x", "y"]].to_numpy(np.float32) * ctx.um_per_px_per_cell()[:, None]
            batcher = CellBatcher(cell_idx=rows, X_centre=X, GRID=ctx.GRID, NB=ctx.NB, NB_DIST=ctx.NB_DIST, XY_um=xy_um,
                                  SID=ctx.SID, Y=Y, LIN=ctx.lineage, expression_embedding=ctx.expression_embedding,
                                  batch_cells=min(1024, len(rows)), name="parity")
            mean, _, _, _ = predict(model, batcher, ctx.cfg, ctx.device, variant_flags("full"))
            a = vmap_c[target.replace("_z", "_pred")].to_numpy(dtype=np.float64)
            b = mean[:, head_targets.index(target)].astype(np.float64)
        except Exception as e:                                   # recorded, and the release gate fails on it
            msg = f"{type(e).__name__}: {e}"
            warn(f"parity: the inference path raised ({msg}); recorded as parity_error, the inference_parity gate fails")
            return float("nan"), msg
        if len(a) != len(b) or len(a) < 3:
            msg = f"{len(a)} inference rows vs {len(b)} training rows"
            warn(f"parity: {msg}; recorded as parity_error")
            return float("nan"), msg
        r = float(np.corrcoef(a, b)[0, 1])
        say(f"parity: pearson(morphology-only path, training path) on {len(a)} cells = {r:.5f} "
            f"(release gate inference_parity asks for > 0.95)")
        return r, ""

    # ------------------------------------------------------------------ steps
    def _check_slide(self, ctx: Context, ck: Dict[str, Any]) -> None:
        from ..inference.virtual_map_v3 import sanity_check

        section("deployment 1/3: morphology-only inference check on a training slide")
        s0 = str(ctx.obs["sample"].iloc[0])
        say(f"slide {s0} (in deployment this would be an unseen slide; here it measures agreement with the annotation)")
        img = ctx.images.get(s0)
        um_px = self._um_per_px(ctx, s0)
        say(f"slide {s0}: {um_px:.4f} um/px at the working level (the same value training used for its neighbour offsets)")
        with timed(f"virtual program map on {s0}"):
            vmap = self._vmap(ctx, img, ck, s0, um_per_px=um_px)
        vmap.to_csv(self.path(ctx, "vmap_check.csv"), index=False)
        target = _main_target(ctx.targets)
        obs_slide = ctx.obs[ctx.obs["sample"] == s0]
        if target not in obs_slide and ctx.Y is not None:
            obs_slide = obs_slide.copy()
            obs_slide[target] = ctx.Y[obs_slide.index, ctx.targets.index(target)]
        sanity = sanity_check(vmap, obs_slide, ctx.images.offset(s0), target, ctx.cfg, um_per_px=um_px)
        sanity.insert(0, "slide", s0)
        parity, parity_error = self._parity(ctx, ck, s0, target, um_px)
        sanity["parity_pearson"] = parity
        sanity["parity_error"] = parity_error
        sanity["um_per_px"] = um_px
        sanity.to_csv(self.path(ctx, "sanity_check.csv"), index=False)
        ctx.set("sanity", sanity)
        ctx.set("vmaps", {s0: vmap})
        exports = self._export(ctx, vmap, s0, um_per_px=um_px)
        ctx.set("exports", exports)

    def _batch_slides(self, ctx: Context, ck: Dict[str, Any]) -> None:
        section("deployment 2/3: batch inference over a slide folder")
        cfg = ctx.cfg
        if not cfg.slide_dir or not os.path.isdir(cfg.slide_dir):
            say("gate: set slide_dir to a folder of OME-TIFF / TIFF slides to run batch inference")
            return
        from ..data.images import load_image
        paths = sorted(glob.glob(os.path.join(cfg.slide_dir, "*.tif*")))
        say(f"{len(paths)} slides in {cfg.slide_dir}")
        target = _main_target(ctx.targets).replace("_z", "_pred")
        rows = []
        for p in paths:
            slide = os.path.splitext(os.path.basename(p))[0].replace(".ome", "")
            um_px = self._um_per_px(ctx, slide, image_path=p)
            with timed(f"slide {slide} ({um_px:.4f} um/px)"):
                img = load_image(p, cfg.img_level, None, cfg.z_project, cfg.stain_mode)
                vmap = self._vmap(ctx, img, ck, slide, um_per_px=um_px)
            vmap.to_csv(self.path(ctx, f"vmap_{slide}.csv"), index=False)
            ctx.vmaps[slide] = vmap
            ctx.exports.extend(self._export(ctx, vmap, slide, um_per_px=um_px))
            burden = float((vmap[target] > np.percentile(vmap[target], 75)).mean()) if len(vmap) else np.nan
            rows.append(dict(slide=slide, n_nuclei=len(vmap), burden_top_quartile=burden))
        if rows:
            batch = pd.DataFrame(rows)
            batch.to_csv(self.path(ctx, "batch_burden.csv"), index=False)
            show_table(batch, title="batch inference summary:")

    def _readme(self, ctx: Context) -> None:
        section("deployment 3/3: export README")
        out_dir = os.path.join(self.dir(ctx), "exports")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "README.txt"), "w") as fh:
            fh.write(README_TEXT)
        say(f"{len(ctx.exports)} export files under {out_dir}")

    # ------------------------------------------------------------------ protocol
    def _run(self, ctx: Context) -> None:
        from ..inference.virtual_map_v3 import load_checkpoint
        ck = load_checkpoint(ctx.final_model_path, ctx.device)
        self._check_slide(ctx, ck)
        self._batch_slides(ctx, ck)
        self._readme(ctx)

    def load(self, ctx: Context) -> None:
        vmap = pd.read_csv(self.path(ctx, "vmap_check.csv"))
        sanity = pd.read_csv(self.path(ctx, "sanity_check.csv"))
        slide = str(sanity["slide"].iloc[0]) if len(sanity) else "check"
        vmaps = {slide: vmap}
        for p in glob.glob(self.path(ctx, "vmap_*.csv")):
            name = os.path.basename(p)[5:-4]
            if name != "check":
                vmaps[name] = pd.read_csv(p)
        ctx.set("vmaps", vmaps)
        ctx.set("sanity", sanity)
        ctx.set("exports", sorted(glob.glob(os.path.join(self.dir(ctx), "exports", "*.*sv"))))
        say(f"deployment loaded: {len(vmaps)} virtual maps, sanity pearson {sanity['pearson'].iloc[0] if len(sanity) else 'n/a'}, "
            f"{len(ctx.exports)} exports")


def deploy(ctx: Context) -> Context:
    """Component 11: morphology-only inference check, batch slides, exports.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Deployment().run_in(ctx)
