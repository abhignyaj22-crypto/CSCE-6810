"""Ingest: manifest, downloads, Xenium loading, count summaries on every cell, image crops.

Real path: the GEO listing is read at run time, each sample's matrix and cell table are
fetched first (so the panel-size rule can exclude a sample before its multi-gigabyte
image is downloaded), then the image, then the v2 per-sample body ``ingest_one_sample``
runs in the order the design requires: load, axis check, QC, count summaries on all
QC-passing cells, crop and cache the image, subsample for training, attach summaries.
Demo path: the synthetic fixture, which already carries the count summaries.

Count summaries (``cnt_*``, ``nb_*``, ``density_*``) are computed here, before any
subsampling, and Preprocess must not recompute them (D6).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, show_table, warn
from ..resources import load_lineages, load_programs


class Ingest(BaseComponent):
    name = "ingest"
    requires = ()
    provides = ("manifest", "adata_raw", "images", "full_xy")
    artefacts = ("samples.json", "adata_raw.h5ad", "full_xy.npz")

    def is_done(self, ctx: Context) -> bool:
        # demo mode also needs the synthetic images, which load() reads
        if ctx.cfg.demo_mode and not os.path.exists(self.path(ctx, "images.npz")):
            return False
        return super().is_done(ctx)
    config_fields = ("demo_mode", "geo_accession", "sample_limit", "exclude_title_regex", "min_panel_genes",
                     "platform", "img_level", "z_project", "stain_mode", "um_per_px_level0", "min_transcripts",
                     "min_genes", "max_cells_per_sample", "nb_radius_um", "density_radii_px")
    depends_on = ()

    # ------------------------------------------------------------------ helpers
    def _image_store(self, ctx: Context, manifest: List[dict], full_xy: Dict[str, np.ndarray]):
        """Image store keyed on every QC-passing cell so the crop covers any later subsample."""
        from ..data.images import ImageStore
        demo_images = None
        if ctx.cfg.demo_mode:
            demo_images = dict(np.load(self.path(ctx, "images.npz")))
        xy_by_sample = {s: np.asarray(v, dtype=np.float32) for s, v in full_xy.items()}
        return ImageStore(ctx.cfg, manifest, xy_by_sample, demo_images=demo_images)

    def _write(self, ctx: Context, adata, manifest: List[dict], full_xy: Dict[str, np.ndarray]) -> None:
        from ..data.manifest import save_manifest
        adata.write_h5ad(self.path(ctx, "adata_raw.h5ad"))
        save_manifest(manifest, self.path(ctx, "samples.json"))
        np.savez_compressed(self.path(ctx, "full_xy.npz"), **full_xy)
        say(f"wrote adata_raw.h5ad ({adata.n_obs} cells x {adata.n_vars} genes), samples.json "
            f"({len(manifest)} slides), full_xy.npz ({sum(len(v) for v in full_xy.values())} QC-passing cell coordinates)")

    # ------------------------------------------------------------------ paths
    def _run_demo(self, ctx: Context):
        from ..data.synthetic import make_demo
        programs = load_programs()
        lineages = load_lineages()
        # fixture size can be shrunk for tests through optional attributes on the config instance
        n_patients = int(getattr(ctx.cfg, "demo_n_patients", 6))
        cells_per = int(getattr(ctx.cfg, "demo_cells_per", 3000))
        img_size = int(getattr(ctx.cfg, "demo_img_size", 1024))
        say(f"demo mode: building the synthetic cohort ({n_patients} patients x {cells_per} cells, {img_size} px images; "
            f"planted program in plasma-cell blobs)")
        adata, images, manifest = make_demo(programs, lineages, ctx.cfg, n_patients=n_patients, cells_per=cells_per,
                                            img_size=img_size, seed=ctx.cfg.seed)
        np.savez_compressed(self.path(ctx, "images.npz"), **images)
        full_xy = {s: adata.obs.loc[adata.obs["sample"] == s, ["x", "y"]].values.astype(np.float32)
                   for s in adata.obs["sample"].unique()}
        return adata, manifest, full_xy

    def _run_real(self, ctx: Context):
        import anndata as ad
        from ..data.download import fetch_sample_files
        from ..data.images import ImageStore
        from ..data.manifest import build_manifest_from_geo, select_samples
        from ..data.tenx import h5_matrix_shape
        from ..pipeline import ingest_one_sample
        from ..preprocessing.spatial_summaries import attach_summaries, count_summaries, subsample_cells

        cfg = ctx.cfg
        programs = load_programs()
        mdf = select_samples(build_manifest_from_geo(cfg), cfg)
        say(f"{len(mdf)} samples selected from {cfg.geo_accession}")
        store = ImageStore(cfg, [], {})      # crops, caches and deletes each decompressed TIFF in turn
        manifest: List[dict] = []
        full_xy: Dict[str, np.ndarray] = {}
        parts = []
        for _, r in mdf.iterrows():
            row = r.to_dict()
            t0 = time.time()
            say(f"--- {row['sample']} ({row['stage']}) ---")
            row.update({k: v for k, v in fetch_sample_files(row, cfg, need_image=False).items() if v})
            if not row.get("h5") or not row.get("cells"):
                raise FileNotFoundError(f"{row['sample']}: matrix or cell table missing after download; "
                                        f"supplementary files: {row['supp']}")
            n_cells, n_genes = h5_matrix_shape(row["h5"])
            say(f"matrix header: {n_cells} cells x {n_genes} genes", indent=1)
            if n_genes < cfg.min_panel_genes:
                warn(f"{row['sample']} excluded before image download: panel has only {n_genes} genes "
                     f"(< {cfg.min_panel_genes})")
                continue
            row.update({k: v for k, v in fetch_sample_files(row, cfg, need_image=True).items() if v})
            if not row.get("image"):
                raise FileNotFoundError(f"{row['sample']}: morphology image missing after download; "
                                        f"supplementary files: {row['supp']}")
            row["he_image"] = None
            store.rows[row["sample"]] = row
            a_sub, a_full = ingest_one_sample(row, programs, cfg, store, count_summaries, attach_summaries, subsample_cells)
            if a_sub is None:
                continue
            full_xy[row["sample"]] = a_full.obs[["x", "y"]].values.astype(np.float32)
            row["supp"] = list(row["supp"])
            manifest.append(row)
            parts.append(a_sub)
            say(f"{row['sample']}: {a_sub.n_obs} cells kept for training (slide total after QC {a_full.n_obs}), "
                f"{a_sub.n_vars} genes, {time.time() - t0:.0f}s")
        if not parts:
            raise RuntimeError("no slide survived ingest; check the GEO listing and the exclusion rules")
        adata = ad.concat(parts)
        adata.obs_names_make_unique()
        return adata, manifest, full_xy

    # ------------------------------------------------------------------ component API
    def _run(self, ctx: Context) -> None:
        adata, manifest, full_xy = self._run_demo(ctx) if ctx.cfg.demo_mode else self._run_real(ctx)
        self._write(ctx, adata, manifest, full_xy)
        show_table(adata.obs.groupby(["sample", "stage"], observed=True).size().rename("cells").reset_index(),
                   title="cells per slide (after QC and subsampling)")
        ctx.set("manifest", manifest)
        ctx.set("adata_raw", adata)
        ctx.set("full_xy", full_xy)
        ctx.set("images", self._image_store(ctx, manifest, full_xy))

    def load(self, ctx: Context) -> None:
        import anndata as ad
        from ..data.manifest import load_manifest
        adata = ad.read_h5ad(self.path(ctx, "adata_raw.h5ad"))
        manifest = load_manifest(self.path(ctx, "samples.json"))
        full_xy = {k: v for k, v in np.load(self.path(ctx, "full_xy.npz")).items()}
        say(f"loaded adata_raw ({adata.n_obs} cells x {adata.n_vars} genes), {len(manifest)} slides, "
            f"full_xy for {len(full_xy)} slides")
        ctx.set("manifest", manifest)
        ctx.set("adata_raw", adata)
        ctx.set("full_xy", full_xy)
        ctx.set("images", self._image_store(ctx, manifest, full_xy))
        summary_path = self.path(ctx, "samples.json")
        if os.path.exists(summary_path):
            with open(summary_path) as fh:
                say(f"samples: {[r['sample'] for r in json.load(fh)]}")


def ingest(ctx: Context) -> Context:
    """Component 1: GEO manifest, downloads, Xenium loading, count summaries on every cell, image crops.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Ingest().run_in(ctx)
