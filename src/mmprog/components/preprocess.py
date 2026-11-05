"""Preprocess: cell QC, normalisation, missing-gene mask, nucleus detection.

Steps, each printed with its counts:
1. QC metrics and the cell filter (``min_transcripts``, ``min_genes``), per-sample table.
2. ``normalize_log1p`` (counts kept in ``layers["counts"]``; the factorisation and the
   split-half reliability read the counts, the lineage scores read the log values).
3. Missing-gene mask: a panel gene with zero counts in any sample is masked for every
   downstream use, because a program weight on a gene that one sample cannot measure
   would make that sample's score artificially low. This is the missing-value rule for
   panel genes.
4. Nucleus detection on each morphology crop (classical watershed, D21), stored in crop
   coordinates with the crop offset, so deployment and training see the same detector.

Count summaries are not recomputed here; Ingest computed them on all cells (D6).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, show_table, timed, warn


class Preprocess(BaseComponent):
    name = "preprocess"
    requires = ("adata_raw", "images", "manifest")
    provides = ("adata", "obs", "gene_mask", "nuclei")
    artefacts = ("adata.h5ad", "gene_mask.npy", "nuclei.npz", "qc_summary.csv")
    config_fields = ("min_transcripts", "min_genes", "demo_mode", "img_level", "patch_px")
    depends_on = ("ingest",)

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _gene_mask(adata) -> np.ndarray:
        from scipy.sparse import csr_matrix
        counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
        X = csr_matrix(counts)
        samples = adata.obs["sample"].to_numpy()
        present = np.ones(adata.n_vars, dtype=bool)
        for s in pd.unique(samples):
            per_gene = np.asarray(X[samples == s].sum(axis=0)).ravel()
            present &= per_gene > 0
        n_masked = int((~present).sum())
        say(f"missing-gene mask: {n_masked}/{adata.n_vars} genes have zero counts in at least one sample and are masked")
        if n_masked:
            say("first masked genes: " + ", ".join(adata.var_names[~present][:20]), indent=1)
        return present

    @staticmethod
    def _detect_all(ctx: Context) -> Dict[str, np.ndarray]:
        from ..preprocessing.nuclei import detect_nuclei
        nuclei: Dict[str, np.ndarray] = {}
        obs = ctx.adata.obs
        for sample in pd.unique(obs["sample"].to_numpy()):
            img = ctx.images.get(sample)
            ox, oy = ctx.images.offset(sample)
            with timed(f"nucleus detection on {sample} ({img.shape[1]}x{img.shape[0]} px)"):
                pts = detect_nuclei(img)
            n_seg = int(ctx.full_xy[sample].shape[0]) if ctx.full_xy and sample in ctx.full_xy else int((obs["sample"] == sample).sum())
            ratio = len(pts) / max(n_seg, 1)
            say(f"{sample}: {len(pts)} nuclei detected vs {n_seg} segmented cells (ratio {ratio:.2f}); crop offset ({ox}, {oy})", indent=1)
            if ratio < 0.5 or ratio > 2.0:
                warn(f"{sample}: detector and segmentation disagree by more than 2x; check polarity and the area window")
            nuclei[sample] = np.column_stack([pts, np.full(len(pts), ox, dtype=np.float32),
                                              np.full(len(pts), oy, dtype=np.float32)]).astype(np.float32)
        return nuclei

    # ------------------------------------------------------------------ component API
    def _run(self, ctx: Context) -> None:
        from ..preprocessing.qc import apply_cell_qc, calculate_qc_metrics, normalize_log1p
        adata = ctx.adata_raw.copy()
        n0 = adata.n_obs
        if "total_counts" not in adata.obs:
            calculate_qc_metrics(adata)
        adata = apply_cell_qc(adata, ctx.cfg)
        say(f"QC: kept {adata.n_obs}/{n0} cells ({ctx.cfg.min_transcripts} transcripts, {ctx.cfg.min_genes} genes minimum)")
        totals = adata.obs["total_counts"].to_numpy()
        adata = normalize_log1p(adata)
        from ..observations import observe
        observe(ctx.cfg.run_dir, "log1p_normalise",
                before=f"raw counts, median library size {np.median(totals):.0f} over {len(totals)} cells",
                after=f"each cell scaled to 10000 counts then log1p; {adata.n_obs} x {adata.n_vars}",
                note="raw counts stay in layers['counts']; lineage scores read the log matrix",
                examples=[f"cell {i} on {adata.obs['sample'].iloc[i]} raw total {totals[i]:.0f}"
                          for i in range(0, min(len(totals), 5))])
        qc = (adata.obs.groupby("sample", observed=True)
              .agg(cells=("total_counts", "size"), median_transcripts=("total_counts", "median"),
                   median_genes=("n_genes_by_counts", "median")).reset_index())
        show_table(qc, title="per-sample QC after filtering")
        qc.to_csv(self.path(ctx, "qc_summary.csv"), index=False)

        gene_mask = self._gene_mask(adata)
        adata.var["measured_in_all_samples"] = gene_mask
        masked = list(adata.var_names[~gene_mask][:5])
        observe(ctx.cfg.run_dir, "gene_mask",
                before=f"{adata.n_vars} panel genes",
                after=f"{int(gene_mask.sum())} measured in every slide, {int((~gene_mask).sum())} masked",
                note="a gene with zero counts in any slide is masked so it cannot pull that slide's program score down",
                examples=masked or ["no gene was masked"])
        ctx.set("adata", adata)
        ctx.set("obs", adata.obs.reset_index(drop=True))
        ctx.set("gene_mask", gene_mask)

        nuclei = self._detect_all(ctx)
        ctx.set("nuclei", nuclei)
        ratios = []
        for sample, pts in list(nuclei.items())[:5]:
            n_seg = int(ctx.full_xy[sample].shape[0]) if ctx.full_xy and sample in ctx.full_xy else 0
            ratios.append(f"{sample}: {len(pts)} detected nuclei, {n_seg} segmented cells")
        observe(ctx.cfg.run_dir, "nucleus_detection",
                before="morphology crop plus the Xenium cell coordinates",
                after=f"{sum(len(v) for v in nuclei.values())} watershed nuclei over {len(nuclei)} slides",
                note="a ratio far from 1 means the detector and the segmentation disagree",
                examples=ratios)

        adata.write_h5ad(self.path(ctx, "adata.h5ad"))
        np.save(self.path(ctx, "gene_mask.npy"), gene_mask)
        np.savez_compressed(self.path(ctx, "nuclei.npz"), **nuclei)
        say(f"wrote adata.h5ad ({adata.n_obs} cells), gene_mask.npy ({int(gene_mask.sum())} usable genes), "
            f"nuclei.npz ({sum(len(v) for v in nuclei.values())} nuclei), qc_summary.csv")

    def load(self, ctx: Context) -> None:
        import anndata as ad
        adata = ad.read_h5ad(self.path(ctx, "adata.h5ad"))
        gene_mask = np.load(self.path(ctx, "gene_mask.npy"))
        nuclei = {k: v for k, v in np.load(self.path(ctx, "nuclei.npz")).items()}
        say(f"loaded adata ({adata.n_obs} cells x {adata.n_vars} genes), {int(gene_mask.sum())} usable genes, "
            f"nuclei for {len(nuclei)} slides")
        ctx.set("adata", adata)
        ctx.set("obs", adata.obs.reset_index(drop=True))
        ctx.set("gene_mask", gene_mask)
        ctx.set("nuclei", nuclei)


def preprocess(ctx: Context) -> Context:
    """Component 2: cell QC, log normalisation, missing-gene mask across slides, nucleus detection.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Preprocess().run_in(ctx)
