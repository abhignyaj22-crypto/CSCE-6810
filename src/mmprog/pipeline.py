"""v2 stage runner. The twelve-component entry is ``main.py`` at the repository root.

This module stays importable. Tests, ingest, and the installed ``mmprog`` command
still call ``Pipeline``. ``python main.py`` is the program that calls the twelve
components in order.

Stage order
    S0 configure        resolve config, device, run directory, gene resources
    S1 ingest           GEO manifest, downloads, Xenium loading, QC, count summaries, subsampling
    S2 targets          program scores at two scales, lineage labels
    S2b reliability     split-half noise ceiling per target and slide
    S3 features         frozen encoder embeddings (memmap) and morphometrics
    S4 cross-validation patient-level CV: full model, ablations, baselines, seeds, bootstrap
    S5 clinical         patient-level program burden, stage trend, optional Cox
    S6 final fit        model on all patients; morphology-only inference check
    S7 report           QA tiers, figures, model card, phase status, CONTRIBUTIONS.md

Every stage writes its artefact to the run directory and skips itself when that
artefact exists, so a run can be resumed after an interruption and a later stage
can be re-run alone with ``--stop-after``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, resolve_device, save_config
from .logging_utils import attach_log_file, banner, say, section, show_table, timed, warn
from .paths import RunDir
from .resources import (load_lineages, load_program_notes, load_programs, load_stage_vocabulary, print_program_annotations,
                        seed_everything)

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z"]


def ingest_one_sample(row, programs, cfg, image_store, count_summaries, attach_summaries, subsample_cells):
    """The per-sample body of real-data ingest, in the order the design requires.

    1. load matrix and centroids (microns -> working-level pixels),
    2. read the image geometry once and reconcile the coordinate axes with it,
    3. cell QC,
    4. count summaries (program counts, neighbourhood sums, density) on EVERY QC-passing cell,
    5. crop, cache and release the image (one decompressed TIFF at a time),
    6. subsample for training and attach the summaries by exact coordinate match.

    Returns ``(subsampled AnnData, full QC-passed AnnData)`` or ``(None, None)`` when excluded.
    ``image_store`` needs ``prepare(sample, xy)``; tests pass a stub.
    """
    from .data.images import check_axes_against_image, level_shape, um_per_px_level0
    from .data.tenx import load_row
    from .preprocessing.qc import apply_cell_qc, calculate_qc_metrics

    sample = row["sample"]
    # one decompression pass serves metadata, geometry and (inside prepare) the crop
    row["um_per_px_l0"] = um_per_px_level0(row["image"], cfg.um_per_px_level0)
    a = load_row(row, cfg)
    if a.n_vars < cfg.min_panel_genes:
        warn(f"{sample} excluded: panel has only {a.n_vars} genes (< {cfg.min_panel_genes})")
        return None, None
    h, w = level_shape(row["image"], cfg.img_level)
    say(f"coordinate check: image level {cfg.img_level} is {w}x{h} px; cells span x<= {a.obs['x'].max():.0f}, "
        f"y<= {a.obs['y'].max():.0f}; um/px at level 0 = {row['um_per_px_l0']}", indent=1)
    xy, swapped = check_axes_against_image(a.obs[["x", "y"]].values, w, h, sample)
    row["axes_swapped"] = bool(swapped)
    if swapped:
        a.obs["x"], a.obs["y"] = xy[:, 0], xy[:, 1]   # before summaries, crops or patches use them

    calculate_qc_metrics(a)
    a = apply_cell_qc(a, cfg)
    a.obs["n_cells_total_slide"] = a.n_obs
    px_per_um = 1.0 / (row["um_per_px_l0"] * 2 ** cfg.img_level)
    summ = count_summaries(a, programs, cfg, px_per_um=px_per_um)   # all QC-passing cells, before subsampling
    a_full = a
    image_store.prepare(sample, a_full.obs[["x", "y"]].values.astype(np.float32))
    a_sub = subsample_cells(a, cfg.max_cells_per_sample, cfg.seed)
    a_sub = attach_summaries(a_sub, summ, a_full)
    return a_sub, a_full


class Pipeline:
    """Holds the state shared between stages and runs them in order."""

    STAGES = ["s0_configure", "s1_ingest", "s2_targets", "s2b_reliability", "s3_features",
              "s4_cross_validation", "s4b_spatial_boost", "s5_clinical", "s5b_manova", "s5c_motifs",
              "s6_final_fit", "s6b_codex", "s7_report"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run = RunDir(cfg)
        attach_log_file(self.run.art("run_log"))
        from .config import say_config_source
        say_config_source(self.cfg)
        # state filled by stages
        self.programs: Dict[str, List[str]] = {}
        self.lineages: Dict[str, List[str]] = {}
        self.stage_order: Dict[str, int] = {}
        self.manifest: List[dict] = []
        self.adata = None
        self.obs: Optional[pd.DataFrame] = None
        self.images = None
        self.full_xy: Dict[str, np.ndarray] = {}
        self.targets: List[str] = []
        self.kept_programs: Dict[str, List[str]] = {}
        self.lin_names: List[str] = []
        self.reliability: Optional[pd.DataFrame] = None
        self.reliability_summary: Optional[pd.DataFrame] = None
        self.rel_lookup: Dict = {}
        self.emb = None
        self.mor = None
        self.morpho_names: List[str] = []
        self.cv_data = None
        self.res: Optional[pd.DataFrame] = None
        self.res_adj: Optional[pd.DataFrame] = None
        self.pred_full: Optional[np.ndarray] = None
        self.fold_logs: Dict[str, pd.DataFrame] = {}
        self.histories: Dict[str, pd.DataFrame] = {}
        self.full_scores: Optional[pd.DataFrame] = None
        self.srb_results: Optional[pd.DataFrame] = None
        self.srb_maps: Dict[str, np.ndarray] = {}
        self.srb_rings: Optional[pd.DataFrame] = None
        self.manova: Optional[pd.DataFrame] = None
        self.motifs: Optional[pd.DataFrame] = None
        self.codex: Optional[dict] = None
        self.multiseed: Optional[pd.DataFrame] = None
        self.decisions: List[str] = []
        self.verdict: Dict[str, bool] = {}
        self.burden: Optional[pd.DataFrame] = None
        self.burden_stats = None
        self.final_model = None
        self.final_fs = None
        self.final_selector = None
        self.selection_report: Optional[dict] = None
        self.device = "cpu"

    # ------------------------------------------------------------------ helpers
    @property
    def main_target(self) -> str:
        for t in MAIN_TARGET_PREFERENCE:
            if t in self.targets:
                return t
        return self.targets[0]

    @property
    def main_idx(self) -> int:
        return self.targets.index(self.main_target)

    def _build_image_store(self):
        """Image store keyed on every QC-passing cell (full_xy) so the crop covers any subsample."""
        from .data.images import ImageStore
        xy_by_sample = {s: self.obs.loc[self.obs["sample"] == s, ["x", "y"]].values.astype(np.float32)
                        for s in self.obs["sample"].unique()}
        for s, xy in self.full_xy.items():
            if s in xy_by_sample and len(xy) >= len(xy_by_sample[s]):
                xy_by_sample[s] = np.asarray(xy, dtype=np.float32)
        demo_images = None
        if self.cfg.demo_mode:
            demo_images = dict(np.load(self.run.art("demo_images")))
        self.images = ImageStore(self.cfg, self.manifest, xy_by_sample, demo_images=demo_images)

    # ------------------------------------------------------------------ S0
    def s0_configure(self) -> None:
        banner("S0 configure")
        self.cfg.validate()
        self.device = resolve_device(self.cfg)
        seed_everything(self.cfg.seed)
        say(self.cfg.describe())
        say(f"compute device: {self.device}")
        save_config(self.cfg, self.run.art("config"))
        self.programs = load_programs()
        self.lineages = load_lineages()
        _, self.stage_order = load_stage_vocabulary()
        notes = load_program_notes()
        for p, note in notes.items():
            say(f"program {p}: {note}", indent=1)
        section("gene annotations (class, evidence grade, role); see docs/research/PATHWAY_INTERPRETATION.md")
        print_program_annotations(self.programs)
        self.run.report()

    # ------------------------------------------------------------------ S1
    def s1_ingest(self) -> None:
        banner("S1 ingest")
        import anndata as ad
        from .data.manifest import load_manifest, save_manifest
        from .preprocessing.spatial_summaries import count_summaries, attach_summaries, subsample_cells

        if self.run.done("adata_raw"):
            self.adata = ad.read_h5ad(self.run.art("adata_raw"))
            self.manifest = load_manifest(self.run.art("manifest"))
            if self.run.done("full_xy"):
                self.full_xy = dict(np.load(self.run.art("full_xy")))
            say(f"S1 loaded from run dir: {self.adata.n_obs} cells x {self.adata.n_vars} genes, {len(self.manifest)} slides")
        elif self.cfg.demo_mode:
            from .data.synthetic import make_demo
            with timed("synthetic fixture"):
                self.adata, images, self.manifest = make_demo(self.programs, self.lineages, seed=self.cfg.seed, cfg=self.cfg)
            self.adata.write_h5ad(self.run.art("adata_raw"))
            np.savez_compressed(self.run.art("demo_images"), **images)
            save_manifest(self.manifest, self.run.art("manifest"))
            self.full_xy = {s: self.adata.obs.loc[self.adata.obs["sample"] == s, ["x", "y"]].values.astype(np.float32)
                            for s in self.adata.obs["sample"].unique()}
            np.savez_compressed(self.run.art("full_xy"), **self.full_xy)
        else:
            self._ingest_real(count_summaries, attach_summaries, subsample_cells, save_manifest)

        self.obs = self.adata.obs.reset_index(drop=True)
        section("S1 summary")
        show_table(self.obs.groupby(["sample", "stage"]).size().rename("cells").reset_index(), title="cells per slide")
        self._build_image_store()

    def _ingest_real(self, count_summaries, attach_summaries, subsample_cells, save_manifest) -> None:
        import anndata as ad
        from .data.download import fetch_sample_files
        from .data.images import ImageStore
        from .data.manifest import build_manifest_from_geo, select_samples
        from .data.tenx import h5_matrix_shape

        mdf = select_samples(build_manifest_from_geo(self.cfg), self.cfg)
        # one store for the whole loop: it crops, caches and deletes each decompressed TIFF in turn
        store = ImageStore(self.cfg, [], {})
        parts = []
        for _, r in mdf.iterrows():
            row = r.to_dict()
            t0 = time.time()
            say(f"--- {row['sample']} ({row['stage']}) ---")
            # matrix and cell table first: the panel-size rule can exclude the sample before the image is fetched
            row.update({k: v for k, v in fetch_sample_files(row, self.cfg, need_image=False).items() if v})
            if not row.get("h5") or not row.get("cells"):
                raise FileNotFoundError(f"{row['sample']}: matrix or cell table missing after download; "
                                        f"supplementary files: {row['supp']}")
            n_cells, n_genes = h5_matrix_shape(row["h5"])
            say(f"matrix header: {n_cells} cells x {n_genes} genes", indent=1)
            if n_genes < self.cfg.min_panel_genes:
                warn(f"{row['sample']} excluded before image download: panel has only {n_genes} genes "
                     f"(< {self.cfg.min_panel_genes})")
                continue
            row.update({k: v for k, v in fetch_sample_files(row, self.cfg, need_image=True).items() if v})
            if not row.get("image"):
                raise FileNotFoundError(f"{row['sample']}: morphology image missing after download; "
                                        f"supplementary files: {row['supp']}")
            row["he_image"] = None
            store.rows[row["sample"]] = row
            a_sub, a_full = ingest_one_sample(row, self.programs, self.cfg, store,
                                              count_summaries, attach_summaries, subsample_cells)
            if a_sub is None:
                continue
            self.full_xy[row["sample"]] = a_full.obs[["x", "y"]].values.astype(np.float32)
            row["supp"] = list(row["supp"])
            self.manifest.append(row)
            parts.append(a_sub)
            say(f"{row['sample']}: {a_sub.n_obs} cells kept for training (slide total after QC {a_full.n_obs}), "
                f"{a_sub.n_vars} genes, {time.time() - t0:.0f}s")
        if not parts:
            raise RuntimeError("no slide survived ingest; check the GEO listing and the exclusion rules")
        self.adata = ad.concat(parts)
        self.adata.obs_names_make_unique()
        self.adata.write_h5ad(self.run.art("adata_raw"))
        save_manifest(self.manifest, self.run.art("manifest"))
        np.savez_compressed(self.run.art("full_xy"), **self.full_xy)

    # ------------------------------------------------------------------ S2
    def s2_targets(self) -> None:
        banner("S2 targets and lineage labels")
        import anndata as ad
        from .preprocessing.qc import apply_cell_qc, calculate_qc_metrics, normalize_log1p
        from .targets.programs import build_targets, score_lineages

        if self.run.done("adata_proc"):
            self.adata = ad.read_h5ad(self.run.art("adata_proc"))
            meta = json.load(open(self.run.art("targets")))
            self.targets, self.lin_names, self.kept_programs = meta["targets"], meta["lineages"], meta["programs"]
            say(f"S2 loaded: {len(self.targets)} targets, {len(self.lin_names)} lineages")
        else:
            if "total_counts" not in self.adata.obs:
                calculate_qc_metrics(self.adata)
            self.adata = apply_cell_qc(self.adata, self.cfg)
            self.adata = normalize_log1p(self.adata)
            self.adata, self.targets, self.kept_programs = build_targets(self.adata, self.programs, self.cfg)
            self.adata, self.lin_names = score_lineages(self.adata, self.lineages, self.cfg)
            self.adata.write_h5ad(self.run.art("adata_proc"))
            with open(self.run.art("targets"), "w") as fh:
                json.dump(dict(targets=self.targets, lineages=self.lin_names, programs=self.kept_programs), fh, indent=2)
        self.obs = self.adata.obs.reset_index(drop=True)
        say(f"targets: {self.targets}")
        say(f"main target for patient-level endpoints: {self.main_target}")
        if "nb_n" in self.obs:
            show_table(self.obs.groupby("sample")["nb_n"].median().rename("median_neighbours_in_radius").reset_index())
        if "lineage_true" in self.obs:
            agreement = float((self.obs["lineage_true"] == self.obs["lineage_name"]).mean())
            say(f"synthetic check: lineage label agreement with planted lineage = {agreement:.3f}")
        self._build_image_store()

    # ------------------------------------------------------------------ S2b
    def s2b_reliability(self) -> None:
        banner("S2b target reliability (noise ceiling)")
        from .targets.reliability import reliability_lookup, reliability_summary, reliability_table

        if self.run.done("reliability"):
            self.reliability = pd.read_csv(self.run.art("reliability"))
            say("S2b loaded")
        else:
            self.reliability = reliability_table(self.obs, self.kept_programs, self.cfg)
            self.reliability.to_csv(self.run.art("reliability"), index=False)
        self.rel_lookup = reliability_lookup(self.reliability)
        self.reliability_summary = reliability_summary(self.reliability)

    # ------------------------------------------------------------------ S3
    def s3_features(self) -> None:
        banner("S3 features: encoder embeddings and morphometrics")
        from .features.encoder import embed_cells, embedding_cache_key, get_encoder
        from .features.morphometrics import MORPHO_NAMES, morphometrics

        key = embedding_cache_key(self.manifest, len(self.obs), self.cfg)
        emb_path = self.run.art(f"emb_{self.cfg.encoder}_p{self.cfg.patch_px}_{key}.npy")
        mor_path = self.run.art(f"morpho_{key}.npy")
        if os.path.exists(emb_path):
            self.emb = np.load(emb_path, mmap_mode="r")
            say(f"embeddings loaded from cache: {self.emb.shape} {self.emb.dtype}")
            if self.emb.shape[0] != len(self.obs):
                raise RuntimeError(f"cached embeddings have {self.emb.shape[0]} rows but there are {len(self.obs)} cells; "
                                   f"delete {emb_path} and re-run S3")
            zero_frac = _zero_row_fraction(self.emb)
            say(f"embedding integrity: {zero_frac:.2%} all-zero rows")
            if zero_frac > 0.01:
                raise RuntimeError(f"{zero_frac:.1%} of cached embedding rows are all zero: the file is truncated "
                                   f"or from an interrupted run; delete {emb_path} and re-run S3")
        else:
            with timed(f"embedding {len(self.obs)} cells with {self.cfg.encoder}"):
                enc = get_encoder(self.cfg.encoder, self.device)
                self.emb = embed_cells(self.obs, self.images, enc, emb_path, self.cfg, self.device)
                del enc
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        if os.path.exists(mor_path):
            self.mor = np.load(mor_path)
            say(f"morphometrics loaded from cache: {self.mor.shape}")
        else:
            with timed("morphometrics"):
                self.mor = morphometrics(self.obs, self.images, self.cfg)
                np.save(mor_path, self.mor)
        self.morpho_names = MORPHO_NAMES(self.cfg)
        say(f"feature branches: embeddings {self.emb.shape}, morphometrics {self.mor.shape} {self.morpho_names}")

    # ------------------------------------------------------------------ S4
    def s4_cross_validation(self) -> None:
        banner("S4 patient-level cross-validation")
        from .evaluation.cross_validation import build_cv_data, run_all_variants, run_multiseed
        from .evaluation.decisions import apply_decision_rules

        self.cv_data = build_cv_data(self.obs, self.targets, self.lin_names, self.cfg)
        if self.run.done("cv_results"):
            self.res = pd.read_csv(self.run.art("cv_results"), index_col=0)
            self.res_adj = pd.read_csv(self.run.art("cv_results_adj"), index_col=0) if self.run.done("cv_results_adj") else None
            self.pred_full = np.load(self.run.art("pred_full"))
            say("S4 loaded")
        else:
            (self.res, self.res_adj, _, self.fold_logs, self.pred_full,
             cv_histories, self.full_scores) = run_all_variants(
                self.cv_data, self.emb, self.mor, self.cfg, self.device, self.run, rel_lookup=self.rel_lookup)
            self.histories.update(cv_histories)
        show_table(self.res, title="per-patient median Pearson r (rows: targets, columns: model variants)")
        if self.res_adj is not None:
            show_table(self.res_adj, title="reliability-adjusted r / sqrt(ceiling)")
        if self.run.done("cv_multiseed"):
            self.multiseed = pd.read_csv(self.run.art("cv_multiseed"), index_col=0)
        elif len(self.cfg.seeds) > 1:
            self.multiseed = run_multiseed(self.cv_data, self.emb, self.mor, self.cfg, self.device, self.run,
                                           rel_lookup=self.rel_lookup, precomputed={self.cfg.seed: self.full_scores})
        self.decisions, self.verdict = apply_decision_rules(self.res, self.res_adj, self.main_target)

    # ------------------------------------------------------------------ S5
    def s5_clinical(self) -> None:
        banner("S5 patient-level endpoints")
        from .evaluation.clinical import cox_pfs, program_burden, stage_trend

        if self.cfg.target_norm == "slide":
            warn("slide-standardised targets have no between-patient component; the stage trend is not interpretable")
        self.burden = program_burden(self.obs, self.pred_full, self.main_idx, self.cfg, self.stage_order)
        self.burden.to_csv(self.run.art("burden"), index=False)
        self.burden_stats = stage_trend(self.burden)
        if self.cfg.clinical_csv:
            cox_pfs(self.burden, self.cfg.clinical_csv, self.run.art("cox_pfs"))
        else:
            say("no clinical table supplied: PFS analysis skipped (set clinical_csv with patient, pfs_months, pfs_event)")

    # ------------------------------------------------------------------ S6
    def s6_final_fit(self) -> None:
        banner("S6 final fit and morphology-only inference check")
        from .inference.final_fit import fit_final_model, load_final_model
        from .inference.virtual_map import sanity_check_on_training_slide, virtual_program_map

        from .features.selection import save_selection_report

        if self.run.done("final_model"):
            self.final_model, self.final_fs, self.final_selector, _ = load_final_model(self.run, self.cfg, self.device)
        else:
            self.final_model, self.final_fs, self.final_selector = fit_final_model(
                self.cv_data, self.emb, self.mor, self.cfg, self.device, self.run)
            if self.final_selector is not None:
                save_selection_report(self.final_selector, self.run.art("feature_selection"))
        if self.run.done("feature_selection"):
            self.selection_report = json.load(open(self.run.art("feature_selection")))
        if self.run.done("final_fit_history.csv"):
            self.histories["final_fit"] = pd.read_csv(self.run.art("final_fit_history.csv"))
        s0 = self.obs["sample"].iloc[0]
        vmap_name = f"virtual_map_{s0}.csv"
        if self.run.done(vmap_name):
            vmap = pd.read_csv(self.run.art(vmap_name))
            say(f"inference check loaded from {vmap_name} ({len(vmap)} nuclei)")
        else:
            say(f"inference smoke test on slide {s0} (in deployment this is an unseen slide)")
            vmap = virtual_program_map(self.images.get(s0), self.final_model, self.final_fs, self.final_selector,
                                       self.cfg, self.device, self.targets, self.cv_data.n_samples)
            vmap.to_csv(self.run.art(vmap_name), index=False)
        obs_slide = self.obs[self.obs["sample"] == s0]
        sanity_check_on_training_slide(vmap, obs_slide, self.images.offset(s0), self.main_target, self.cfg)


    # ------------------------------------------------------------------ S4b
    def s4b_spatial_boost(self) -> None:
        """Spatially regularized boosting (Xiang et al. 2009) on patch-grid features.

        Answers a question the MLP cannot: where in the 27 um neighbourhood does the
        program signal live? The importance map is spatially coherent by construction
        of the regulariser, and the lambda = 0 column is the AdaBoost control.
        """
        banner("S4b spatially regularized boosting on patch-grid features")
        if not self.cfg.srb_enabled:
            say("srb_enabled is false: stage skipped")
            return
        from .evaluation.spatial_boost_eval import ring_summary, run_spatial_boost_cv
        from .features.patch_grid import grid_locations_um, patch_grid_features

        grid = self.cfg.srb_grid
        import hashlib
        # the cell set, not only its size, determines the features: hash the cell identifiers into the key
        cell_hash = hashlib.md5("|".join(map(str, self.adata.obs_names[:: max(1, len(self.obs) // 5000)])).encode()).hexdigest()[:8]
        key = f"grid{grid}_p{self.cfg.patch_px}_{len(self.obs)}_{cell_hash}"
        grid_path = self.run.art(f"patch_grid_{key}.npy")
        if os.path.exists(grid_path):
            G = np.load(grid_path)
            say(f"patch grid features loaded from cache: {G.shape}")
        else:
            G = patch_grid_features(self.obs, self.images, self.cfg, grid=grid)
            np.save(grid_path, G)
        locations = grid_locations_um(grid, self.cfg.patch_px, self.cfg.um_per_px)
        if self.run.done("srb_results"):
            self.srb_results = pd.read_csv(self.run.art("srb_results"))
            self.srb_maps = dict(np.load(self.run.art("srb_maps")))
            say("S4b loaded")
        else:
            y = self.obs[self.main_target].values.astype(np.float64)
            self.srb_results, self.srb_maps, _ = run_spatial_boost_cv(
                G, locations, y, self.cv_data.SID, self.cv_data.GROUPS, self.cfg, grid,
                lambdas=tuple(self.cfg.srb_lambdas), radius_um=self.cfg.srb_radius_um,
                n_rounds=self.cfg.srb_rounds, max_train_cells=self.cfg.srb_max_train_cells, seed=self.cfg.seed)
            self.srb_results.to_csv(self.run.art("srb_results"), index=False)
            np.savez_compressed(self.run.art("srb_maps"), **self.srb_maps)
        rings = []
        for k, m in self.srb_maps.items():
            r = ring_summary(m, grid, self.cfg.patch_px / grid * self.cfg.um_per_px)
            r.insert(0, "model", k)
            rings.append(r)
        self.srb_rings = pd.concat(rings, ignore_index=True)
        self.srb_rings.to_csv(self.run.art("srb_rings"), index=False)
        show_table(self.srb_rings, title="importance by block-centre distance from the nucleus (centre < 5 um, inner 5 to 10 um, outer >= 10 um); per_block = share / number of blocks:")
        say(f"main target for the boosting task: {self.main_target}")

    # ------------------------------------------------------------------ S5b
    def s5b_manova(self) -> None:
        """Patient-level MANOVA: does the 4-program vector differ by disease stage, measured and predicted?"""
        banner("S5b patient-level MANOVA of program vectors by stage")
        from .evaluation.manova import run_manova_stage

        if self.run.done("manova"):
            self.manova = pd.read_csv(self.run.art("manova"))
            say("S5b loaded")
            show_table(self.manova)
            return
        # Non-redundant variables: the composite 18-gene set is a function of its two subsets, so it is dropped
        # when both subsets are present (otherwise the within-group covariance is near singular).
        targets = list(self.targets)
        if any("wnt_core" in t for t in targets) and any("adhesion_survival" in t for t in targets):
            targets = [t for t in targets if "prog_wnt_camdr_" not in t]
            say("MANOVA variables exclude the composite wnt_camdr set because wnt_core and adhesion_survival are present")
        # Stages collapsed to control / precursor / myeloma: with 2 MGUS patients a four-group test is not estimable.
        collapse = {"control": "control", "MGUS": "precursor", "SMM": "precursor", "MM": "myeloma", "RM": "myeloma"}
        obs2 = self.obs.copy()
        obs2["stage"] = obs2["stage"].astype(str).map(lambda s: collapse.get(s, s))   # plain str: no stale categories
        say("stages collapsed for MANOVA: " + ", ".join(f"{k}->{v}" for k, v in collapse.items()))
        order = {"control": 0, "precursor": 1, "myeloma": 2}
        pred_cols = [self.targets.index(t) for t in targets]
        self.manova = run_manova_stage(obs2, self.pred_full[:, pred_cols], targets, self.run, self.cfg, order)

    # ------------------------------------------------------------------ S5c
    def s5c_motifs(self) -> None:
        """Neighbourhood motif discovery by byte-pair encoding over lineage sequences; enrichment for the main program."""
        banner("S5c neighbourhood motifs (byte-pair encoding over lineage sequences)")
        if not self.cfg.motifs_enabled:
            say("motifs_enabled is false: stage skipped")
            return
        from .analysis.motifs import run_motif_stage

        if self.run.done("motifs"):
            self.motifs = pd.read_csv(self.run.art("motifs"))
            say("S5c loaded")
            show_table(self.motifs.head(10))
        else:
            self.motifs = run_motif_stage(self.obs, self.main_target, self.cfg, self.run, nb_table=self.cv_data.NB)

    # ------------------------------------------------------------------ S6b
    def s6b_codex(self) -> None:
        """Data-gated CODEX bridge: run the DAPI model on a CODEX section and validate against protein markers."""
        banner("S6b CODEX spatial-proteomics bridge")
        from .evaluation.codex_validation import run_codex_stage

        if self.run.done("codex"):
            self.codex = json.load(open(self.run.art("codex")))
            say("S6b loaded")
            return
        self.codex = run_codex_stage(self.cfg, self.run, self.final_model, self.final_fs, self.final_selector,
                                     self.targets, self.cv_data.n_samples, self.device, self.lin_names)

    # ------------------------------------------------------------------ S7
    def s7_report(self) -> None:
        banner("S7 report: QA tiers, figures, model card, contributions")
        from .reporting import (flag_tiers, importance_map_figure, overlay_figure, reliability_figure, results_figure,
                                training_curves, write_contributions, write_model_card)

        flag_tiers(self.obs, self.pred_full, self.main_idx, self.main_target, self.run)
        overlay_figure(self.obs, self.pred_full, self.main_idx, self.main_target, self.images, self.run.art("overlays"))
        results_figure(self.res, self.run.art("cv_results.png"))
        reliability_figure(self.reliability_summary, self.res, self.run.art("reliability.png"))
        if self.histories:
            training_curves(self.histories, self.run.art("training_curves.png"))
        if self.srb_maps:
            importance_map_figure(self.srb_maps, self.cfg.srb_grid, self.cfg.patch_px / self.cfg.srb_grid * self.cfg.um_per_px,
                                  self.run.art("srb_importance_maps.png"))
        cohorts = sorted(self.obs["cohort"].unique())
        phase_status = pd.DataFrame([
            dict(phase=1, name="Xenium within-cohort", data_present=True, ran=self.run.done("cv_results")),
            dict(phase=2, name="Visium HD + transfer", data_present=len(cohorts) >= 2, ran=self.run.done("transfer")),
            dict(phase="2b", name="CODEX proteomics bridge", data_present=bool(self.cfg.codex_table and self.cfg.codex_dapi),
                 ran=self.run.done("codex")),
            dict(phase=3, name="Clinical endpoints", data_present=bool(self.cfg.clinical_csv or self.cfg.commpass_csv),
                 ran=self.run.done("cox_pfs") or self.run.done("cox_signature")),
            dict(phase=4, name="Encoder / stain harmonisation", data_present=True,
                 ran=self.cfg.encoder != "resnet50" or self.cfg.stain_mode == "hematoxylin"),
            dict(phase=5, name="Batch deployment", data_present=bool(self.cfg.slide_dir), ran=self.run.done("batch_burden")),
        ])
        phase_status.to_csv(self.run.art("phase_status"), index=False)
        show_table(phase_status, title="phase status")
        write_model_card(self.run, self.cfg, self.obs, self.targets, self.kept_programs, self.morpho_names,
                         self.reliability_summary, self.res, self.res_adj, self.decisions, self.verdict, phase_status,
                         selection_report=self.selection_report)
        write_contributions(self.run, self.cfg, self.res, self.res_adj, self.reliability_summary, self.decisions,
                            self.verdict, self.main_target, burden_stats=self.burden_stats, multiseed=self.multiseed,
                            n_cells=len(self.obs), n_patients=int(self.obs["patient"].nunique()),
                            srb_results=self.srb_results, srb_rings=self.srb_rings, manova=self.manova,
                            motifs=self.motifs, codex=self.codex)
        self.run.report()

    # ------------------------------------------------------------------ driver
    def run_all(self, stop_after: Optional[str] = None) -> "Pipeline":
        t0 = time.time()
        banner(f"v2 stage runner: run '{self.cfg.run_name}' (the twelve-component entry is main.py)")
        for name in self.STAGES:
            t1 = time.time()
            getattr(self, name)()
            say(f"stage {name} finished in {time.time() - t1:.1f}s")
            if stop_after and name == stop_after:
                say(f"stopping after {stop_after} as requested")
                break
        say(f"pipeline finished in {(time.time() - t0) / 60:.1f} min; artefacts in {self.run.path}")
        return self


def _zero_row_fraction(emb, chunk: int = 50000) -> float:
    """Fraction of all-zero rows, read chunk by chunk so a float16 memmap is never copied whole."""
    n_zero = 0
    for i in range(0, len(emb), chunk):
        block = np.asarray(emb[i:i + chunk])
        n_zero += int((~block.any(axis=1)).sum())
    return n_zero / max(len(emb), 1)


def main(cfg: Config, stop_after: Optional[str] = None) -> Pipeline:
    """Run the complete pipeline with a resolved configuration."""
    return Pipeline(cfg).run_all(stop_after=stop_after)
