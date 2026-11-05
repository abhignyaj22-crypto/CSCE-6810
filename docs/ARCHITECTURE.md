# Architecture and module contracts

This file is the contract between modules. Each module exposes the functions
listed here with exactly these signatures. The orchestrator (`mmprog/pipeline.py`)
calls them in order and nothing else. Agents implementing a module may add private
helpers (leading underscore) but must not change a public signature.

Shared objects
- `cfg: Config` (`mmprog.config.Config`), always passed explicitly, never global.
- `run: RunDir` (`mmprog.paths.RunDir`), owns artefact paths; `run.art(name)`, `run.done(name)`.
- `say / section / banner / show_table / timed` from `mmprog.logging_utils`: every module prints through these.
- `programs: Dict[str, List[str]]`, `lineages: Dict[str, List[str]]` from `mmprog.resources`.
- `adata: anndata.AnnData` with `obs` columns: `x, y` (pixels at `cfg.img_level`), `sample, patient, stage, cohort, platform`,
  QC columns `total_counts, n_genes_by_counts`, count summaries `cnt_total, cnt_<program>, nb_n, nb_cnt_total, nb_cnt_<program>, density_r<r>`,
  targets `prog_<program>_<space>` and `prog_<program>_<space>_z`, lineage scores `lin_<name>`, `lineage` (int), `lineage_name`.
- `OBS = adata.obs.reset_index(drop=True)`: the row order used by every array below (EMB, MOR, Y, XY, SID, LIN, NB, GROUPS).
- `images: ImageStore`: `images.get(sample) -> np.uint8 array (H, W) or (H, W, 3)`; `images.offset(sample) -> (x0, y0)` crop origin to subtract from `obs.x, obs.y`.

Dependencies allowed: numpy, pandas, scipy, scikit-learn, scikit-image, tifffile, anndata, h5py, torch, timm, torchvision, matplotlib, pyyaml, lifelines (optional, imported lazily), GEOparse (optional, lazy). No scanpy (we implement the three functions we need: 10x h5 reading, normalize+log1p, marker scoring).

## mmprog/data

`manifest.py`
- `build_manifest_from_geo(cfg) -> pd.DataFrame` columns `sample, patient, stage, title, cohort, platform, supp(list[str])`. Human samples only; excludes `cfg.exclude_title_regex`; stage parsed with `resources.load_stage_vocabulary()`. Prints the table.
- `select_samples(mdf, cfg) -> pd.DataFrame`: `sample_limit` rule = one per stage first, then fill.
- `ManifestRow = Dict[str, Any]` with keys `sample, patient, stage, title, cohort, platform, supp, h5, cells, image, he_image, um_per_px_l0`.
- `save_manifest(rows, path)`, `load_manifest(path)`.

`download.py`
- `fetch_sample_files(row, cfg, need_image=True) -> dict(h5=..., cells=..., image=...)` (wget -c, skips transcripts and boundaries, extracts tar once, marker file).
- `ensure_raw(row, cfg, need_image=True) -> row` re-downloads what is missing.
- `ensure_decompressed(path) -> path` streaming gunzip once, prints size and time.

`tenx.py` (10x readers without scanpy)
- `read_10x_h5(path) -> anndata.AnnData` (CSR float32 X, `var_names` = gene names made unique, `var["gene_ids"]`).
- `load_xenium(row, cfg) -> AnnData` (obs x,y in working-level pixels, sample/patient/stage), `load_visium_hd(row, cfg)`, `load_row(row, cfg)`.

`images.py`
- `um_per_px_level0(path, default) -> float` from OME-XML `PhysicalSizeX`.
- `level_shape(path, level) -> (H, W)`.
- `load_image(path, level, bbox=None, z_project="max", stain_mode="native") -> np.uint8` memmap decode, z-projection inside bbox in 512-row blocks, chunked uint8 conversion, never a whole-slide float copy.
- `class ImageStore(cfg, manifest_rows, obs_xy_by_sample: Dict[str, np.ndarray], demo_images=None)` with `get(sample)`, `offset(sample)`; uses the crop cache `cfg.crop_dir/<sample>_L<level>_z<zp>_m<margin>_<stain>.npz`; transposed-axes check; deletes the decompressed multi-GB copy after cropping; keeps one image in memory at a time.

`synthetic.py`
- `make_demo(programs, lineages, n_patients=6, cells_per=3000, img_size=1024, seed=7) -> (AnnData, Dict[str, np.uint8 image], List[ManifestRow])` with `obs.lineage_true` and the planted spatial structure (plasma blobs carrying the wnt_camdr program, exhausted T cells in blobs). Must also fill `cnt_*`, `nb_*`, `density_*` through `preprocessing.spatial_summaries` so the synthetic run exercises the whole pipeline.

## mmprog/preprocessing

`qc.py`
- `calculate_qc_metrics(adata) -> adata` adds `total_counts, n_genes_by_counts` (in place, returns adata).
- `apply_cell_qc(adata, cfg) -> adata` filters and prints kept/dropped per sample with medians.
- `normalize_log1p(adata, target_sum=1e4) -> adata` (keeps raw counts in `adata.layers["counts"]`).

`spatial_summaries.py`
- `count_summaries(adata_full, programs, cfg) -> pd.DataFrame` indexed by obs_names: `cnt_total, cnt_<p>, nb_n, nb_cnt_total, nb_cnt_<p>, density_r<r>` on ALL QC-passing cells (before subsampling); counts are sums so split-half thinning is valid.
- `attach_summaries(adata_sub, summ, adata_full) -> adata_sub` exact coordinate join with assertion.
- `subsample_cells(adata, max_cells, seed) -> adata`.

`nuclei.py`
- `detect_nuclei(img, min_area=8, max_area=400, bg_sigma=25, tile=2048, overlap=64) -> np.ndarray (N, 2) float32 (x, y)`; tiled watershed with median-based polarity test; prints count and polarity decision.

`stain.py`
- `harmonise_stain(img, stain_mode) -> np.uint8`.

## mmprog/targets

`programs.py`
- `program_score(cnt, total) -> np.ndarray` = `log1p(1e4 * cnt / max(total, 1))`.
- `standardise(series, groups, mode) -> np.ndarray`.
- `build_targets(adata, programs, cfg) -> (adata, targets: List[str], kept_programs: Dict[str, List[str]])` prints panel coverage per program and skips programs with `< cfg.min_program_genes` genes in panel.
- `score_lineages(adata, lineages, cfg) -> (adata, lin_names: List[str])`: marker score = mean log expression of markers minus mean of an expression-matched random background (scanpy's score_genes logic, implemented here); argmax -> `lineage`, `lineage_name`. Prints the composition table per sample.

`reliability.py`
- `split_half(cnt_p, total, rng) -> ((a, ta), (b, tb))` binomial thinning.
- `reliability_table(obs, kept_programs, cfg) -> pd.DataFrame` columns `target, sample, r_half, r_sb, n_cells`.
- `reliability_summary(tab) -> pd.DataFrame` index target, columns `median, min, max, ceiling_sqrt`; prints it with the one-line reading guide.
- `reliability_lookup(tab) -> Dict[(target, sample), r_sb]`.

## mmprog/features

`patches.py`
- `extract_patch(img, x, y, p) -> np.uint8 (p, p, 3)` zero-padded at borders.
- `preprocess_batch(patches_uint8, device) -> torch.Tensor (B, 3, 224, 224)` ImageNet-normalised.

`encoder.py`
- `get_encoder(name, device) -> torch.nn.Module` with `.num_features` (resnet50 via timm `num_classes=0`; `uni` gated; `conch` raises NotImplementedError with the pointer).
- `embed_cells(obs, images, enc, out_path, cfg, device) -> np.memmap (N, D)` float16 memmap, per-sample loop, prints cells/s.
- `embedding_cache_key(manifest_rows, n_obs, cfg) -> str`.

`morphometrics.py`
- `MORPHO_NAMES(cfg) -> List[str]` = `p_mean, p_std, p_p10, p_p90, p_darkfrac, density_r<r>...`.
- `morphometrics(obs, images, cfg) -> np.ndarray (N, 5 + len(radii)) float32`; uses `obs.density_r<r>` when present (full-slide density), else computes from the available centroids.

`feature_space.py`
- `class FeatureSpace(pca_dim, use_morpho, fit_max=100000, chunk=20000)` with `fit(emb, mor, idx=None, seed=0, keep_emb_dims=None, keep_mor_dims=None) -> self`, `transform(emb, mor) -> np.ndarray float32`, property `dim`, attribute `explained_variance_ratio_`. Standardise, PCA on embeddings, standardise morphometrics, concatenate. Prints variance explained by the kept PCs.

`selection.py` (scientifically justified, fit INSIDE each CV fold on training cells only)
- `class FeatureSelector(cfg)` with `fit(emb, mor, Y, obs, train_idx) -> self` and attributes `keep_emb_dims: np.ndarray[int]`, `keep_mor_dims: np.ndarray[int]`, `report: dict`.
  Steps, each printed: (1) variance floor on embedding dims (drops dead ReLU channels); (2) correlation pruning among morphometrics at `selection_corr_threshold`; (3) univariate screen: Spearman of each morphometric with each target on training cells, reported not used for dropping (avoids selection bias); (4) PCA retained-variance report (how many PCs reach 90%/95%); (5) permutation importance on a ridge model over [PCs + morphometrics] as a report. The selector drops only by variance and redundancy (steps 1-2); everything else is a report, so no target leakage enters the model. Rationale written in `docs/DECISIONS.md`.
- `save_selection_report(selector, path)`.

## mmprog/models

`neighbours.py`
- `knn_table(obs, k) -> np.ndarray (N, k) int64` per-sample kNN; `norm_xy(obs) -> np.ndarray (N, 2) float32` in [-1, 1] per slide.

`program_net.py`
- `class ProgramNet(nn.Module)(d_in, n_targets, n_lin, n_samples, hidden, dropout, use_spatial, use_comp, n_fourier, fourier_scale, slide_emb_dim)`; `forward(x, xy, sid) -> (y_hat, comp_logits or None)`; index `n_samples` is the unseen-slide embedding (zeros). `describe()` prints parameter counts per block.

`losses.py`
- `program_loss(yhat_pos, y, yhat_nb, keep_mask, comp_logits, lin, lambda_smooth, lambda_comp, use_spatial, use_comp) -> (loss, parts: dict)`.

`train.py`
- `train_model(X, Y, XY, SID, LIN, NB, tr_idx, va_idx, n_samples, n_lin, cfg, device, use_spatial=True, use_comp=True, seed=0, verbose=True) -> (model, best_val, epochs_run, history: pd.DataFrame)`; neighbourhood-union minibatching with searchsorted and clamp; early stopping on inner validation patient; prints per-epoch train loss parts and val MSE.
- `predict(model, X, XY, SID, device, batch=8192) -> np.ndarray`.

## mmprog/evaluation

`cross_validation.py`
- `class CVData` (dataclass): `Y, XY, SID, LIN, NB, GROUPS, ONEHOT, targets, lin_names, n_samples`; `build_cv_data(obs, targets, lin_names, cfg) -> CVData`.
- `run_cv(variant, data, EMB, MOR, cfg, device, seed, selector_enabled) -> (preds, Y_used, fold_log: pd.DataFrame)`; variants `full, no_spatial, no_comp, no_morpho, shuffled, oracle`; feature space and selector fit on inner-train cells only.
- `composition_only(data, cfg) -> preds` ridge on one-hot lineage.
- `run_all_variants(data, EMB, MOR, cfg, device, run) -> (RES, RES_ADJ, per_patient_df, fold_logs, pred_full)` writes the csv artefacts.
- `run_multiseed(...) -> pd.DataFrame`.

`metrics.py`
- `score(preds, Y, targets, groups, rel_lookup, min_cells) -> pd.DataFrame` (pooled Pearson/Spearman, per-patient median, reliability-adjusted median, per_patient dicts).
- `bootstrap_ci(per_patient: dict, n_boot, seed) -> (lo, hi)`.

`decisions.py`
- `apply_decision_rules(RES: pd.DataFrame, RES_ADJ, main_target) -> List[str]`: implements the four rules (full > shuffled and > composition_only; full ~ composition_only; oracle >> full; no_spatial ~ full) with numeric margins; returns plain-language statements, prints them. This is what feeds CONTRIBUTIONS.md.

`clinical.py`
- `program_burden(obs, pred_full, main_target_idx, cfg, stage_order) -> pd.DataFrame`; `stage_trend(burden) -> (rho, p, n)`; `cox_pfs(burden, clinical_csv, path) -> Optional[pd.DataFrame]` lazy lifelines; `signature_concordance(...)` for Phase 3 gate.

`transfer.py`
- `run_transfer(train_cohort, test_cohort, data, EMB, MOR, cfg, device, seed) -> pd.DataFrame` with unseen-slide embedding.

## mmprog/inference

`final_fit.py`
- `fit_final_model(data, EMB, MOR, cfg, device, run) -> (model, feature_space, selector)` saved to `final_model.pt` with `torch.save(dict(state, fs, selector, targets, lineages, cfg))`; `load_final_model(run, cfg, device, n_targets, n_lin, n_samples)`.

`virtual_map.py`
- `virtual_program_map(img, model, fs, selector, cfg, device, targets, n_samples, encoder=None) -> pd.DataFrame` columns `x, y, sample, <target>_pred`.
- `sanity_check_on_training_slide(vmap, obs_slide, offset, main_target, cfg) -> float` matched-nucleus Pearson.

`export.py`
- `export_qupath(vmap, path, um_per_px)`, `export_xenium_explorer(vmap, path, um_per_px)`, `batch_inference(slide_dir, ..., run)`.

## mmprog/reporting

`qa_tiers.py` -> `flag_tiers(obs, pred_full, main_idx, main_target, run) -> (tier2_df, tier3_df)`.
`figures.py` -> `overlay_figure(obs, pred_full, main_idx, main_target, images, path, n_samples=4)`, `reliability_figure`, `results_figure(RES, path)`, `training_curves(history_by_fold, path)`.
`model_card.py` -> `write_model_card(...) -> dict`.
`contributions.py` -> `write_contributions(run, RES, RES_ADJ, reliability_summary, decisions, burden_stats, cfg) -> str` writes `CONTRIBUTIONS.md` in the run dir: what was established, with the numbers, and what was not.

## mmprog/pipeline.py

`class Pipeline(cfg)` with one method per stage `s0_configure … s8_report`, each idempotent through `run.done`, and `run_all()`. `scripts/run_pipeline.py` exposes `main(argv=None)` that parses `--config`, `--set key=value`, `--stage`, and calls `Pipeline(cfg).run_all()`.
