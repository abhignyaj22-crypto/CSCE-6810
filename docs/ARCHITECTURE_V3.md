# Architecture v3: one entry point, one context object, twelve components

This document is the contract for the v3 repository. It replaces the stage methods of
`mmprog/pipeline.py` (v2) with a single `main.py` that runs a fixed sequence of
components. Every component is a class with the same interface, reads and writes a
typed context object, writes one artefact set, and skips itself when that artefact set
already exists. The scientific design (DECISIONS.md D1 to D27) does not change; v3 changes
how the code is organised, adds a unified multimodal model, and makes the training stack
explicit for a single NVIDIA H100 80 GB. v3.1 (D33 to D36) makes `main.py` call every
component directly, adds the `external_validation` component, replaces MANOVA by
PERMANOVA and an ordered trend test as the default stage test, and adds release gates
(`RESULTS.json`) and a reproducibility record (`environment.json`).

Rules carried over from v2 unchanged: `cfg` is passed explicitly and never global; every
number in a report comes from a file in the run directory (D23); patient-level
cross-validation only (D18); baselines, ablations and decision rules are mandatory (D19);
no scanpy dependency; no em dashes in any generated document.

## 1. Entry point

`main.py` at the repository root is the only executable. `main()` reads like an ordinary
program: it builds the configuration and the run context, then calls the twelve
components directly, one line each, and prints the release gates last:

```
cfg = load_config(args.config, args.set)          # YAML + section.key=value overrides
plan = RunPlan.from_args(args.only, args.force)     # run / load / skip per component
ctx = start_run(cfg, plan, argv)                    # Context.new, ctx.plan, environment.json
ctx = ingest(ctx)
ctx = preprocess(ctx)
ctx = infer_gene_programs(ctx)
ctx = extract_features(ctx)
ctx = select_features(ctx)
ctx = build_unified_model(ctx)
ctx = train_unified_model(ctx)
ctx = evaluate(ctx)
ctx = interpret_programs(ctx)
ctx = validate_externally(ctx)
ctx = deploy(ctx)
ctx = report(ctx)
write_run_manifest(ctx, t0)                         # run_manifest.json
print_run_summary(ctx)                              # release gates, key artefact paths
```

Each function is defined at the bottom of its own module under `src/mmprog/components/`
(`ingest.py` ends with `def ingest(ctx)`, and so on), so "go to definition" lands on the
component; `mmprog.components` only re-exports them. The function body is one line,
`<Class>().run_in(ctx)`, and the class carries the contract. `COMPONENT_ORDER` in
`core/pipeline.py` is the canonical order used to validate `--only` and `--force`:

```
ingest, preprocess, program_inference, feature_extraction, feature_selection,
unified_model, training, evaluation, interpretation, external_validation, deployment, reporting
```

Command line: `python main.py --config configs/default.yaml --set training.max_epochs=40
--only evaluation,reporting --force training --dry-run`. `--only` runs exactly the named
components; components before the last requested one are loaded from their artefacts
(computed if the artefacts are missing or stale) and components after it are skipped.
`--force` recomputes the named components and everything after them even when their
artefacts are valid. `--dry-run` prints the resolved configuration and what each
component would do under the plan, then exits without creating a run. `--list-components`
prints the twelve names. `run_pipeline(cfg, only, force)` in `core/pipeline.py` is the same
sequence as one function, for tests and notebooks.

`start_run` writes `<run_dir>/environment.json`: Python, platform, torch, CUDA and GPU
name, numpy, pandas and scikit-learn versions, git commit, the command line, the start
time and the resolved configuration. It is the reproducibility record a second machine
is compared against.

## 2. Component interface

```python
class Component(Protocol):
    name: str                       # "ingest", "preprocess", ...
    requires: tuple[str, ...]       # ctx fields that must be non-None on entry
    provides: tuple[str, ...]       # ctx fields set on exit
    artefacts: tuple[str, ...]      # files written under run_dir/<name>/

    def run(self, ctx: Context, force: bool = False) -> Context: ...
    def is_done(self, ctx: Context) -> bool: ...     # all artefacts exist and manifest hash matches
    def load(self, ctx: Context) -> None: ...        # fill `provides` from artefacts without recomputing
    def execute(self, ctx: Context, mode: str, force: bool) -> Context: ...   # run | load | skip
    def run_in(self, ctx: Context) -> Context: ...   # execute under ctx.plan; mode "run" when ctx.plan is None
```

`BaseComponent` implements the shared behaviour once:

- `run` checks `requires`, calls `is_done`; if done it calls `load` and returns; otherwise
  it opens a `section(name)` log block, calls the subclass `_run(ctx)`, verifies that every
  field in `provides` is now set and every file in `artefacts` exists, writes
  `run_dir/<name>/manifest.json` (config section hash, input artefact hashes, git commit,
  wall time, device), and returns ctx.
- Subclasses implement `_run` and `load` only. No subclass touches another component's
  directory.
- `run_in(ctx)` is what the one-line component functions call: it reads the `RunPlan`
  stored on `ctx.plan` by `start_run` and dispatches to `execute` with that component's
  mode (run, load or skip) and forced flag. A context built by hand (a notebook, a unit
  test) has no plan and runs the component normally.

Skip-if-done rule: a component is done when every file in `artefacts` exists under
`run_dir/<name>/` and `manifest.json` records the same hash of (its own config section,
the manifests of the components it requires). A config change in an upstream section
therefore invalidates everything downstream, and `--force` is only needed to redo a
component whose inputs are unchanged. This replaces v2's `run.done(name)` marker files,
which did not record what the artefact was computed from.

## 3. The context object

`Context` is a frozen-field dataclass (fields are assigned once; a component that needs
to replace a field must do so through `ctx.replace(field=value)`, which logs the change).
Arrays are numpy or memmap, never torch tensors, so the context pickles and the CPU-only
test path works.

```python
@dataclass
class Context:
    # run
    cfg: Config
    run_dir: Path
    device: str                                  # "cuda" or "cpu"
    log: Logger
    seed: int

    # Ingest
    manifest: list[ManifestRow] | None = None    # sample, patient, stage, cohort, platform, paths, um_per_px_l0
    adata_raw: AnnData | None = None             # counts, obs x y sample patient stage, var gene names
    images: ImageStore | None = None             # images.get(sample), images.offset(sample)

    # Preprocess
    adata: AnnData | None = None                 # QC-passed, normalised, layers["counts"], obs qc columns
    obs: pd.DataFrame | None = None              # adata.obs.reset_index(drop=True); row order for all arrays below
    gene_mask: np.ndarray | None = None          # (G,) bool, False for panel genes missing in any sample
    nuclei: dict[str, np.ndarray] | None = None  # sample -> (N_s, 2) detected centroids on the morphology image
    summaries: pd.DataFrame | None = None        # cnt_*, nb_*, density_r* on all QC-passing cells

    # ProgramInference
    programs: dict[str, list[str]] | None = None     # literature-seeded anchors (resources/gene_programs.json)
    program_weights: pd.DataFrame | None = None      # (P, G) learned gene weights, index program names, columns var_names
    program_scores: np.ndarray | None = None         # (N, P, S) per-cell scores at S scales (cell, nb)
    targets: list[str] | None = None                 # "prog_<p>_<scale>_z"
    Y: np.ndarray | None = None                      # (N, T) standardised targets
    lineage: np.ndarray | None = None                # (N,) int
    lin_names: list[str] | None = None
    LIN: np.ndarray | None = None                    # (N, L) one-hot composition
    reliability: pd.DataFrame | None = None          # target, sample, r_half, r_sb, n_cells
    rel_lookup: dict[tuple[str, str], float] | None = None

    # FeatureExtraction
    EMB: np.memmap | None = None                 # (N, D_enc) float16, encoder embedding of the centre patch
    GRID: np.memmap | None = None                # (N, g*g, D_grid) float16, patch-grid tokens (g x g blocks)
    MOR: np.ndarray | None = None                # (N, M) float32 morphometrics
    morpho_names: list[str] | None = None
    NB: np.ndarray | None = None                 # (N, k) int64 per-sample kNN indices into obs rows
    NB_DIST: np.ndarray | None = None            # (N, k) float32 distances in um
    XY: np.ndarray | None = None                 # (N, 2) float32 slide-normalised coordinates in [-1, 1]
    SID: np.ndarray | None = None                # (N,) int sample index; n_samples is the unseen-slide slot
    GROUPS: np.ndarray | None = None             # (N,) patient id for GroupKFold

    # FeatureSelection (fit inside folds; this field holds the report and the fold-free part)
    selection: SelectionReport | None = None     # keep_emb_dims, keep_mor_dims, report dict, written per fold too

    # UnifiedModel
    model_spec: ModelSpec | None = None          # resolved dims, token counts, head list, parameter count

    # Training
    folds: list[Fold] | None = None              # train/val/test patient ids per outer fold
    cv_preds: dict[str, np.ndarray] | None = None   # variant -> (N, T) out-of-fold predictions
    cv_unc: dict[str, np.ndarray] | None = None     # variant -> (N, T) predictive std
    cv_comp: dict[str, np.ndarray] | None = None    # variant -> (N, L) composition probabilities
    histories: dict[str, pd.DataFrame] | None = None
    final_model_path: Path | None = None

    # Evaluation
    results: pd.DataFrame | None = None          # per variant, per target: pooled r, per-patient median, CI
    results_adj: pd.DataFrame | None = None      # reliability-adjusted
    calibration: pd.DataFrame | None = None      # coverage of 50/90 % intervals per target
    decisions: list[str] | None = None           # plain-language outcomes of the decision rules
    stage_assoc: pd.DataFrame | None = None      # PERMANOVA + Jonckheere-Terpstra rows (stage_association.csv)
    manova: pd.DataFrame | None = None           # only when evaluation.stage_test is "manova" or "both"

    # Interpretation
    srb_maps: dict[str, np.ndarray] | None = None        # target -> (g, g) block importance
    srb_rings: pd.DataFrame | None = None
    pathway_attr: pd.DataFrame | None = None             # program token attention mass per target
    gene_backbone: pd.DataFrame | None = None            # genes whose weights survive stability selection

    # ExternalValidation
    validation: dict[str, dict] | None = None            # validator -> {status: ran|skipped|failed, reason, headline numbers}
    validation_tables: dict[str, pd.DataFrame] | None = None   # validator -> its table (also written as CSV)
    codex: dict | None = None                            # the protein_bridge entry, kept under its v2 name for reporting

    # Deployment
    vmaps: dict[str, pd.DataFrame] | None = None         # sample -> x, y, <target>_pred, <target>_std
    sanity: pd.DataFrame | None = None                   # sanity_check.csv as a one-row table
    exports: list[Path] | None = None

    # Reporting
    model_card: dict | None = None
    contributions_path: Path | None = None
    gates: dict[str, dict] | None = None                 # release gates: name -> {passed, status, value, threshold, source, meaning}

    # bookkeeping
    plan: RunPlan | None = None                          # set by start_run; read by BaseComponent.run_in
```

`Fold` is `(fold_id, train_patients, val_patient, test_patients)` in `core/context.py`.
The selection report, model spec and manifest rows are plain dicts in the implementation
(`selection`, `model_spec`, `manifest`); the dataclasses of the first draft were not
introduced. The field list above is the contract; `src/mmprog/core/context.py` is
authoritative for names and comments.

## 4. Components, artefacts, reuse

Every component writes under `run_dir/<name>/`. "Reused verbatim" means the v2 module is
imported unchanged and wrapped; "rewritten" means a new module in v3.

### 4.1 Ingest

- Reads the GEO manifest, selects samples, downloads what is missing, loads 10x matrices
  and centroids into working-level pixels, reconciles coordinate axes with the image
  geometry, prepares the image crop cache. Synthetic mode replaces downloads with
  `data/synthetic.make_demo`.
- Provides `manifest, adata_raw, images, full_xy`.
- Artefacts: `samples.json`, `adata_raw.h5ad`, `full_xy.npz` (and `images.npz` in demo
  mode); the crop cache lives in the shared `crops/` directory (key includes level,
  projection, margin and stain mode as in D3). `manifest.json` is the component manifest.
- Reused verbatim: `data/manifest.py`, `data/download.py`, `data/tenx.py`,
  `data/images.py`, `data/synthetic.py`, `data/synthetic_codex.py`, `data/codex.py`.
- Reused: the per-sample loop `pipeline.ingest_one_sample` is imported by
  `components/ingest.py`; QC and count summaries on every cell stay inside it (before
  subsampling, D6), so Preprocess does not recompute them.

### 4.2 Preprocess

- The D6 order (QC on every cell, count summaries on every QC-passing cell, then
  subsampling) is executed inside Ingest; Preprocess re-applies the QC thresholds to the
  subsampled cells, log-normalises, and adds missing-gene masking (a gene absent from any sample's panel is masked
  in `gene_mask` and excluded from program learning so that a 5K-panel slide and a smaller
  panel are scored on the intersection), stain handling (`harmonise_stain` on each crop
  when `stain_mode != native`), and nucleus detection on the morphology image so that
  the same `nuclei` field serves training (sanity check against Xenium centroids) and
  deployment (the only centroid source).
- Provides `adata, obs, gene_mask, nuclei`.
- Artefacts: `adata.h5ad`, `gene_mask.npy`, `nuclei.npz`, `qc_summary.csv`.
- Reused verbatim: `preprocessing/qc.py`, `preprocessing/spatial_summaries.py`,
  `preprocessing/stain.py`, `preprocessing/nuclei.py`.
- Rewritten: the component wrapper (gene mask computed inline).

### 4.3 ProgramInference

- Learns gene programs from expression with literature-seeded anchors. Input: normalised
  log expression of QC-passing cells restricted to `gene_mask`. Anchors are the v2 gene
  sets in `resources/gene_programs.json` (D24). Method: non-negative matrix factorisation with
  anchor regularisation (each program's anchor genes receive a prior weight; the penalty
  pulls weights of non-anchor genes toward zero unless the data support them), fitted once
  on all patients (D28: unsupervised, no morphology, no labels; the dependence of the target
  definition on test-patient expression is stated in the report), and refitted on random
  patient subsamples only for the stability-selection backbone (Interpretation). Output per program: gene weights (G,) and per-cell
  scores at the cell scale and the 30 um neighbourhood scale (D7), standardised across the
  cohort (D8). The v2 fixed-anchor scores (`program_score` on summed counts) remain as the
  `anchor_only` target set so the gain from learned weights is measured, not assumed.
- Split-half reliability (D9) is computed on the anchor-count targets, where binomial
  thinning is exact; for learned-weight targets the ceiling is computed by thinning the
  counts and re-projecting with fixed weights, and this is stated in the report.
- Lineage labels by marker-score argmax (D10) stay here because the composition head
  consumes them.
- Provides `programs, program_weights, program_scores, targets, Y, lineage, lin_names,
  LIN, reliability, rel_lookup`.
- Artefacts: `program_weights.csv`, `program_scores.npy`, `targets.json`, `Y.npy`,
  `lineage.csv`, `target_reliability.csv`, `reliability_summary.csv`, `panel_coverage.csv`.
- Reused verbatim: `targets/programs.py` (program_score, standardise, score_lineages),
  `targets/reliability.py`, `resources.py`.
- Rewritten: `programs/anchored_nmf.py` (new), `programs/targets_builder.py` (replaces
  `build_targets` so learned and anchor-only targets share one builder).

### 4.4 FeatureExtraction

- Four outputs from the morphology image, all indexed by `obs` rows: encoder embedding of
  the centre patch (`EMB`), a g x g grid of block embeddings over the patch (`GRID`,
  the tokens the unified model attends over and the blocks the boosting map scores),
  morphometrics (`MOR`, D13), and the neighbourhood graph (`NB, NB_DIST, XY, SID, GROUPS`).
- Encoders are registered by name: `resnet50` (timm, ImageNet, frozen, D11), `uni`,
  `virchow2`, `hoptimus0` through timm `hf-hub:` ids, `phikon_v2` through transformers
  `AutoModel`; `conch` is listed in section 9 but not registered (its loader does not go
  through timm); `virchow2` returns the 1280-d class token, not the card's 2560-d
  concatenation. Gated encoders require a Hugging Face
  token; the component prints the licence string from the registry and refuses to run a
  non-commercial-licensed encoder when `cfg.features.commercial_use` is true. Encoder
  registry facts are listed in section 9.
- Provides `EMB, GRID, MOR, morpho_names, NB, NB_DIST, XY, SID, GROUPS`.
- Artefacts: `emb.f16.npy`, `grid.f16.npy` (float16 memmaps; the cache key with encoder
  name, patch size, grid and cell-id hash is recorded in `encoder_card.json`), `morpho.npy`,
  `graph.npz`, `encoder_card.json` (name, dim, input size, licence, source URL, weight hash).
- Reused verbatim: `features/patches.py`, `features/morphometrics.py`,
  `models/neighbours.py`, `features/encoder.py::embedding_cache_key`.
- Rewritten: `features/encoders/registry.py` (replaces `get_encoder`),
  `features/embed.py` (replaces `embed_cells`; adds GRID extraction, channels_last,
  bf16 autocast, pinned DataLoader, optional `torch.compile`), `features/patch_grid.py`
  is kept but its output becomes a memmap.

### 4.5 FeatureSelection

- Unchanged policy (D14): drop dead embedding dimensions and redundant morphometrics;
  report everything else. Fitted inside each fold on training cells by Training; this
  component fits the fold-free report on all cells for `feature_selection.json` and
  validates that the fold-time selector is deterministic for a given train index.
- Provides `selection`.
- Artefacts: `feature_selection.json`, `selection_report.csv`.
- Reused verbatim: `features/selection.py`, `features/feature_space.py`.

### 4.6 UnifiedModel

- Builds (does not train) the single multimodal network and writes its specification.
  Tokens:
  - morphology tokens: the g*g GRID block embeddings plus one centre token from EMB,
    each projected to `d_model`, with learned 2-D block position embeddings;
  - spatial neighbour tokens: for each of the k neighbours, its centre embedding projected
    to `d_model` plus a relative-offset encoding (random Fourier features of dx, dy in um,
    D16) and the slide embedding (zero slot for unseen slides);
  - program and pathway tokens: one learned query per program (and per pathway subset such
    as `wnt_core`, `adhesion_survival`), initialised from the program weight vector
    projected through a gene-to-token linear map, so the token carries the gene content.
  Blocks: L layers of self-attention over morphology + neighbour tokens with a block-diagonal
  mask that lets neighbour tokens attend to their own centre only; then cross-attention
  where program tokens are queries and morphology + neighbour tokens are keys and values.
  Heads from the attended program tokens:
  - continuous program scores: linear, T outputs;
  - lineage composition: linear over the pooled centre token, L logits (multi-task loss, D15);
  - calibrated uncertainty: heteroscedastic Gaussian head (mean and log-variance per target)
    trained with the Gaussian negative log-likelihood, post-hoc temperature scaling on the
    inner validation patient, coverage reported by Evaluation.
  Losses: Gaussian NLL on targets, cross-entropy on composition, Laplacian smoothness over
  the k neighbours computed on the union batch as in v2 (`losses.program_loss` is reused for
  the smoothness and composition terms).
- Ablation variants are configuration, not code: `no_neighbours` (k = 0), `no_grid`
  (centre token only), `no_program_tokens` (skips every token and attention layer; the raw
  centre feature vector goes straight into a v2-shaped two-layer MLP head, so the variant is
  the v2 head on the same features; the v2 ProgramNet itself, with its coordinate prior, is
  kept as the regression reference), `no_comp`, `no_smooth`.
- Attention uses `F.scaled_dot_product_attention` with the boolean block mask; because a
  mask is supplied, the flash kernel will not be selected and the memory-efficient or math
  kernel will run (section 8). Head dimension is fixed at 64 so that the fused kernels
  remain eligible when no mask is needed (the self-attention mask is present whenever
  neighbour tokens are used, in training and in inference; only the program cross-attention
  and the no_neighbours variant run without a mask).
- Provides `model_spec`.
- Artefacts: `model_spec.json` (dims, token counts, parameter count per block, the
  attention mask description).
- Reused verbatim: `models/losses.py`, `models/program_net.py` (kept as the regression
  reference for the `no_program_tokens` variant, tests/regression).
- Rewritten: `models/unified.py` (new), `models/tokens.py` (new), `models/heads.py` (new).

### 4.7 Training

- Patient-level GroupKFold with seven outer folds and one inner validation patient for
  early stopping and temperature scaling (D17, D18); three seeds (D20). The out-of-fold
  predictions for every variant are the only predictions Evaluation sees. A final model
  on all patients is fitted last for Deployment.
- Data path: cells are sampled by neighbourhood-union minibatches as in v2; GRID and EMB
  rows are gathered from memmaps into a pinned host buffer by a DataLoader with
  `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`, and copied with
  `non_blocking=True`.
- H100 settings (facts in section 8): `torch.autocast("cuda", dtype=torch.bfloat16)`
  with no GradScaler; `torch.backends.cuda.matmul.allow_tf32 = True` and
  `torch.backends.cudnn.allow_tf32 = True` (the matmul flag defaults to False); model and
  inputs in `channels_last` for the convolutional encoder only; `torch.compile(model)` in
  mode `default` with `dynamic=False` and fixed batch shapes (the last partial batch is
  padded and masked, never shrunk, so no recompilation); `reduce-overhead` (CUDA graphs) is
  off by default and enabled by config only when the whole step is static. No
  DistributedDataParallel: one process, one GPU.
- Provides `folds, cv_preds, cv_unc, cv_comp, histories, final_model_path`.
- Artefacts: `folds.json`, `cv_preds_<variant>.npy`, `cv_unc_<variant>.npy`,
  `cv_comp_<variant>.npy`, `history_<variant>_f<k>_s<seed>.csv`, `final_model.pt`
  (state dict, feature space, selector, targets, lineage names, cfg, model_spec),
  `hardware.json` (device name, memory, torch version, bf16 and compile flags actually used).
- Reused verbatim: `evaluation/cross_validation.py::build_cv_data` and
  `composition_only`, `inference/final_fit.py` loading helpers.
- Rewritten: `training/loop.py` (replaces `models/train.py`; the v2 loop is kept callable
  for the `no_program_tokens` regression), `training/cv.py` (replaces `run_cv`,
  `run_all_variants`, `run_multiseed`; variants become a list in config).

### 4.8 Evaluation

- Scores every variant with v2 metrics (pooled and per-patient median Pearson and
  Spearman, patient bootstrap CI, reliability-adjusted values), adds calibration
  (empirical coverage of 50 % and 90 % Gaussian intervals and of the split-conformal
  interval whose quantile comes from the in-fold validation predictions in
  `training/cv_val_preds_full.npz`, expected calibration error on the predictive std),
  the decision rules (D19) with the v3 ablation sentences, clinical burden and stage
  trend, and the patient-level stage association (D34): PERMANOVA on the patient vector
  of predicted neighbourhood-scale program burdens and a Jonckheere-Terpstra trend test
  per program along control < MGUS < SMM < MM, both run on the predicted and on the
  measured vectors, with Benjamini-Hochberg q-values across programs. When every stage
  has at least two patients and there are at least three stages the full order is used;
  otherwise stages collapse to control / precursor / myeloma. The classical MANOVA (D26)
  still runs when `evaluation.stage_test` is `manova` or `both`. Cross-cohort transfer
  (`evaluation/transfer.py`) is not run by the component until a second cohort exists.
  Baselines: `shuffled`, `composition_only`, `oracle`, `anchor_only_targets` (the model
  trained on fixed-anchor targets, to test whether learned programs are more predictable
  from morphology or merely different).
- Requires `cv_preds, Y, targets, GROUPS, obs, rel_lookup, reliability_summary`.
- Provides `results, results_adj, calibration, decisions, verdict, stage_assoc, burden,
  burden_stats, multiseed` (`manova` only under `stage_test: manova|both`).
- Artefacts: `cv_results.csv`, `cv_results_reliability_adjusted.csv`, `calibration.csv`,
  `decisions.txt`, `program_burden.csv`, `stage_association.csv` (columns `vectors, test,
  program, statistic, p_value, q_value, r2_or_rho, n_patients, n_groups, groups, note`;
  `vectors` in {predicted, measured}, `test` in {permanova, jt_trend}, `program` is `all`
  for PERMANOVA, a `note` starting with `skipped:` marks a degenerate test),
  `stage_patient_table.csv`, `per_patient.csv`; `manova_results.csv` and `cox_pfs.csv`
  when their inputs exist.
- Config fields in the skip hash: `n_boot, min_cells_per_patient_metric, coverage_levels,
  conformal_alpha, stage_test, stage_n_perm, manova_n_perm, clinical_csv,
  min_plasma_for_burden, n_folds`.
- Reused verbatim: `evaluation/metrics.py`, `evaluation/decisions.py`,
  `evaluation/clinical.py`, `evaluation/transfer.py`, `evaluation/manova.py`.
- New: `evaluation/calibration.py`, `evaluation/stage_association.py`
  (`permanova`, `jonckheere_terpstra`, `bh_qvalues`, `stage_association_table`),
  `components/evaluation.py` (the component).

### 4.9 Interpretation

- Three outputs. (1) Spatially regularized boosting (D25) over the GRID blocks, lambda = 0
  as control, ring summary. (2) Pathway attribution: for each target, the cross-attention
  mass each program and pathway token places on centre, neighbour-block and neighbour-cell
  tokens, averaged over out-of-fold cells, plus integrated gradients from each target head
  back to the program token inputs; written per target and per lineage. (3) Gene backbone:
  stability selection over refits of the anchored factorisation on 20 random half-patient
  subsamples (genes whose weight is at least `backbone_threshold` of the program's largest
  weight in at least `backbone_min_fold_fraction` of the refits), reported with the evidence grades from
  `resources/gene_annotations.json` (D24) so a backbone gene whose literature evidence
  points the other way is flagged in the same table.
- Provides `srb_maps, srb_rings, pathway_attr, gene_backbone`.
- Artefacts: `srb_results.csv`, `srb_maps.npz`, `srb_rings.csv`, `pathway_attribution.csv`,
  `attention_maps.npz`, `gene_backbone.csv`.
- Reused verbatim: `models/spatial_boost.py`, `evaluation/spatial_boost_eval.py`,
  `analysis/motifs.py` (motifs stay as an exploratory table under this component).
- Rewritten: `interpretation/attribution.py` (new), `interpretation/backbone.py` (new).

### 4.10 ExternalValidation (new in v3.1, D35)

- Modality-agnostic validation beyond the cross-validation metrics. Five validators run
  in a fixed order; each records `status` (`ran`, `skipped` or `failed`), a `reason` and
  its headline numbers in `validation_summary.json`, and a validator that raises is
  recorded as `failed` with the exception text while the others still run (set
  `strict=True` on the class or the environment variable `MMPROG_STRICT_VALIDATION` to
  re-raise in tests). The CODEX bridge (D27) is now one of the five, not a deployment step.
  1. `program_specificity` (always): out-of-fold predictions of the full model against
     expression-matched random gene sets (Tirosh et al. 2016 control-set construction,
     adaptation) and against two random halves of the program's weighted genes.
  2. `spatial_fidelity` (always): Moran's I of predicted and measured target on the kNN
     graph per patient, and the plasma-cell-dense niche contrast (Cohen's d) predicted
     versus measured.
  3. `perturbation_robustness` (when `final_model_path` and the image store exist): the
     morphology-only path on one slide's fixed detected nuclei under each entry of
     `validation_perturbations` (`gamma_<g>`, `blur_<sigma>`, `rot90`, `noise_<sd>`,
     `scale_<f>`), Spearman against the clean prediction per target.
  4. `independent_cohort` (when `validation_cohort_manifest` is set): an independent
     Xenium or Visium HD cohort loaded through `ingest_one_sample`, scored morphology-only,
     compared with the cohort's own anchor scores and the learned programs projected onto
     its genes.
  5. `protein_bridge` (when `codex_table` and `codex_dapi` are set): the earlier CODEX
     bridge, unchanged in behaviour (`validation/protein_bridge.py`, thin re-export in
     `evaluation/codex_validation.py`).
- Requires `cv_preds, Y, targets, obs, adata, gene_mask, program_weights, programs, GROUPS,
  NB, lineage, lin_names`; `final_model_path` and `images` are optional and gate validators
  3 to 5.
- Provides `validation, validation_tables` (and sets `ctx.codex` to the `protein_bridge`
  entry so reporting keeps its v2 reader).
- Artefacts: `validation_summary.json` (always); when the validator ran,
  `program_specificity.csv` and `program_specificity_summary.csv`, `spatial_fidelity.csv`,
  `perturbation_robustness.csv`, `independent_cohort.csv`, `protein_bridge.csv`,
  `protein_bridge.json`, `protein_bridge_matched_nuclei.csv`. Column lists are in
  `docs/research/VALIDATION_METHODS.md` and `docs/RESULTS_SPEC.md`.
- Config fields in the skip hash: `validation_n_random_sets, validation_perturbations,
  validation_cohort_manifest, codex_table, codex_dapi, codex_um_per_px,
  codex_dapi_um_per_px, nb_radius_um, knn_k, encoder, patch_px, img_level, z_project,
  stain_mode, target_spaces, target_norm`.
- New: `components/external_validation.py`, `validation/{perturbations, program_specificity,
  spatial_fidelity, robustness, independent_cohort, protein_bridge}.py`.

### 4.11 Deployment

- Morphology-only inference on a DAPI or H&E image: `harmonise_stain`, `detect_nuclei`,
  morphometrics from centroids, encoder on patches, neighbour graph from detected
  centroids, unified model with the zero slide embedding (the neighbour block mask is
  still applied; only the program cross-attention runs without a mask), outputs
  `x, y, <target>_pred, <target>_std, lineage_prob_*`. Sanity check on a training slide
  (matched-nucleus Pearson against the measured target, calibration offset, 90 %
  coverage) plus the inference parity number (D36): the same path fed the slide's own
  annotated centroids against the training path (precomputed features, saved feature
  space, selector and temperature) on the same cells, written as `parity_pearson`. Batch
  inference over `slide_dir`. Exports for QuPath and Xenium Explorer (D22). The CODEX
  bridge moved to ExternalValidation.
- Inference settings: `torch.inference_mode()`, bf16 autocast, fixed patch batch size with
  padding, `torch.compile` reused from training when the artefact records the same shapes.
- Requires `final_model_path, images, obs, targets, lin_names`.
- Provides `vmaps, sanity, exports`.
- Artefacts: `vmap_check.csv` (the training-slide check), `sanity_check.csv` (columns
  `slide, main_target, n_detected, n_segmented, n_matched, match_um, pearson, pred_mean,
  measured_mean, coverage_90, parity_pearson, parity_error, um_per_px`), `vmap_<slide>.csv` and `batch_burden.csv`
  for batch slides, `exports/<slide>_qupath.tsv`, `exports/<slide>_xenium_explorer.csv`,
  `exports/README.txt`.
- Config fields in the skip hash: `slide_dir, encoder, patch_px, knn_k, img_level,
  z_project, stain_mode, nb_radius_um`.
- Reused verbatim: `inference/export.py`.
- Rewritten: `inference/virtual_map_v3.py` (the model call changes; nucleus detection and
  patch extraction stay; v2 `inference/virtual_map.py` is untouched).

### 4.12 Reporting

- Writes `model_card.json` (data, encoder card, model spec, hardware.json, every metric
  file name it draws from), figures, QA tiers, the release gates and `CONTRIBUTIONS.md`
  from the decision statements, reliability summary, calibration, burden, stage
  association, backbone, enrichment, attribution and gate tables. Nothing in the report is
  computed here; every number is read from an artefact (D23).
- Release gates (D36, `reporting/gates.py`): nine checks, each read from one artefact and
  one column; a gate whose artefact or column is absent is `not_evaluated`, never passed.
  Written to `reporting/release_gates.csv` and `<run_dir>/RESULTS.json`
  (`main_target`, `gates`, `summary` with passed / failed / not_evaluated counts), and
  printed by `main.py` last. The gates: `above_chance`, `beyond_identity`,
  `ci_excludes_zero`, `shuffled_at_chance` (cv_results.csv), `calibrated`
  (calibration.csv), `backbone_stable` (gene_backbone.csv), `inference_parity`
  (sanity_check.csv), `stage_association` (stage_association.csv, predicted vectors only),
  `external_validation` (validation_summary.json: at least one validator ran, every
  validator that ran wrote `passed: true` from its own stated bar, none failed; a ran
  validator without a verdict leaves the gate not evaluated). `inference_parity` also
  fails when `parity_error` is non-empty (the inference path raised).
- Requires `results, decisions, reliability_summary, obs`; depends on evaluation,
  interpretation, external_validation and deployment manifests.
- Provides `model_card, contributions_path, gates`.
- Artefacts: `model_card.json`, `CONTRIBUTIONS.md`, `release_gates.csv`,
  `figures/cv_results.png` and other figures, `qa_tier2.csv`, `qa_tier3.csv`,
  `phase_status.csv`; `RESULTS.json` at the run root.
- Reused verbatim: `reporting/figures.py`, `reporting/qa_tiers.py`,
  `reporting/model_card.py`; `reporting/contributions.py` extended with the stage
  association, calibration, backbone, enrichment, attribution and gate sections;
  `reporting/gates.py` new.

## 5. Logging

- One logger (`mmprog/logging_utils.py` reused verbatim: `say, section, banner,
  show_table, timed, warn, describe_array, progress`) writes to stdout and to
  `run_dir/run.log`; every component's `_run` is wrapped by `section(name)` and `timed`.
- Each component also writes `run_dir/<name>/manifest.json` (section 2); the driver
  writes `run_dir/run_manifest.json` with every component's hash, seconds and whether it
  was recomputed or loaded.
- Training writes per-epoch loss parts and validation NLL to the history CSV and prints
  them; it also prints `torch.cuda.max_memory_allocated()` per fold into `hardware.json`.
- Warnings that change the science (a program dropped for panel coverage, a slide with
  swapped axes, slide-level standardisation with a stage endpoint) go through `warn` and
  are copied into `CONTRIBUTIONS.md` by Reporting.

## 6. Config schema

One YAML file with one section per component plus `run`. Config is one flat dataclass;
the YAML may be grouped by section and is flattened on load (DESIGN_V3.md section 3).
`--set section.key=value` overrides with type coercion as in v2 `config.py`. Each
component hashes the fields in its `config_fields` tuple plus `seed`.

```
run:            run_name, root, local_data, seed, device, demo_mode, phases
ingest:         geo_accession, sample_limit, exclude_title_regex, min_panel_genes, platform,
                img_level, z_project, stain_mode, crop_margin
preprocess:     min_transcripts, min_genes, max_cells_per_sample, nb_radius_um,
                nuclei: {min_area, max_area, bg_sigma, tile, overlap}, mask_missing_genes
programs:       anchor_file, n_programs, anchor_prior_weight, l1_nonanchor, max_iter,
                target_spaces, target_norm, min_program_genes, reliability_seed,
                include_anchor_only_targets
features:       encoder, encoder_token_env, commercial_use, patch_px, grid, emb_batch,
                emb_dtype, density_radii_px, knn_k, num_workers, pin_memory,
                persistent_workers, prefetch_factor, channels_last, compile_encoder
selection:      enabled, variance_floor, corr_threshold, report_top, pca_dim
model:          d_model, n_layers, n_heads, head_dim, dropout, n_fourier, fourier_scale,
                slide_emb_dim, uncertainty: {kind: gaussian, temperature_scaling: true},
                lambda_comp, lambda_smooth
training:       n_folds, seeds, variants, batch_cells, max_epochs, patience, lr,
                weight_decay, warmup_steps, amp: {enabled, dtype: bfloat16}, tf32: true,
                compile: {enabled, mode: default, dynamic: false}, cuda_graphs: false,
                grad_clip
evaluation:     n_boot, min_cells_per_patient_metric, coverage_levels, conformal_alpha,
                stage_test (permanova | manova | both), stage_n_perm, manova_n_perm,
                clinical_csv, commpass_csv, min_plasma_for_burden
interpretation: srb_* (grid, lambdas, radius_um, rounds, max_train_cells),
                attribution_n_cells, ig_steps, backbone_threshold, backbone_min_fold_fraction
validation:     validation_n_random_sets, validation_perturbations,
                validation_cohort_manifest, codex_table, codex_dapi, codex_um_per_px,
                codex_dapi_um_per_px
deployment:     slide_dir
reporting:      figures, n_overlay_samples
```

The names above are the flat `Config` field names as the code reads them
(`src/mmprog/config.py` is authoritative); the nested spellings of the first draft were not
implemented.

`configs/default.yaml` is the GSE299193 run, `configs/smoke.yaml` four samples,
`configs/synthetic.yaml` the demo fixture with `run.device: cpu`, `configs/h100.yaml`
the training section tuned for one H100 (bf16, tf32, compile default, batch_cells to be
set after a measured sweep; see section 8, batch sizes).

## 7. Test strategy

- Unit tests, one file per component under `tests/unit/` (plus `test_pipeline.py` for the
  run plan, `--dry-run` and `--list-components`, and `test_external_validation.py`), run
  on CPU in a few minutes; tests marked `slow` build a real checkpoint and check that
  interpretation, deployment and the inference parity number work on it. Each test constructs a `Context` from a small synthetic fixture (`tests/fixtures.py`
  wraps `data/synthetic.make_demo` with 3 patients and 300 cells), runs one component, and
  asserts `provides` fields, artefact files, the manifest hash, and the skip rule (second
  `run` returns without recomputation, verified by a counter on `_run`). Science
  invariants carried from v2 tests are kept: ingest order (summaries before subsampling),
  reliability near zero on a null target and high on a planted one, selector leakage test
  (selector fitted on train index does not change when test rows are permuted), decision
  rules on hand-built result tables, MANOVA degenerate cases, export round trips.
- Model tests: `UnifiedModel` forward shapes at tiny dims; the block mask lets a neighbour
  token attend only to its own centre (checked by zeroing one centre and observing which
  outputs change); `no_program_tokens` has the v2 `ProgramNet` head shape and bypasses the
  attention layers; Gaussian NLL decreases on a planted signal; temperature scaling leaves
  the mean unchanged.
- Synthetic end-to-end: `tests/e2e/test_main_synthetic.py` runs `main.py` with
  `configs/synthetic.yaml` on CPU, then asserts that `CONTRIBUTIONS.md`, `model_card.json`,
  `release_gates.csv`, `cv_results.csv`, `gene_backbone.csv`, `validation_summary.json`,
  `RESULTS.json` and `environment.json` exist, that every gate has a status in
  {PASS, FAIL, NOT EVALUATED}, that the better of the unified and MLP paths beats
  `shuffled` on the planted program with `shuffled` itself at chance, and that a second
  invocation loads every component. A second run with `--force training` recomputes training and everything
  after it and nothing before.
- GPU-optional: tests marked `@pytest.mark.gpu` are skipped when
  `torch.cuda.is_available()` is false. They check bf16 autocast produces finite losses,
  that `torch.compile` with `dynamic=False` does not recompile across epochs (asserted
  through `torch._dynamo.utils.counters`), that channels_last propagates through the
  encoder, that a padded last batch gives the same predictions as an unpadded one, and
  they record throughput into `hardware.json` without asserting a number.
- Regression: `tests/regression/test_v2_parity.py` runs the v2 `ProgramNet` path and the v3
  `no_program_tokens` variant on the synthetic fixture with the same seed and requires
  per-patient median Pearson within 0.15. The reference is the v2 `no_spatial` model (the
  same two-layer MLP and composition head); the v2 `full` model adds a coordinate prior
  g(fourier(x, y), slide embedding) that the v3 MLP path does not have. A second check
  requires the v3 `full` model (neighbour tokens) not to fall behind v2 `full` by more than 0.15.

## 8. Verified facts for training and inference on one H100 80 GB

Each fact below was read on the cited page on 2026-11-03. Where a page did not state a
value, the entry says so; nothing is inferred.

Hardware
- H100 SXM: 80 GB GPU memory, 3.35 TB/s; FP16 and BF16 Tensor Core 1,979 teraFLOPS,
  TF32 989 teraFLOPS, FP8 3,958 teraFLOPS, all footnoted "With sparsity". The page lists
  SXM and NVL (94 GB) only. https://www.nvidia.com/en-us/data-center/h100/

Mixed precision
- `torch.autocast` default dtype is `torch.float16` for CUDA and `torch.bfloat16` for CPU.
  Float16 training "uses torch.autocast and torch.amp.GradScaler together"; for CPU
  bfloat16 the page says it "only uses torch.autocast". The page does not state in words
  that CUDA bfloat16 needs no GradScaler. https://docs.pytorch.org/docs/stable/amp.md
- GradScaler exists because float16 gradients with small magnitude "will flush to zero
  ('underflow'), so the update for the corresponding parameters will be lost"; gradient
  scaling "improves convergence for networks with float16 ... by minimizing gradient
  underflow". https://docs.pytorch.org/docs/2.5/amp.html and
  https://docs.pytorch.org/docs/2.15/notes/amp_examples.html
- CUDA ops that autocast to float16 include conv1d/2d/3d, linear, matmul, mm, bmm, addmm;
  ops that autocast to float32 include cross_entropy, layer_norm, log_softmax, softmax,
  mse_loss, nll_loss, binary_cross_entropy_with_logits. https://docs.pytorch.org/docs/2.5/amp.html
- Decision for v3: bf16 autocast on the H100 with no GradScaler (bf16 keeps the float32
  exponent range, so the underflow the scaler addresses does not arise; this reasoning is
  the author's, the page states only the float16 case). amp_dtype float16 is accepted by
  the config but the training loop refuses it, because no GradScaler is implemented and
  float16 would train unscaled and silently underflow; use bfloat16 on Ampere, Ada or
  Hopper GPUs. The FlashAttention README states "bf16
  requires Ampere, Ada, or Hopper GPUs". https://github.com/Dao-AILab/flash-attention
- NVIDIA mixed-precision guide: "On FP16 inputs, all three dimensions (M, N, K) must be
  multiples of 8" for Tensor Core matmul; "Choose mini-batch to be a multiple of 8"; the
  page does not mention bfloat16 or TF32.
  https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

TF32
- `torch.backends.cuda.matmul.allow_tf32` "defaults to False in PyTorch 1.12 and later";
  `torch.backends.cudnn.allow_tf32` "defaults to True"; the page says the old `allow_tf32`
  settings are going to be deprecated in favour of `fp32_precision` settings (after 2.9)
  and that `set_float32_matmul_precision()` sets matmul precision more broadly.
  https://docs.pytorch.org/docs/2.15/notes/cuda.html

channels_last
- "The channels last memory format is an alternative way of ordering NCHW tensors in
  memory"; "implemented for 4D NCHW Tensors only"; "the most significant performance gains
  are observed on NVIDIA's hardware with Tensor Cores support running on reduced precision
  (torch.float16)"; ResNet-50 with AMP: "observed 22% performance gain"; Conv and Batchnorm
  with cuDNN support it. The page's figures are from Volta, not H100.
  https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html

torch.compile
- Modes: "default" is "a good balance between performance and overhead";
  "reduce-overhead" "reduces the overhead of python with CUDA graphs, useful for small
  batches"; "max-autotune" "leverages Triton or template based matrix multiplications";
  "max-autotune-no-cudagraphs" is "similar to max-autotune but without CUDA graphs".
  `dynamic=None` (default): "we automatically detect if dynamism has occurred and compile a
  more dynamic kernel upon recompile"; `dynamic=False`: "we will NEVER generate dynamic
  kernels, we will always specialize"; `fullgraph=True`: "we require that the entire
  function be capturable into a single graph".
  https://docs.pytorch.org/docs/stable/generated/torch.compile.md
- Dynamic shapes: "By default, PyTorch assumes all input shapes to be static"; the first
  compile assumes static shapes and a recompile marks changed dimensions dynamic;
  `dynamic=True` "forces all sizes and integers to be dynamic, increasing the chance of
  encountering dynamic shape bugs"; "graphs with sizes 0 and 1 will specialize"; "Dynamic
  shapes do not support dynamic rank programs". `mark_dynamic` "must not be called inside
  any function that is being compiled".
  https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- Graph breaks: "Dynamo does not support tracing dynamic control flow"; `fullgraph=True`
  raises on the first graph break; "the observed speedup may vary on factors such as model
  architecture and batch size".
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Decision for v3: fixed batch shapes with padding, `dynamic=False`, mode `default`.

scaled_dot_product_attention
- Three implementations for CUDA inputs: FlashAttention-2 (cited as arXiv 2307.08691),
  Memory-Efficient Attention (xformers), and a C++ math implementation; "All
  implementations are enabled by default" and SDPA "attempts to automatically select the
  most optimal implementation based on the inputs"; "Each of the fused kernels has specific
  input limitations" (the page does not list dtype or head-dimension limits per kernel);
  "An error is thrown if both attn_mask and is_causal are set"; a boolean mask uses True
  for "take part in attention" (the inverse of `key_padding_mask`); when a fused kernel
  cannot run, "a warning will be raised with the reasons"; backend forced with
  `torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])`; GQA via
  `enable_gqa` is experimental; the math backend keeps intermediates in float32 for half
  and bfloat16 inputs.
  https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
  and https://docs.pytorch.org/docs/2.5/generated/torch.nn.attention.sdpa_kernel.html
- FlashAttention-2 README: "Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper
  GPUs)"; "All head dimensions up to 256"; backward for head dim > 192 needs A100/A800 or
  H100/H800; the README documents causal, sliding-window and ALiBi options and does not
  document arbitrary additive masks. FlashAttention-3 is a Hopper beta (FP16/BF16 forward
  and backward, FP8 forward; H100 and CUDA 12.3 or newer).
  https://github.com/Dao-AILab/flash-attention
- FlashAttention-2 venue: the README BibTeX gives "International Conference on Learning
  Representations (ICLR)", 2024; the Princeton publication record lists "12th International
  Conference on Learning Representations, ICLR 2024"; the OpenReview forum page
  (openreview.net/forum?id=mZn2Xyh9Ec) returned a browser-verification page and could not
  be read, so the ICLR status is taken from the README and the Princeton record. The paper
  (openreview.net/pdf/971944d83cc3dae6e8682791d32759cfe8a171bc.pdf) benchmarks head
  dimension "64 or 128" and reports "up to 335 TFLOPs/s" on H100 (forward and backward).
  https://collaborate.princeton.edu/en/publications/flashattention-2-faster-attention-with-better-parallelism-and-wor/
- Consequence for the unified model: the block-diagonal attention mask routes attention
  to the memory-efficient or math kernel; the flash kernel is used only in the mask-free
  single-slide inference path. Whether flash is actually selected must be checked with the
  warning text from `sdpa_kernel`, not assumed.

DistributedDataParallel on one node
- DDP "Implement[s] distributed data parallelism based on torch.distributed at module
  level"; "Creation of this class requires that torch.distributed to be already
  initialized"; for multi-GPU "you should spawn up N processes, ensuring that each process
  exclusively works on a single GPU"; "DDP's performance advantage comes from overlapping
  allreduce collectives with computations during backwards". Neither page describes a
  benefit on a single GPU; with one process there is no allreduce to overlap, so v3 does
  not use DDP (author's conclusion from the quoted mechanism).
  https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html
  and https://docs.pytorch.org/docs/2.15/notes/ddp.html
- torchrun single node: `torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS
  script.py`; it sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT;
  `--nproc-per-node` may be an integer, "gpu", "cpu", "xpu" or "auto".
  https://docs.pytorch.org/docs/stable/elastic/run.md

DataLoader
- `pin_memory=True`: "the data loader will copy Tensors into device/CUDA pinned memory
  before returning them"; "Host to GPU copies are much faster when they originate from
  pinned (page-locked) memory"; `persistent_workers=True`: "the data loader will not shut
  down the worker processes after a dataset has been consumed once" (default False);
  `prefetch_factor`: "Number of batches loaded in advance by each worker", default None
  when `num_workers=0` and 2 otherwise; `num_workers` default 0 loads in the main process.
  Pinned tensors allow asynchronous copies with `non_blocking=True`; "If you overuse pinned
  memory, it can cause serious problems when running low on RAM".
  https://docs.pytorch.org/docs/stable/data.md and https://docs.pytorch.org/docs/stable/notes/cuda.md

CUDA graphs
- "Dynamic shapes are prohibited"; "Ops that synchronize the CPU with the GPU (e.g.,
  .item() calls) are prohibited"; "Dynamic control flow (based on CPU or GPU data) is
  prohibited"; "Capture must occur on a non-default stream"; "Warmup must occur on a side
  stream"; "Every replay reads from and writes to the same (virtual) memory addresses";
  "For typical optimizers, GradScaler.step syncs the CPU with the GPU, which is prohibited
  during capture"; unsafe parts can run eagerly with `make_graphed_callables` graphing the
  safe parts. `make_graphed_callables` defaults to 3 warmup iterations, requires
  `autocast(cache_enabled=False)`, and does not support double backward.
  https://docs.pytorch.org/docs/stable/notes/cuda.md,
  https://docs.pytorch.org/docs/main/generated/torch.cuda.make_graphed_callables.html,
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- Decision for v3: off by default; the neighbourhood-union minibatch has a data-dependent
  size unless padded, and the training loop reads `.item()` for logging. Enabling
  `training.cuda_graphs` requires the padded fixed-size batch path and deferred logging.

Throughput and batch sizes
- timm's results README states its benchmark CSVs were "Currently only run on RTX 3090 w/
  AMP for inference"; no H100 figure for resnet50 or vit_base_patch16_224 is published on
  the timm docs pages opened (resnet, vision-transformer, results README). No official
  page opened states a recommended batch size for a frozen ResNet-50 at 224 px on H100.
  Therefore: measure. `features/embed.py` runs a short sweep (256, 512, 1024, 2048 patches)
  on the first slide, records patches per second and peak memory into `hardware.json`, and
  keeps the largest batch that fits with headroom.
  https://github.com/huggingface/pytorch-image-models/blob/main/results/README.md,
  https://huggingface.co/docs/timm/models/vision-transformer

Memory estimate for the embedding store (arithmetic, not a documentation fact)
- 600,000 cells x 2048 dims x 2 bytes (float16) = 2,457,600,000 bytes = 2.29 GiB (2.46 GB);
  float32 would be 4.92 GB. A 1536-d encoder (H-optimus-0) at float16 is 1.84 GB; a 2560-d
  Virchow2 concatenated embedding is 3.07 GB. The g x g GRID store at g = 4 multiplies these
  by 16, which is why GRID is a float16 memmap on disk and never fully resident on the GPU.
  At the default grid of 7 (the ResNet-50 feature map) the multiplier is 49, giving
  600,000 x 49 x 2048 x 2 bytes = 120 GB for 600,000 cells at float16; the memmap is the
  only reason this fits.

## 9. Encoder registry facts (model cards opened 2026-11-03)

| id in config | hub id and loader | gated | licence | input | dim | paper and venue (as the card states) |
|---|---|---|---|---|---|---|
| resnet50 | timm `resnet50` | no | Apache-2.0 (timm) | 224 | 2048 | n/a |
| uni | `timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)` | yes, terms required, non-commercial academic research | CC-BY-NC-ND-4.0 | 224 | 1024 (ViT-L/16) | Chen et al., "Towards a General-Purpose Foundation Model for Computational Pathology", Nature Medicine 2024, doi 10.1038/s41591-024-02857-3 |
| conch | `conch.open_clip_custom.create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=...)`; the card's usage does not go through timm | yes, institutional email must match | CC-BY-NC-ND-4.0 | not stated on the card | not stated on the card (vision encoder ViT-B/16, 90M params) | Lu et al., "A visual-language foundation model for computational pathology", Nature Medicine 30:863-874, 2024 |
| virchow2 | `timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)` | yes, non-commercial academic research, no diagnostic use | CC-BY-NC-ND-4.0 | 224 x 224 | 1280 per token; class token + mean patch token = 2560 (ViT-H/14, 631M) | Zimmermann et al., "Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology", arXiv 2408.00738, 2024 (no journal venue on the card) |
| hoptimus0 | `timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)`; mean (0.707223, 0.578729, 0.703617), std (0.211883, 0.230117, 0.177517) | yes (agree to share HF information) | Apache-2.0 | 224 x 224 at 0.5 um/px | 1536 (1.1B-parameter ViT) | `@software` citation to a GitHub repository, 2024; no peer-reviewed paper on the card |
| phikon_v2 | `transformers.AutoModel.from_pretrained("owkin/phikon-v2")`, CLS token; not a timm id | not shown as gated | Owkin non-commercial licence ("other") | trained on 224 x 224 at 20x | 1024 (ViT-L/16, DINOv2) | Filiot et al., "Phikon-v2, A large and public feature extractor for biomarker prediction", arXiv 2409.09173, 2024 (no journal venue on the card) |

Sources: https://huggingface.co/MahmoodLab/UNI, https://huggingface.co/MahmoodLab/CONCH,
https://huggingface.co/paige-ai/Virchow2, https://huggingface.co/bioptimus/H-optimus-0,
https://huggingface.co/owkin/phikon-v2; gating status also confirmed through the Hugging
Face Hub API (UNI, CONCH, Virchow2, H-optimus-0 reported gated; phikon-v2 not). The timm
hub-loading page documents the `hf_hub:` spelling; the UNI, Virchow2 and H-optimus-0 cards
use `hf-hub:`; both spellings appear on official pages and the registry accepts either
(https://huggingface.co/docs/timm/hf_hub). The timm hub id for UNI is spelled `MahmoodLab/uni`
in the card's snippet and `MahmoodLab/UNI` in the repository name.

Only `resnet50` and `hoptimus0` carry licences that permit commercial use; the registry
refuses the others when `features.commercial_use` is true. The two cards that cite a
peer-reviewed venue are UNI and CONCH (Nature Medicine 2024); Virchow2 and Phikon-v2 cite
arXiv preprints; H-optimus-0 cites software only.

## 10. Repository layout

```
main.py                         twelve direct component calls, release gates printed last
scripts/release_check.sh        lint, fast tests, synthetic run, RESULTS.json
configs/{default,smoke,synthetic,h100}.yaml   sectioned YAML (run, ingest, ..., validation, deployment, reporting)
resources/                      unchanged from v2
src/mmprog/
  config.py                     flat Config dataclass, sectioned YAML flattened on load, typed --set overrides
  core/context.py               Context (typed run state, ctx.plan, gates), Fold
  core/component.py             BaseComponent: requires / provides / artefacts, manifest hash, execute, run_in
  core/pipeline.py              COMPONENT_ORDER, RunPlan, start_run (environment.json), describe_dry_run,
                                write_run_manifest, run_pipeline
  logging_utils.py paths.py resources.py          reused verbatim
  components/                   one file per component, each ending with the function main.py calls
  data/ preprocessing/ targets/ features/ models/ inference/ analysis/
                                v2 packages, modules reused verbatim or rewritten as listed above
  programs/                     anchored_nmf.py, targets_builder.py
  training/                     data.py, loop.py, cv.py
  evaluation/                   metrics, decisions, calibration, clinical, stage_association (new), manova,
                                transfer, spatial_boost_eval, codex_validation (re-export)
  interpretation/               attribution.py, backbone.py, enrichment.py
  validation/                   perturbations, program_specificity, spatial_fidelity, robustness,
                                independent_cohort, protein_bridge                  (new in v3.1)
  reporting/                    figures, qa_tiers, model_card, contributions, gates (new in v3.1)
tests/unit/ tests/e2e/ tests/regression/
docs/ARCHITECTURE_V3.md (this file), DESIGN_V3.md, DECISIONS.md (D28 to D36 for v3 and v3.1),
docs/RESULTS_SPEC.md, docs/VALIDATION_ROADMAP.md, docs/research/VALIDATION_METHODS.md
```
