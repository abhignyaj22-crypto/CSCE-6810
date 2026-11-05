# v3 design brief: the unified model and its verified components

Companion to `ARCHITECTURE_V3.md` (software contract) and the two scouting reports in
`docs/research/` (every citation below was verified there by opening the venue page;
anything the scouts marked UNVERIFIED is not used). This file states, per component,
which published algorithm is adopted, what is adapted, and what is this repository's
own choice. Agents implementing a component must not describe a method as published
when the table says "adaptation" or "own choice".

## 0. The research question the model answers

Per cell in a bone marrow trephine, infer the coordinated gene programs associated with
drug resistance (non-canonical Wnt and adhesion, plasma-cell stress and survival, T-cell
exhaustion and regulation, cytotoxicity as control) from nuclear morphology and its
spatial context, and show that the morphology carries information beyond cell
composition and position. Two outputs: continuous program scores per cell with
calibrated uncertainty, and the gene backbone of each program (the genes whose weights
survive patient-level stability selection) with its pathway annotation.

## 1. Component to algorithm map

| component | adopted (verified venue) | adaptation or own choice | fallback |
|---|---|---|---|
| Preprocess | v2 QC, normalisation, summaries (D6 to D8); classical watershed nuclei (D21) | missing-gene mask across samples (own choice) | none needed |
| ProgramInference | seeded non-negative factorisation in the style of Spectra (Kunes et al., Nature Biotechnology 2024, biology venue; the scouts found no ICML/ICLR/NeurIPS 2023-2026 method that factorises observational expression with user gene-set seeds) | anchored NMF implemented here with multiplicative updates and an L1 penalty on non-anchor weights of seeded programs (own implementation, not Spectra's code); unseeded extra programs are plain NMF components that absorb structure the literature sets do not describe; the consensus-over-restarts step of cNMF (Kotliar et al., eLife 2019) is not implemented | fixed-set scores (v2) kept as the `anchor_only` target family |
| Gene backbone | stability selection (Meinshausen and Buhlmann, JRSS-B 2010) with patient-level subsampling | threshold and fold fraction are own choices (config) | model-X knockoffs are documented but not run at n = 21 |
| Pathway annotation | hypergeometric enrichment with the 5,001-gene panel as background (standard; GSEA, PNAS 2005, is the reference for the ranked variant) | gene sets limited to the curated resistance sets plus the Wnt core and adhesion subsets | none |
| Morphology encoder | frozen pathology or ImageNet encoder compared under this repository's patient-held-out protocol, by analogy with the HEST-Benchmark (Jaume et al., NeurIPS 2024 Datasets and Benchmarks); the HEST ridge-on-PCA baseline itself is not run | registry: resnet50 default; uni, virchow2, hoptimus0, phikon_v2 through timm or transformers with licence check (own choice) | resnet50 |
| Morphology tokens | grid of block embeddings over the patch (own choice, serves both the transformer and the boosting map) | PANTHER / MMP prototype tokens (CVPR 2024 / ICML 2024) are slide-level; here tokens are blocks, not prototypes | centre token only (`no_grid`) |
| Spatial context | masked self-attention over the k nearest neighbours with relative-offset Fourier features; FlashAttention-2 (Dao, ICLR 2024) through `F.scaled_dot_product_attention` where no mask is supplied (the program cross-attention and the no_neighbours variant; the neighbour block mask is present in training and inference otherwise) | Point Transformer V3 (CVPR 2024) and Exphormer (ICML 2023) motivate sparse local attention; their serialisation and expander edges are not needed at k <= 16 and are not implemented | `no_neighbours` |
| Program tokens and cross-attention | program tokens as queries over morphology and neighbour tokens, in the spirit of the pathway tokens of SurvPath (Jaume et al., CVPR 2024) and MMP (Song et al., ICML 2024), both verified at abstract level; the attention direction is this repository's choice | tokens are initialised from the learned program gene-weight vectors through a gene-to-token linear map (own choice) | `no_program_tokens` reproduces the v2 MLP |
| Morphology to expression alignment | symmetric InfoNCE between the pooled morphology representation and the per-cell program-space vector, as in TANGLE (Jaume et al., CVPR 2024, slide level) | applied at cell level with the expression side being the P-dimensional program scores (adaptation); weight `lambda_align` | off at `lambda_align = 0` |
| Heads and uncertainty | heteroscedastic Gaussian negative log-likelihood head (standard; natural-parameter variant Immer et al., NeurIPS 2023 is the reference) with temperature scaling on the inner validation patient; split conformal intervals on normalised residuals, calibrated on the cells of one held-out patient (the inner validation patient) and applied to other patients, so the guarantee across patients is approximate and coverage is reported per patient, in the spirit of Conformal Risk Control (Angelopoulos et al., ICLR 2024) and Kandinsky conformal prediction (Bairaktari, Wu, Wu, ICML 2025) | the patient-clustered guarantee is approximate and is stated as such; coverage is reported per patient | plain MSE head (`uncertainty: none`) |
| Composition head and smoothness | multi-task lineage cross-entropy and Laplacian smoothness (v2, D15) | unchanged | `no_comp`, `no_smooth` |
| Training on H100 | bf16 autocast without GradScaler, TF32 enabled, optional torch.compile default mode with static shapes, pinned persistent DataLoader (facts in ARCHITECTURE_V3.md section 8) | single process, single GPU (no DDP benefit documented for one GPU) | CPU path for tests |
| Interpretation | spatially regularized boosting (Xiang et al., NIPS 2009, Advances in Neural Information Processing Systems 22) on grid blocks with lambda = 0 control (D25); cross-attention mass per program token; integrated gradients from each head to program token inputs | attention mass as attribution is descriptive, not causal, and is labelled so | none |
| Evaluation | v2 metrics, reliability ceiling, decision rules (D9, D19), calibration coverage | `anchor_only_targets` baseline tests whether learned programs are more predictable or merely different | none |
| Stage association (Evaluation, D34) | PERMANOVA (Anderson, Austral Ecology 2001; classical statistics, not venue-restricted) on patient-level z-scored burden vectors, Euclidean distance, whole-patient label permutation; Jonckheere-Terpstra trend test (Jonckheere, Biometrika 1954; Terpstra 1952 not re-verified) per program along control < MGUS < SMM < MM; Benjamini-Hochberg q-values (JRSS-B 1995) | both implemented here in numpy with permutation p-values (own implementation of published tests); the control / precursor / myeloma collapse when a stage has fewer than two patients is an own choice; PERMDISP is not implemented and the dispersion caveat is printed | classical MANOVA with Pillai's trace (D26) under `stage_test: manova` or `both` |
| External validation: program specificity (D35) | control gene sets matched to the program gene by gene in expression bins, the construction of Tirosh et al. (Science 2016; verified on PubMed, DOI 10.1126/science.aad0501), used by Seurat's AddModuleScore | applied as a null for a morphology prediction rather than as a score baseline (adaptation); the linear proxy for NMF usages, the 25 bins, the 1 % and 90 % weight-mass gene rule, the split-gene halves and the noise floor on the null spread are own choices | none; always runs |
| External validation: spatial fidelity (D35) | Moran's I (classical) on the model's kNN graph; Cohen's d niche contrast (classical) | plasma-dense flag (at least five plasma cells within nb_radius_um) and the predicted-versus-measured comparison are own choices | none; always runs |
| External validation: perturbation robustness (D35) | none adopted; gamma, Gaussian blur, quarter-turn rotation, additive noise and intensity scaling on fixed detected nuclei with Spearman against the clean prediction | own choice (engineering); a proxy for scanner shift, not a multi-site study, and labelled so | skipped without a checkpoint or image |
| External validation: independent cohort (D35) | the training ingest path (`ingest_one_sample`) and the main decision rule on a new cohort | learned programs projected onto the cohort's genes as a weight-normalised mean (own choice) | data-gated on `validation_cohort_manifest` |
| External validation: protein bridge (D27, D35) | CODEX bridge unchanged: composition agreement in centred log-ratio space, plasma-dense niche contrast | moved from deployment; no public myeloma CODEX pairs with Xenium | data-gated on `codex_table` and `codex_dapi` |
| Deployment | v2 inference path with the unified model, QuPath and Xenium Explorer exports (D22) | DAPI and H&E both accepted; H&E through the haematoxylin channel (D2); the inference parity number (D36) is an own check | none |
| Reporting and release gates (D36) | none adopted; nine pass / fail / not-evaluated checks read from artefacts (`RESULTS.json`) | own choice (engineering discipline); thresholds are the decision-rule margins already in D19 plus calibration, backbone, parity, stage and validation bars stated in `docs/RESULTS_SPEC.md` | none |

## 2. Decisions new in v3 (DECISIONS.md D28 to D32; v3.1 adds D33 to D36)

D28. Programs are learned, with the literature sets as anchors, and the fixed-set scores
are kept as a baseline target family. The factorisation is unsupervised and uses no
morphology and no labels; it is fitted once on a random subsample of at most
programs_fit_cells QC-passing cells drawn from every patient, and all cells are then
projected onto the fixed gene weights, so every patient's targets are on one scale. This
makes the target definition transductive: unlabelled test-patient expression enters the
gene weights (no morphology and no labels do), which is a weaker dependence than
supervised leakage but is not the same as using a gene set fixed before the study; it is
stated in the report. Stability selection refits on patient subsamples define the backbone.
A second stated limitation: the learned neighbourhood-scale targets average the cell-scale
usage over the cells kept for training (after `max_cells_per_sample` subsampling), while
the fixed-set neighbourhood targets were summed at ingest over every QC-passing cell, so
the two families measure neighbourhoods of different cell density; the fix path is to
project all cells onto the fixed gene weights at ingest, as the count summaries are.

D29. One network, tokens for three sources (patch blocks, neighbour cells, programs),
cross-attention from programs to morphology. Every ablation is a configuration switch,
and `no_program_tokens` skips every token and attention layer and feeds the raw centre
feature vector (PCA embedding plus morphometrics, the v2 ProgramNet input) into a
v2-shaped two-layer MLP head, so the variant measures what the token architecture adds
over the v2 head on identical features; the v2 ProgramNet without its coordinate prior
(v2 `no_spatial`) is the regression reference (tests/regression), agreeing within 0.15
per-patient median Pearson on the synthetic fixture, and the v3 full model is required not to
fall behind the v2 full model (coordinate prior) by more than the same margin.

D30. Uncertainty is heteroscedastic and calibrated on a held-out patient; conformal
intervals are reported with per-patient coverage because cells within a patient are not
exchangeable with cells of another.

D31. Alignment to expression is an auxiliary contrastive loss at cell level, adapted from
a slide-level method; it is reported as an ablation and dropped if it does not help.

D32. Single-GPU H100 settings are configuration, verified against official documentation;
throughput is measured and recorded, never assumed.

D33 to D36 (v3.1, written in DECISIONS.md): `main.py` calls the twelve components directly
and the run plan lives on `ctx.plan`; PERMANOVA and the Jonckheere-Terpstra trend test
replace MANOVA as the default stage test; external validation is a modality-agnostic
component with the CODEX bridge as one data-gated validator; release gates are read from
artefacts and a gate that cannot be evaluated is reported as such, never passed.

## 3. Contract additions beyond ARCHITECTURE_V3.md

- Config is the flat `Config` dataclass; YAML may be grouped by section (flattened on
  load). Each component lists its `config_fields` for the skip hash.
- `Context` fields are in `src/mmprog/core/context.py`; components set them with
  `ctx.set(name, value)` and must set every field in `provides`.
- Targets: `prog_<p>_<scale>_z` for learned programs (p in the seeded names plus
  `extra_<i>`), `anchor_<p>_<scale>_z` for the fixed sets; `ctx.target_family` maps each
  target to its family. The main target is `prog_wnt_camdr_nb_z`.
- `ctx.expression_embedding` is the (N, P) learned program-score matrix (cell scale,
  standardised), used by the alignment loss.
- Every component prints: inputs (shapes), decisions taken with their reason, outputs
  (shapes, file names), and timings.
- Each component module ends with the function `main.py` calls (`ingest`, `preprocess`,
  `infer_gene_programs`, `extract_features`, `select_features`, `build_unified_model`,
  `train_unified_model`, `evaluate`, `interpret_programs`, `validate_externally`, `deploy`,
  `report`); the function is `<Class>().run_in(ctx)` and nothing else.
- `RESULTS.json` at the run root and `reporting/release_gates.csv` hold the release gates;
  `environment.json` the reproducibility record.
