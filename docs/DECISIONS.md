# Design decisions and their scientific basis

Each entry states the decision, the reason, and the evidence. "Pilot" refers to the
notebook runs on four GSE299193 trephines that preceded this repository.

## Data

**D1. Train on Xenium 5K trephines (GSE299193, Yip et al., Blood 2025).**
This is the only public dataset with matched whole-slide nuclear morphology and
single-cell expression in myeloma bone marrow (21 trephines: 4 control, 2 MGUS,
5 smouldering, 10 newly diagnosed). Yip et al. report spatially restricted plasma
cell subpopulations in 5 of 10 myeloma trephines and conclude that no universal
marrow niche supports myeloma expansion; they do not analyse Wnt signalling,
drug resistance or survival. The plasma-cell-dense niche enriched for
non-canonical Wnt and adhesion-related expression is reported by Ohlstrom et al.
(2026, bioRxiv preprint), who use the Yip et al. Xenium data for single-cell
validation of Visium HD findings. Progression-free survival in that preprint is
a separate 15-gene signature, which this repository does not score. The
tissue is the routine diagnostic trephine, so a model trained here targets the
specimen every myeloma patient already has.
References: Yip et al. 2025, Blood 146(15):1837. https://doi.org/10.1182/blood.2025028896;
Ohlstrom et al. 2026, bioRxiv. https://doi.org/10.64898/2026.03.23.713195 (preprint).

**D2. The morphology channel is DAPI, not H&E.** GEO ships the Xenium
`morphology.ome.tif`, which is a DAPI z-stack. The pipeline therefore learns from
nuclear morphology only. H&E transfer is a later phase that requires either
stain harmonisation (`stain_mode: hematoxylin`, which maps H&E to its nuclear
channel) or an H&E cohort. This is stated in every output rather than hidden.

**D3. Maximum-intensity projection over the z-stack.** The 13 planes are
0.75 um apart; a nucleus spans several. Max projection keeps every nucleus in
focus in one 2D image at the cost of some blurring. "mid" is available for
comparison. The pilot bug of treating (13, H, W) as a 2D array produced invalid
embeddings; cache keys include the projection mode so stale embeddings are never
reused.

**D4. Pyramid level 1 (0.425 um/px), 64 px patches (27 um).** One plasma cell is
about 10 to 15 um; the patch holds the cell and its immediate neighbours, which
is the scale at which adhesion-mediated signalling operates. Level 0 doubles
memory for no additional biological context at this patch size.

**D5. Relapse sample excluded; panels below 1000 genes excluded.** RM-1 was run
on a 377-gene panel in the GEO series; its program coverage is incomplete and
its stage is not on the control to MM axis. The exclusion is a config regex and
a gene-count rule, both printed when applied.

**D6. Subsample to 30,000 cells per slide for embeddings, but compute density and
neighbourhood sums on all QC-passing cells first.** The pilot computed density
on the subsample, making a 78,000-cell biopsy look 2.6 times sparser than it was.
Order of operations is now fixed in `s1_ingest` and tested.

## Targets

**D7. Count-based program scores at two scales: single cell and 30 um
neighbourhood.** With 80 to 130 transcripts per cell, a per-cell score from 18
genes is close to Poisson noise (the pilot maps were salt-and-pepper and the
model collapsed to the mean). Summing counts within a 30 um radius (about three
cell diameters; the preprint defines plasma-cell-predominant neighbourhoods on
16 um Visium HD bins and the ledger records no radius, so 30 um is this
repository's choice) raises the signal-to-noise ratio. In the terms of SpatialCOC
(Li et al. 2026) this is a kernel-smoothed target with a uniform disc kernel; a
Gaussian kernel is not used because the split-half reliability estimate (D9)
requires integer counts. Both scales are kept so
the gain is measured, not assumed.

**D8. Standardise targets across the cohort, not within slide.** Within-slide
z-scoring removes every between-patient difference, which makes a disease-stage
endpoint meaningless by construction. Cohort standardisation keeps the
between-patient component. The slide option remains for comparison and prints a
warning when it is used with the stage endpoint.

**D9. Measure the noise ceiling of every target by split-half reliability.** Each
target is built from summed transcript counts, so it can be split into two
independent halves by binomial thinning. The Spearman-Brown corrected
correlation between the halves is the agreement an independent re-measurement
would reach; its square root bounds the correlation any image model can reach.
Every reported correlation is accompanied by this ceiling and by the
reliability-adjusted value. This is the standard treatment of measurement error
in psychometrics and in neural-response prediction; without it, a low
correlation on a noisy target is uninterpretable.
Validated in `tests/test_reliability.py`: a null target gives r_sb near 0, a
planted structure gives high r_sb.

**D10. Lineage labels from marker scores (argmax of expression-matched marker
scores).** Used for the composition head and the composition-only baseline. The
marker lists are the Xenium 5K panel markers used for cell typing in the source
paper. The argmax rule is simple and reproducible; a clustering-based annotation
would add a non-deterministic step between runs.

## Features

**D11. Frozen ImageNet ResNet-50 embeddings, no fine-tuning.** The cohort has 21
patients. Fine-tuning a 25M-parameter encoder on 21 patients invites memorising
patient identity. A frozen encoder plus a small head is the textbook setting
where the number of trainable parameters is matched to the data. Pathology
foundation encoders (UNI, CONCH) are supported behind a flag as a later phase;
the comparison `oracle - full` before and after the swap tells whether identity
was the bottleneck.

**D12. PCA to 128 dimensions, fit inside each cross-validation fold.** The 2048-d
head overfit four training patients in the pilot. PCA fit on the fold's training
cells only, so no test-cell statistic leaks into the feature space.

**D13. Morphometric branch (patch intensity statistics and cell density at three
radii).** These are the quantities a haematopathologist reads: cellularity and
nuclear staining. They are computed from centroids and intensities only, so the
inference stage can compute them from detected nuclei with no transcriptomics.

**D14. Feature selection drops only dead and redundant features; everything else
is reported, not acted on.** Selection that uses the target on training cells and
then evaluates on test cells of the same fold is valid, but selection that is
tuned across folds, or by eye, leaks. The selector therefore (1) drops embedding
dimensions with near-zero variance (dead channels), (2) drops one of each pair
of morphometrics correlated above 0.98 (redundant), and (3) reports, without
acting, the univariate Spearman screen, the PCA retained-variance table and a
ridge permutation importance on the fold's training cells. The report is written
to `feature_selection.json` for interpretation; the model never sees a feature
chosen because it correlated with the target.

## Model

**D15. Two-layer MLP head (256 units, GELU, dropout 0.3) with a composition head
and a Laplacian smoothness term.** The composition head is a multi-task auxiliary loss (the features must also
explain cell identity), which regularises the program head; it is only loosely
analogous to SpatialCOC's cross-omics correction, whose CCA-based transfer is
deferred (METHODS_NOTES.md, Q3.5). The Laplacian term over six spatial
neighbours encodes the prior that program scores vary smoothly at the niche
scale; it is computed on the union of the minibatch and its neighbours so
minibatching does not break it. Both are ablated (`no_comp`, `no_spatial`).
In v3 the Laplacian term is prediction to prediction as well: each training step writes its detached predictions into a cache and the next batches read their neighbours' predictions from it, so the term costs no extra forward pass and no neighbour target enters it (training neighbours only).

**D16. Shared spatial prior: random Fourier features of slide-normalised
coordinates plus a 16-d learned slide embedding.** One network per slide (as in
SIREN-style approaches) scales parameters with the number of slides and has no
definition on an unseen slide. The shared prior has constant parameter count,
and setting the slide embedding to zero defines inference on a new slide.

**D17. Early stopping on an inner validation patient.** The pilot showed more
training steps hurt held-out performance. The outer fold's training patients
are split again; one patient is the early-stopping set. Test patients are never
seen during training or model selection.

## Evaluation

**D18. Patient-level GroupKFold, seven folds.** One trephine per patient, so
patient-level and slide-level cross-validation coincide; seven folds give three
test patients per fold and eighteen training patients. Four patients in the pilot
gave two training patients per fold, which is not a learning problem.

**D19. Baselines and ablations are mandatory, and the decision rules are
numeric.** `shuffled` (within-slide label permutation) is the chance bound.
`composition_only` (ridge on one-hot lineage) is what a cell-type detector would
achieve. `oracle` (features plus true lineage) bounds what the features could do
with perfect identity. The four decision rules in `evaluation/decisions.py`
compare these with a fixed margin and write plain sentences into
`CONTRIBUTIONS.md`. A result that does not beat `composition_only` is reported
as a cell-type detector, not as program prediction.

**D20. Per-patient median Pearson with patient bootstrap 95% CI, plus
multi-seed.** Pooled correlation across patients is inflated by between-patient
offsets; the per-patient median is the within-patient statistic that matters
for a spatial map. The bootstrap over patients is the right unit of replication.
Three seeds report training variance.

## Inference and deployment

**D21. Classical nucleus detection (background subtraction, local threshold,
watershed), tiled.** The deployed model must run on a slide with no
segmentation model and no transcriptomics. Tiling keeps a 200-megapixel core
within Colab memory. Polarity (bright nuclei on dark for DAPI; dark on light for
H&E) is decided by the image median and printed.

**D22. Exports for QuPath and Xenium Explorer.** Pathologist review happens in
those tools; a CSV the reviewer cannot open is not a deliverable.

## Reporting discipline

**D23. Every number in a report comes from a file in the run directory.**
`cv_results.csv`, `cv_results_reliability_adjusted.csv`, `cv_multiseed.csv`,
`target_reliability.csv`, `program_burden.csv`, `model_card.json`,
`CONTRIBUTIONS.md`. Slides are built from these files, never from a screenshot.

**D24. The gene sets are investigator-defined, scored as abundances, and the Wnt
members are also scored on their own.** The 18-gene set mixes Wnt ligands,
receptors and secreted antagonists (which have opposing effects on pathway
output) with eight adhesion, survival and cytokine genes that are not Wnt
components. It is therefore reported as a gene-set abundance associated with drug
resistance in the literature, never as Wnt pathway activity. `wnt_core` (ten Wnt
members) and `adhesion_survival` (eight non-Wnt members) are scored separately so
the morphological signal can be attributed to one or the other. Per-gene
evidence grades are printed at S0 from `resources/gene_annotations.json`; three
members (WNT10A, NCAM1, BCL2) carry evidence pointing toward drug sensitivity or
better outcome, and this is stated in CONTRIBUTIONS.md (section 9) of every run. Full audit:
`docs/research/PATHWAY_INTERPRETATION.md`.

**D25. Spatially regularized boosting answers "where in the patch", not "how well".**
Stage S4b fits AdaBoost over decision stumps on a grid of patch blocks with the
spatial regulariser of Xiang et al. (2009), whose selection bonus gamma = -2 lambda
K beta favours blocks near already important ones. The lambda = 0 column is the
control; coherence and accuracy are both reported because the original paper found
the regulariser can cost accuracy. The ring summary (centre, inner ring, outer
ring) is the interpretable output.

**D26. MANOVA at the patient level only, Pillai's trace, with a permutation
alternative.** Cells are nested in patients, so cell-level MANOVA is
pseudo-replication. Twenty-one patients in three collapsed groups (control 4,
precursor 7, myeloma 10) is at the lower limit for a five-variable MANOVA
(wnt_core, adhesion_survival, pc_stress_survival, t_exhaustion, cytotoxicity; the
composite wnt_camdr is omitted because it is a function of its two subsets);
Pillai's trace is the statistic most robust to unequal
groups, and the patient-level permutation test does not assume multivariate
normality. Results are supporting evidence, not the primary endpoint
(`docs/research/METHODS_NOTES.md`, Q2).

**D27. The CODEX bridge is data-gated and uses the DAPI channel.** No public dataset
pairs CODEX with spatial transcriptomics on a myeloma trephine. The stage runs the
DAPI-trained model on a CODEX section's DAPI channel and validates composition and
plasma-dense niche contrasts against protein markers; the public CODEX atlas of
Bandyopadhyay et al. 2024 (healthy femoral-head marrow plus a few AML and
lymphoma-staging iliac crest trephines) is the first available test
(`docs/research/METHODS_NOTES.md`, Q1).

**D28. Gene programs are learned by anchored non-negative factorisation, with the literature sets as anchors, and the fixed-set scores are kept as a baseline target family.** The seeded factorisation is written here from scratch in the spirit of Spectra (Kunes et al., Nature Biotechnology 2024), which the scouts found to be the only published method that factorises observational expression with user gene sets as priors; no ICML, ICLR or NeurIPS 2023 to 2026 main-conference method does this (ALGORITHM_SCOUT_PROGRAMS.md, A.3). Unseeded extra programs absorb structure the literature sets do not describe; they are plain NMF components and the consensus step of cNMF (Kotliar et al., eLife 2019) is not implemented. The factorisation is unsupervised, uses no morphology and no labels, and is fitted once on a random subsample of at most programs_fit_cells QC-passing cells drawn from every patient; all cells are then projected onto the fixed gene weights so every patient's targets are on one scale. This makes the target definition transductive (unlabelled test-patient expression enters the gene weights), which is stated in CONTRIBUTIONS.md. The learned neighbourhood-scale targets average the cell-scale usage over the cells kept for training after subsampling, while the fixed-set neighbourhood targets were summed at ingest over every QC-passing cell; the two families therefore measure neighbourhoods of different cell density, which is stated in CONTRIBUTIONS.md, and the fix path is to project all cells onto the fixed gene weights at ingest. The v2 fixed-set scores remain as the anchor_only target family so the gain from learned weights is measured against them. The gene backbone of each program is the set of genes whose weight survives stability selection (Meinshausen and Buhlmann, Journal of the Royal Statistical Society B 2010;72(4):417-473) over refits on random half-patient subsamples; patients, not cells, are the resampling unit. Pathway annotation is a hypergeometric over-representation test against the curated sets with the measured panel as background (GSEA, Subramanian et al., PNAS 2005, is the reference for the ranked variant).
References: Kunes RZ et al. 2024, Nat Biotechnol 42(7):1084-1095. https://doi.org/10.1038/s41587-023-01940-3; Kotliar D et al. 2019, eLife 8:e43803. https://doi.org/10.7554/eLife.43803; Meinshausen N, Buhlmann P 2010, J R Stat Soc B 72(4):417-473; Subramanian A et al. 2005, PNAS 102(43):15545-15550. https://doi.org/10.1073/pnas.0506580102.

**D29. One network with tokens from three sources (patch blocks, neighbour cells, programs) and cross-attention from program tokens to morphology; every ablation is a configuration switch.** Morphology tokens are a grid of block embeddings over the patch plus a centre token (own choice; the prototype tokens of PANTHER and MMP, CVPR 2024 and ICML 2024, are slide level). Neighbour tokens are the k nearest cells with random Fourier features of the relative offset (D16); the self-attention mask lets a neighbour attend only to itself and the centre token. Program tokens are learned queries initialised from the program gene weights through a gene-to-token map and cross-attend to the morphology and neighbour tokens, in the spirit of the pathway tokens of SurvPath (Jaume et al., CVPR 2024) and MMP (Song et al., ICML 2024); the attention direction is this repository's choice. Sparse local attention follows the motivation of Point Transformer V3 (CVPR 2024) and Exphormer (ICML 2023), whose serialisation and expander edges are not needed at k <= 16 and are not implemented. Attention runs through scaled_dot_product_attention; FlashAttention-2 (Dao, ICLR 2024) is eligible only where no mask is supplied (the program cross-attention and the no_neighbours variant), and its selection is checked, not assumed. The variants no_neighbours, no_grid, no_program_tokens, no_comp, no_smooth and align are switches on one network; no_program_tokens skips every token and attention layer and feeds the raw centre feature vector (the v2 ProgramNet input) into a v2-shaped two-layer MLP head, so it measures what the token architecture adds on identical features; the v2 ProgramNet without its coordinate prior (v2 no_spatial) is the strict regression reference for that path, and the v3 full model is required not to fall behind the v2 full model with its coordinate prior (tests/regression).
References: Song AH et al. 2024, CVPR, pp. 11566-11578 (PANTHER); Song AH et al. 2024, ICML, PMLR 235:46050-46073 (MMP); Jaume G et al. 2024, CVPR, pp. 11579-11590 (SurvPath); Wu X et al. 2024, CVPR, pp. 4840-4851 (Point Transformer V3); Shirzad H et al. 2023, ICML, PMLR 202:31613-31632 (Exphormer); Dao T 2024, ICLR (FlashAttention-2).

**D30. Uncertainty is heteroscedastic, calibrated on a held-out patient, and conformal coverage is reported per patient.** Each head predicts a mean and a log-variance trained with the Gaussian negative log-likelihood (standard parameterisation; the natural-parameter variant of Immer et al., NeurIPS 2023, is the reference). A scalar temperature on the predictive standard deviation is chosen on the inner validation patient and leaves the mean unchanged. Split conformal intervals use the quantile of normalised residuals on that patient's cells, in the spirit of Conformal Risk Control (Angelopoulos et al., ICLR 2024) and Kandinsky conformal prediction (Bairaktari, Wu and Wu, ICML 2025); because cells within a patient are not exchangeable with cells of another, the guarantee across patients is approximate and is stated as such, and coverage at 50 % and 90 % is reported pooled and per patient (calibration.csv).
References: Immer A et al. 2023, NeurIPS; Angelopoulos A et al. 2024, ICLR; Bairaktari K, Wu J, Wu S 2025, ICML, PMLR 267:2581-2602.

**D31. Alignment to expression is an auxiliary contrastive loss at cell level, adapted from a slide-level method, and is reported as an ablation.** A symmetric InfoNCE between the pooled morphology representation and the standardised learned program-score vector of the same cell adapts the slide-level contrastive alignment of TANGLE (Jaume et al., CVPR 2024), where the expression side was a bulk profile; here it is the P-dimensional program vector and the adaptation is this repository's. The weight lambda_align is configuration, the no_align variant is run alongside the others, and the term is dropped if it does not help on the main target.
References: Jaume G et al. 2024, CVPR, pp. 9632-9644 (TANGLE).

**D32. Single-GPU H100 settings are configuration, verified against official documentation, and throughput is measured, never assumed.** bf16 autocast without a GradScaler (bf16 keeps the float32 exponent range; the PyTorch amp page states only the float16 case), TF32 enabled for matmul and cuDNN (the matmul flag defaults to False), optional torch.compile in mode default with dynamic=False on padded fixed-shape batches, pinned persistent DataLoader workers with prefetch_factor set only when num_workers > 0, CUDA graphs off by default, and no DistributedDataParallel (one process, one GPU, nothing to overlap). The device, flags actually used, peak memory and cells per second are written to hardware.json; no batch size or throughput figure is taken from a vendor page (ARCHITECTURE_V3.md section 8).
References: ARCHITECTURE_V3.md section 8 (PyTorch amp, cuda, torch.compile, SDPA, DDP and DataLoader pages opened 2026-11-03).

**D33. `main.py` calls the twelve components directly, and the run plan lives on the context.** A generic driver that loops over a component list hides the pipeline behind the loop: a reader cannot see from the entry point what runs, in what order, or what each step consumes. `main()` is therefore twelve one-line calls (`ctx = ingest(ctx)` through `ctx = report(ctx)`), each with a plain-language comment, the way a program in a software company is written, and each function is defined at the bottom of its own component module so "go to definition" lands on the component. The skip rule, the manifest hash and the artefact contract stay on the component class; the function body is `<Class>().run_in(ctx)`. The `--only` / `--force` plan is stored on `ctx.plan` by `start_run`, so the calls need no second argument and a context built by hand (a notebook, a unit test) runs every component normally. `--dry-run` prints what each component would do without creating a run, `environment.json` records interpreter, library and CUDA versions, GPU, git commit, command line and resolved configuration, and `scripts/release_check.sh` runs lint, the fast tests and the synthetic pipeline and prints the gates. Nothing in the science changed.

**D34. PERMANOVA and a Jonckheere-Terpstra trend test replace MANOVA as the default stage test.** MANOVA was listed as a reference, not a requirement. On this cohort it is a poor fit: 21 patients in four stage groups (control 4, MGUS 2, SMM 5, MM 10), seven program variables, multivariate normality and a common covariance assumed, and the classical requirement of more patients than variables plus groups, which the synthetic run already failed (the test skipped itself as degenerate). Two tests ask the same two questions without those requirements. PERMANOVA (Anderson 2001) computes a pseudo-F from the Euclidean distance matrix of z-scored patient-level burden vectors and its null distribution by permuting stage labels across whole patients; it is defined with more variables than patients and assumes only exchangeability under the null; R squared is reported as the effect size. The Jonckheere-Terpstra test asks, per program, whether burden rises monotonically along control < MGUS < SMM < MM, with a two-sided permutation p-value and Benjamini-Hochberg q-values across programs; it is the ordered contrast METHODS_NOTES.md section 2.1 recommends over a three-degree-of-freedom omnibus difference. Both run on the predicted and on the measured patient vectors, so the reader sees whether the morphology model reproduces whatever stage structure the measured programs have. When any stage has fewer than two patients, stages collapse to control / precursor / myeloma, as the MANOVA did. The patient is the unit of replication throughout (METHODS_NOTES.md section 2.2 on pseudo-replication). PERMDISP is not implemented; the dispersion caveat is printed with the result. The classical MANOVA remains available under `evaluation.stage_test: manova` or `both`. With 21 patients both tests are underpowered, so a non-significant result reads "not detected", not "absent", and the stage association is a plausibility check, not a clinical endpoint.
References: Anderson MJ 2001, Austral Ecology 26(1):32-46. https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x; Jonckheere AR 1954, Biometrika 41(1-2):133-145. https://doi.org/10.1093/biomet/41.1-2.133; Terpstra TJ 1952, Indagationes Mathematicae 14:327-333 (citation not re-verified); Benjamini Y, Hochberg Y 1995, J R Stat Soc B 57(1):289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x.

**D35. External validation is a modality-agnostic component; the CODEX bridge is one data-gated validator among five.** The CODEX bridge (D27) was the only orthogonal check, and no public dataset pairs CODEX with Xenium on a myeloma trephine, so as the sole external check it was a gate that never opened. The `external_validation` component runs every validator the available data allow and records for each one whether it ran, was skipped (with the reason) or failed (with the exception), so a run never silently loses a check and a skipped validator is reported as a gap in the evidence. Two validators always run on the training cohort with out-of-fold predictions: program specificity, which compares the prediction's correlation with the real program against its correlation with random gene sets matched to the program gene by gene in expression level (the control-set construction of Tirosh et al. 2016 used by Seurat's AddModuleScore, applied here as a null for a morphology prediction; an adaptation) and against two random halves of the program's genes, so a prediction that only recovers transcript abundance, density or lineage is named as such; and spatial fidelity, which compares Moran's I and the plasma-cell-dense niche contrast of the predicted and the measured map. A third runs whenever a checkpoint and a slide image exist: perturbation robustness under gamma, blur, quarter-turn rotation, additive noise and intensity scaling on fixed detected nuclei, a proxy for scanner shift and labelled so, not a multi-site study. Two are data-gated: an independent cohort (Xenium or Visium HD manifest; the Visium HD cohort of Ohlstrom et al. 2026, preprint, is embargoed as of writing) scored by the final model from its images and compared with the cohort's own anchor scores and projected learned programs; and the protein bridge, the earlier CODEX code unchanged. Pass bars for each validator are stated in `docs/research/VALIDATION_METHODS.md`; each validator that ran judges itself against its bar and writes `passed` and `pass_bar` into the summary, the release gate requires every validator that ran to have passed, and the gate's value lists each validator's outcome, so a weak validator result both fails the gate and stays visible by name. The bridge's own `aborted` and `insufficient_matches` outcomes are recorded as `skipped` with the reason, so no status falls outside ran, skipped and failed.
References: Tirosh I et al. 2016, Science 352(6282):189-196. https://doi.org/10.1126/science.aad0501 (verified on PubMed; the abstract does not describe the control-set construction, which is taken from the method as implemented in Seurat); Ohlstrom DJ et al. 2026, bioRxiv (preprint). https://doi.org/10.64898/2026.03.23.713195.

**D36. Release gates are read from artefacts, and a gate that cannot be evaluated is reported as such, never passed.** A run ends with nine checks (`reporting/gates.py`, `RESULTS.json`, `reporting/release_gates.csv`), each read from one artefact and one column written by an earlier component, nothing recomputed: above chance and beyond identity (the D19 margins of 0.05 on the main target), the lower 95 % bootstrap bound of the full model above zero, the shuffled baseline within 0.05 of zero, pooled 90 % coverage within 0.85 to 0.95, at least three backbone genes at selection fraction 0.8 or more for the main program, inference parity above 0.95, PERMANOVA p below 0.05 or any trend q below 0.1 on the predicted vectors, and at least one external validator ran with every ran validator passing its own bar and none failed. A missing artefact or column yields `not_evaluated`, which is printed and counted separately from pass and fail, because a gate that silently passes on absent evidence is worse than no gate. Inference parity is defined as the Pearson correlation between the morphology-only path fed the slide's own annotated centroids and the training path (precomputed features, saved feature space, selector and temperature) on the same cells; the two differ only in the unseen-slide embedding slot and in the neighbour graph rebuilt from the same centroids, so a value well below 1 means the inference code drifted from training, which is the failure a product cannot afford. For the same reason an exception raised by the inference path during the parity check is written to `parity_error` and fails the gate instead of being swallowed into a missing number; and the inference path takes each slide's own pixel size (manifest value for training slides, OME metadata for new ones, the configured default only as a printed fallback) so that neighbour offsets in microns match what training used for that slide. `--force X --only Y` recomputes X even though it is not in `--only`, because a forced component that is quietly reloaded would contradict the flag. On the synthetic fixture the gates test the plumbing, not the biology, and the run says so; on the real cohort the first seven are analytical and method-level, the stage gate is the only clinical-validity gate the public data can inform, and a run in which every gate the data can inform passes supports a method paper and a pilot study design, not a clinical claim (`docs/VALIDATION_ROADMAP.md`).

## Literature anchoring the target programs

- Yip RKH et al. (2025) Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens with spatial transcriptomics. Blood 146(15):1837-1849. https://doi.org/10.1182/blood.2025028896
- Rasche L, Weinhold N (2025) Unlocking myeloma's spatial world at the micrometer scale. Blood 146(15):1744-1746. https://doi.org/10.1182/blood.2025030205
- Frenquelli M et al. (2020) The WNT receptor ROR2 drives the interaction of multiple myeloma cells with the microenvironment through AKT activation. Leukemia 34(1):257-270. https://doi.org/10.1038/s41375-019-0486-9
- van Andel H et al. (2019) Aberrant Wnt signaling in multiple myeloma: molecular mechanisms and targeting options. Leukemia 33(5):1063-1075. https://doi.org/10.1038/s41375-019-0404-1
- Teoh PJ et al. (2025) Resistance to immunomodulatory drugs in multiple myeloma: the cereblon pathway and beyond. Haematologica 110(5):1074-1091. https://doi.org/10.3324/haematol.2024.285636
- Spaan I et al. (2023) Wnt inhibitors reduce the unfolded protein response and enhance bortezomib-induced cell death in multiple myeloma. Blood Advances 7(7):1103-1107. https://doi.org/10.1182/bloodadvances.2022008442
- Li H et al. (2025) Targeting Wnt/beta-catenin signaling enhances the efficacy of anti-CD38 immunotherapy in multiple myeloma. Neoplasia 70:101242. https://doi.org/10.1016/j.neo.2025.101242
- Ohlstrom DJ et al. (2026) Spatial multi-omics of multiple myeloma uncovers niche-dependent pro-myeloma and immunosuppressive signaling in the bone marrow and extramedullary lesions. bioRxiv (preprint). https://doi.org/10.64898/2026.03.23.713195
- Bingham NE et al. (2026) Spatial transcriptomics identifies a suppressive, T-cell-excluded tumor microenvironment in extramedullary myeloma. Blood Advances 10(18):6221-6235. https://doi.org/10.1182/bloodadvances.2026020500
- Koops M et al. (2026) Spatial profiling identifies a distinct and topographically defined tumor microenvironment that emerges during multiple myeloma evolution. Blood Cancer Discovery 7(6):999-1016. https://doi.org/10.1158/2643-3230.BCD-26-0080
- Pilcher WC et al. (2026) A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma. Nature Cancer 7(1):224-246. https://doi.org/10.1038/s43018-025-01072-4
- Rasche L et al. (2022) The spatio-temporal evolution of multiple myeloma from baseline to relapse-refractory states. Nature Communications 13:4517. https://doi.org/10.1038/s41467-022-32145-y
- Li M et al. (2026) SpatialCOC: an integrative framework for spatial continuous mapping and cross-omics correction in spatial multi-omics data. Nature Communications 17:5268. https://doi.org/10.1038/s41467-026-71882-2
- Bandyopadhyay S et al. (2024) Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging. Cell 187(12):3120-3140.e29. https://doi.org/10.1016/j.cell.2024.04.013
- Xiang ZJ, Xi YT, Hasson U, Ramadge PJ (2009) Boosting with spatial regularization. Advances in Neural Information Processing Systems 22.
- Luo J, Fu J, Lu Z, Tu J (2024) Deep learning in integrating spatial transcriptomics with other modalities (review covering histology-to-expression prediction). Briefings in Bioinformatics 26(1):bbae719. https://doi.org/10.1093/bib/bbae719 (not verified in docs/research)

Verified citations, findings and claim strengths: `docs/research/LITERATURE_VERIFIED.md`.
