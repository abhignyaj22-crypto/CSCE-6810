# External validation: what each validator asks, how, and what a pass looks like

The `external_validation` component runs after interpretation and before deployment. It
exists because a cross-validated correlation on held-out patients, the headline of the
evaluation component, does not answer four questions a reviewer, a regulator or a
customer will ask: does the prediction track the program rather than any gene set that
looks like it, does the predicted map have the tissue's spatial organisation, would a
different scanner give the same map, and what happens on a cohort the model never saw.
The original design reserved the orthogonal check for a CODEX bridge. No public CODEX
dataset pairs with Xenium on a myeloma trephine, so that check alone is a gate that never
opens; it is kept as one validator among five, and the component records for every
validator whether it ran, was skipped (with the reason) or failed (with the error), in
`external_validation/validation_summary.json`. Nothing is silently passed. Each validator
that ran judges itself against the pass bar stated below and writes `passed` and
`pass_bar` into the summary; the `external_validation` release gate (`reporting/gates.py`)
passes only when at least one validator ran, every validator that ran passed and none
failed, and it lists each validator's outcome in its value. The bars are deliberately
strict: on the synthetic fixture with the untrained `tiny_test` encoder the robustness
bar fails under a gamma change, and the gate reports that failure rather than hiding it.

## 1. program_specificity (always runs)

Question. Does the morphology prediction track the gene program specifically, or does it
recover total transcript abundance, density or cell identity, which any expression score
shares?

How. For the main target and every learned neighbourhood-scale program, random gene sets
are drawn that match the program gene by gene in expression level: each program gene is
replaced by a random usable gene from the same of 25 expression bins (ranked mean log
expression), outside the program, inheriting that gene's weight. This is the control gene
set construction described by Tirosh et al. (2016) for scoring single-cell signatures
(Science 352:189-196, https://doi.org/10.1126/science.aad0501; verified on PubMed), the
construction Seurat's `AddModuleScore` uses. Each random set and the real program are
scored the same way on the same cells, as the weight-normalised mean of log-normalised
expression, at the cell scale or averaged over the 30 um neighbourhood with the same
neighbourhood construction and per-slide pixel size as the real targets. The learned
targets are NMF usages, which a random gene set cannot be pushed through, so the real
program is scored by the same linear proxy and the correlation with the NMF target itself
is reported alongside (`r_target` next to `r_proxy`). Per patient, the out-of-fold
prediction's Pearson with the real program is compared with its Pearson against the
random sets: z-score (the null spread is floored at `1/sqrt(n cells)`, the sampling noise
of a correlation, so a null made narrow by a small panel cannot inflate z), empirical p
(`(1 + #null >= real) / (n + 1)`) and the margin of the real correlation over the null's
95th percentile, with the median over patients as the headline. The genes that carry a
program are those above 1 % of the maximum weight that together hold 90 % of the weight
mass (at most 100), so an unseeded program's long tail does not exhaust the spare genes
the null needs. A split-gene check divides the program's weighted genes into
two random halves and scores each; a coordinated program is tracked through either half.

Pass. `median_z` above 2, most patients with `p_empirical` below 0.05, and both
half-correlations positive and of similar size. A prediction that correlates with the
random sets as well as with the program is recovering something the program shares with
every gene set of its expression level, and is reported as such.

Columns of `program_specificity.csv`: target, program, scale, patient, n_cells, n_genes,
r_target, r_proxy, null_mean, null_sd, null_q95, margin_over_null_q95, z, p_empirical,
r_half_a, r_half_b, n_random_sets. `program_specificity_summary.csv` holds the per-target medians.

## 2. spatial_fidelity (always runs)

Question. Does the predicted map reproduce the spatial organisation of the measured
program, not just its per-cell ranking?

How. Per patient, Moran's I of the predicted and of the measured target on the same
k-nearest-neighbour graph the model's neighbour tokens use (binary weights), and their
ratio. Then the plasma-cell-dense niche contrast: cells with at least five plasma cells
within `nb_radius_um` (the rule the protein bridge uses on CODEX tables) against the
rest, as Cohen's d with the pooled standard deviation, for the predicted and the measured
values side by side, with a slide-level paired comparison when four or more slides are
present. The Ohlstrom et al. (2026, preprint) niche is plasma-cell dense, so on real data
the measured contrast of the main target is expected to be positive and the predicted
contrast should have the same sign and a comparable size.

Pass. `moran_ratio` near 1 (a map much smoother or much noisier than the tissue shows as a
ratio far from 1) and `sign_agreement` true with Cohen's d of the same order.

Columns of `spatial_fidelity.csv` (long form, `check` in {moran, niche_contrast}):
check, target, patient, n_cells, moran_pred, moran_measured, moran_ratio; n_dense,
n_other, cohens_d_pred, cohens_d_measured, sign_agreement.

## 3. perturbation_robustness (runs when a checkpoint and a slide image exist)

Question. Would a different scanner, exposure, focus or orientation give the same map?

How. The morphology-only path is run on one slide's clean image and once per entry of
`validation_perturbations` on the same detected nuclei: `gamma_<g>` (intensity gamma),
`blur_<sigma px>` (Gaussian blur), `rot90` (quarter-turn rotation of the image with the
nucleus coordinates rotated the same way), `noise_<sd>` (additive Gaussian noise on the
0 to 1 scale) and `scale_<f>` (intensities multiplied and clipped). For every target the
Spearman correlation between perturbed and clean predictions and the mean absolute shift
in standardised units are reported.

Pass. The working bar is `spearman_vs_clean` above 0.9 on the main target under every
perturbation and a mean absolute shift well below the clean standard deviation. The set is
a proxy for scanner shift, not a multi-site study; a real site-to-site study with slides
scanned on two instruments replaces it, and until then the model card says "robustness
under synthetic perturbations", nothing more.

Columns of `perturbation_robustness.csv`: perturbation, description, target, n_nuclei,
spearman_vs_clean, mean_abs_shift, mean_shift, clean_sd.

## 4. independent_cohort (data-gated on `validation_cohort_manifest`)

Question. What does the model do on a cohort it never saw?

How. The manifest is a JSON file: either a list of rows in the ingest manifest format
with local paths (`sample`, `patient`, `stage`, `h5`, `cells`, `image`; Xenium, or Visium
HD with bins treated as cells), or `{"demo": {"n_patients": 2, "seed": 11}}` for a
synthetic cohort with a different seed that exercises the code path. Real rows go
through the same per-sample ingest body as training (`ingest_one_sample`: load, axis
check, QC, count summaries on every cell, crop), then log normalisation, so the cohort is
processed as the training cohort was. The final checkpoint scores every slide from its
image alone (nucleus detection, features, prediction with the unseen-slide embedding),
detected nuclei are matched to the cohort's cells within 8 um, and the predictions are
compared per patient with two measured quantities built from the cohort's own expression:
the fixed-set anchor scores from the count summaries, and the learned programs projected
onto the cohort's genes as the weight-normalised mean of log expression (weighted genes
missing from the cohort's panel are masked and counted per program). A within-slide
permutation of the measured values gives the chance level and the same 0.05 margin as the
main cross-validation is applied (`above_chance`).

Pass. `median_r` above `median_r_shuffled` by more than 0.05 on the main target, with the
anchor and the learned-proxy rows agreeing. When no manifest is given the validator is
skipped and says so; the Visium HD trephine cohort of Ohlstrom et al. (2026, preprint) is
embargoed as of writing (docs/DATA.md), so there is no public independent cohort yet.

Columns of `independent_cohort.csv`: target, measured_kind (anchor or learned_proxy),
patient, slide, n_matched, pearson, pearson_shuffled, pred_mean, measured_mean.

## 5. protein_bridge (data-gated on `codex_table` and `codex_dapi`)

Question. Do the composition head and the predicted programs agree with protein-level
facts on a section imaged by CODEX or PhenoCycler?

How. Unchanged from the earlier deployment bridge: the model runs on the DAPI channel
resampled to the working pixel size, detected nuclei are matched to the protein cell
table within 8 um, the composition head is compared with the protein lineages in centred
log-ratio space, and predicted programs are contrasted between protein-defined
plasma-dense niches and the rest (`evaluation/codex_validation.py`). Results go to
`protein_bridge.json`, `protein_bridge.csv` (niche contrast) and
`protein_bridge_matched_nuclei.csv`. When no nuclei are detected on the protein-imaging
DAPI image, or fewer than ten match within 8 um, the validator is recorded as `skipped`
with that reason (the bridge's own `aborted` and `insufficient_matches` outcomes are
mapped to `skipped` so the gate and the counts see them). Its verdict is the per-cell
lineage agreement between the composition head and the protein lineage above one half;
a model without a composition head leaves the bridge unjudged.

Pass. A positive CLR correlation of compositions and a positive niche contrast for the
Wnt and adhesion set. No public dataset pairs CODEX with Xenium on the same myeloma
trephine; the code path is exercised on a synthetic fixture in the tests.

## Honest limits

- None of the five validators is a clinical validation. Resistance is inferred from gene
  set abundance associated with resistance in the literature, not measured against
  treatment response; a cohort with outcomes is the next data gate (docs/research).
- Validators 1 and 2 reuse the training cohort's expression with out-of-fold
  predictions; they are internal checks, not external data.
- The perturbation set is synthetic. It bounds sensitivity to a few nuisance factors and
  says nothing about staining protocol or tissue processing differences.
- The learned-program proxy used in validators 1 and 4 is linear; the reported
  `r_target` shows how far the proxy sits from the NMF usage on the training cohort.
