# What the pipeline produces, and what each result means

This is the results contract of `main.py`. One run of the pipeline on the Xenium cohort
produces three things, in this order of importance:

1. **Per-cell program scores predicted from morphology alone**, with calibrated uncertainty,
   for every nucleus of every slide (`deployment/vmap_<slide>.csv`, exports for QuPath and
   Xenium Explorer). This is the product: an image goes in, a map of resistance-associated
   program abundance comes out, and nothing else is needed at inference time.
2. **The evidence that those scores mean something**: patient-level cross-validation of the
   unified model against its baselines and ablations, the attainable ceiling of every target,
   calibration of the uncertainty, the release gates (`RESULTS.json`), and the external
   validators (`external_validation/`).
3. **The gene backbone of each resistance-associated program**: the coordinated set of genes
   whose weight in the learned program survives stability selection, graded by literature
   evidence, with enrichment against the curated sets (`interpretation/gene_backbone.csv`,
   `interpretation/enrichment.csv`). This is the "inferring drug-resistance gene programs"
   output: the method infers, the problem is drug resistance, the strategy is gene programs.

Everything else in the run directory is the audit trail that lets a reader trace every
number in `CONTRIBUTIONS.md` to the artefact it came from.

## Vocabulary

- **Program**: a weighted set of genes. Six are seeded by literature gene sets
  (wnt_camdr, wnt_core, adhesion_survival, pc_stress_survival, t_exhaustion, cytotoxicity)
  and refined by anchored non-negative factorisation of the measured expression; two more
  (`extra_0`, `extra_1`) are learned without a seed and reported but not interpreted as
  resistance programs unless their backbone says so.
- **Target**: a per-cell number the image model learns to predict. `prog_<program>_cell_z`
  is the program's abundance in the cell's own transcripts; `prog_<program>_nb_z` is its
  abundance within 30 um of the cell. Both are log-normalised and z-scored across the
  cohort. `anchor_<program>_<scale>_z` is the same quantity with the fixed literature set
  instead of the learned weights, kept so the gain from learning weights is measured.
- **Main target**: `prog_wnt_camdr_nb_z`, the neighbourhood-scale abundance of an
  investigator-defined Wnt and adhesion gene set, motivated by the plasma-cell-dense
  niche in Ohlstrom et al. (2026, preprint). That preprint's progression-free-survival
  result is a separate 15-gene signature, which this repository does not score.
- **Variant**: one training configuration. `full` is the unified model; the others are
  baselines (`shuffled`, `composition_only`, `oracle`) and ablations (`no_neighbours`,
  `no_grid`, `no_program_tokens`, `no_comp`, `no_smooth`, `align`, `anchor_only_targets`).
- **r**: per-patient Pearson correlation between out-of-fold predictions and measured
  targets, summarised as the median over patients with a bootstrap confidence interval.
  Cells of one patient are never split between training and test.

## Results by component

### program_inference

| artefact | columns | what to expect |
|---|---|---|
| `program_weights.csv` | one row per program, one column per panel gene, non-negative weights normalised to sum 1 | seeded genes carry most of the weight; a few non-seeded genes enter with small weight; the two extra programs are dominated by whatever co-varies most in the cohort (often lineage) |
| `target_reliability.csv` | target, sample, r_half, r_sb, n_cells | r_sb is the split-half reliability after Spearman-Brown correction; sqrt(r_sb) is the ceiling any image model can reach on that target in that patient; neighbourhood-scale targets are far more reliable (0.8 to 0.98) than cell-scale ones (0.0 to 0.6) because they pool transcripts |
| `panel_coverage.csv` | program, n_genes, in_panel, missing | every seeded gene present on the 5K panel; a missing gene is reported, never silently dropped |
| `lineage.csv` | per-cell lineage label and score | plasma cells form the largest class in myeloma trephines and a small one in controls |

### training and evaluation

| artefact | columns | what to expect |
|---|---|---|
| `evaluation/cv_results.csv` | target, one column per variant (median per-patient r), full_95ci | the table every claim is made from; see the reading rules below |
| `evaluation/cv_results_reliability_adjusted.csv` | same, divided by sqrt(r_sb) | the fraction of the attainable ceiling reached |
| `evaluation/per_patient.csv` | target, model, patient, r, r_adj | the spread behind every median; a result that holds in most patients is a finding, one carried by a single patient is not |
| `evaluation/calibration.csv` | target, level, coverage_pooled, coverage_patient_median/min/max, mean_std, kind (gaussian or conformal), q, ece | a 90 % interval should cover about 90 % of cells, in every patient, not only pooled; ece near 0 |
| `evaluation/decisions.txt` | plain sentences | the decision rules applied to the CSV: above chance, beyond identity, identity bottleneck, spatial prior, each ablation, learned versus fixed targets |
| `evaluation/program_burden.csv` | patient, stage, n_plasma, burden, stage_ord | the fraction of plasma cells above the cohort median of the predicted main target, per patient |
| `evaluation/stage_association.csv` | vectors (predicted or measured), test (permanova or jt_trend), program ("all" for PERMANOVA), statistic, p_value, q_value (trend rows only), r2_or_rho, n_patients, n_groups, groups, note | PERMANOVA asks whether the vector of predicted program burdens differs between disease stages at the patient level; the Jonckheere-Terpstra rows ask whether each program's burden rises monotonically from control through MGUS and smouldering to newly diagnosed myeloma; the `measured` rows run the same tests on the measured programs so the reader sees what stage structure there was to reproduce; with 21 patients both are underpowered, so a non-significant result is "not detected", not "absent"; a row whose note starts with `skipped:` means the test was degenerate and was not forced; `stage_patient_table.csv` holds the patient-by-program table the tests ran on |
| `training/hardware.json` | device, dtype, compile, throughput | measured, not assumed; the H100 run records bf16, TF32 and cells per second |

Reading rules for `cv_results.csv` on the main target:

- `full` above `shuffled` by more than 0.05 and the lower bound of `full_95ci` above zero:
  the model has learned a cell-to-program link.
- `full` above `composition_only` by more than 0.05: morphology carries program
  information beyond cell identity. This is the main scientific claim and it is only
  made when this holds.
- `full` within 0.05 of `composition_only`: the model is a cell-type detector; the run
  reports it as such.
- `oracle` well above `full`: cell identity is the bottleneck; the next lever is the
  encoder, not the head.
- Each ablation within 0.02 of `full`: that component neither helps nor hurts on this
  data and the report says so. An ablation above `full` by more than 0.02 is printed as a
  cost, never hidden.
- `prog_*` above `anchor_*` by more than 0.02 on the same program and scale (the
  `anchor_only_targets` variant): learned weights are more predictable from morphology
  than the fixed literature set; otherwise the sentence reads "not more predictable", the
  fixed set is kept as the reference and the learned programs are reported as exploratory.

### interpretation

| artefact | columns | what to expect |
|---|---|---|
| `gene_backbone.csv` | program, gene, selection_fraction, weight_relative_to_max, gene_class, evidence_grade, is_wnt_component, note | genes present in at least 80 % of half-patient refits with a normalised weight above the threshold; the evidence grade (established, multiple-study, single-study, inferred) and the note come from `resources/gene_annotations.json`; genes whose evidence points toward drug sensitivity (BCL2 in the venetoclax-sensitive t(11;14) subgroup, WNT10A, NCAM1) are flagged in the note and must not be read as resistance genes |
| `enrichment.csv` | program, gene_set, backbone_size, set_size_in_panel, overlap, p_value, q_value, overlapping_genes | hypergeometric over-representation with the measured panel as background; a backbone that is enriched only for its own seed set has learned nothing new, which is a valid and expected outcome on a seeded factorisation |
| `pathway_attribution.csv` | attention mass of each program token on centre, inner ring, outer ring and neighbour tokens, by lineage; integrated-gradient summaries per target | descriptive: which parts of the image the prediction attends to; not a causal statement |
| `srb_results.csv`, `srb_rings.csv` | spatially regularized boosting on patch blocks: AUC by lambda, Moran's I coherence, ring summaries by micron radius | whether the signal sits in the cell, the inner ring or the outer ring; a flat ring profile means the morphology signal is not spatially localised |
| `motif_enrichment.csv` | neighbourhood lineage motifs (byte-pair encoded) and the main target with and without the motif, permutation p | which cell arrangements carry the program; plasma-cell runs are expected to carry the Wnt and adhesion program |

### external_validation

| validator | artefact and columns | question | pass looks like |
|---|---|---|---|
| program_specificity | `program_specificity.csv`: target, program, scale, patient, n_cells, n_genes, r_target, r_proxy, null_mean, null_sd, null_q95, margin_over_null_q95, z, p_empirical, r_half_a, r_half_b, n_random_sets; `program_specificity_summary.csv`: per-target medians and the fraction of patients with p below 0.05 and with both halves positive | does the prediction track the program rather than transcript abundance, density or lineage? | `median_z` above 2, most patients with `p_empirical` below 0.05, `margin_over_null_q95` positive, and `r_half_a` and `r_half_b` both positive and of similar size; `r_target` next to `r_proxy` shows how far the linear proxy sits from the NMF usage |
| spatial_fidelity | `spatial_fidelity.csv` (long form, `check` in moran or niche_contrast): check, target, patient, n_cells, moran_pred, moran_measured, moran_ratio; n_dense, n_other, cohens_d_pred, cohens_d_measured, sign_agreement | does the predicted map have the measured spatial structure? | `moran_ratio` near 1 in most patients; `sign_agreement` true with Cohen's d of the same order on the main target |
| perturbation_robustness | `perturbation_robustness.csv`: perturbation, description, target, n_nuclei, spearman_vs_clean, mean_abs_shift, mean_shift, clean_sd | do predictions survive gamma, blur, rotation, noise and intensity scaling? | `spearman_vs_clean` above 0.9 on the main target under every perturbation and `mean_abs_shift` well below `clean_sd`; a drop under rotation points at the encoder, under gamma at stain normalisation |
| independent_cohort | `independent_cohort.csv`: target, measured_kind (anchor or learned_proxy), patient, slide, n_matched, pearson, pearson_shuffled, pred_mean, measured_mean | does the model transfer to patients and a platform it never saw? | data-gated on `validation_cohort_manifest`; `median_r` above `median_r_shuffled` by more than 0.05 on the main target, anchor and learned-proxy rows agreeing |
| protein_bridge | `protein_bridge.json`, `protein_bridge.csv` (niche contrast), `protein_bridge_matched_nuclei.csv` | does the composition head agree with protein lineage calls, and do predicted programs contrast between protein-defined plasma-dense niches and the rest? | data-gated on `codex_table` and `codex_dapi`; positive composition agreement in centred log-ratio space and a positive niche contrast for the Wnt and adhesion set; no public myeloma pairing exists as of writing |

Each validator records `ran`, `skipped` or `failed` with its reason and headline numbers
in `validation_summary.json` (the `_overall` entry counts them); a skipped validator is a
gap in the evidence and is reported as one. Each validator that ran judges itself against
its stated bar and writes `passed` (true or false) and `pass_bar` into the summary; the
`external_validation` release gate passes only when at least one validator ran, every
validator that ran passed its bar and none failed, and its value string lists each
validator with its outcome (`program_specificity:pass; perturbation_robustness:fail; ...`).
A validator that ran without a verdict leaves the gate `not_evaluated`. The protein bridge
reports `skipped` (with the reason) when no nuclei are detected or too few match.

### deployment

| artefact | columns | what to expect |
|---|---|---|
| `sanity_check.csv` | slide, main_target, n_detected, n_segmented, n_matched, match_um, pearson, pred_mean, measured_mean, coverage_90, parity_pearson, parity_error, um_per_px | the morphology-only path (nucleus detection, no Xenium segmentation) on a training slide against the measured target; Pearson near the cross-validated r of that patient, 90 % coverage near 0.9; `parity_pearson` is the same path fed the slide's annotated centroids against the training path on the same cells and should be close to 1 (the final model is trained on every patient, so this is a code-drift check, not a held-out score) |
| `vmap_<slide>.csv` | x, y, `<target>_pred`, `<target>_std`, `lineage_prob_<name>` | one row per detected nucleus; the product output |
| `exports/` | QuPath TSV and Xenium Explorer CSV | the same columns for the two viewers |

### reporting

| artefact | what it is |
|---|---|
| `RESULTS.json`, `reporting/release_gates.csv` | the release gates: name, status (PASS, FAIL, NOT EVALUATED), passed, value, threshold, source artefact, meaning, plus a summary with the three counts and the main target; a gate that cannot be evaluated is `not_evaluated`, never passed by default |
| `reporting/CONTRIBUTIONS.md` | the written claims, every number read from an artefact: ceilings, held-out performance, what the run supports, boosting rings, stage association, motifs, protein bridge, calibration, backbone, enrichment, attribution, the gate table and what the run does not establish |
| `reporting/model_card.json` | task, data, features, targets, reliability, results, verdict, caveats, config, encoder card, hardware |
| `reporting/figures/` | the figures behind CONTRIBUTIONS.md |
| `environment.json`, `run_manifest.json`, `<component>/manifest.json` | reproducibility: versions, git commit, config and input hashes, wall time per component |

## Release gates

| gate | passes when | read from |
|---|---|---|
| above_chance | full minus shuffled on the main target > 0.05 | evaluation/cv_results.csv |
| shuffled_at_chance | abs(shuffled) < 0.05 | evaluation/cv_results.csv |
| beyond_identity | full minus composition_only > 0.05 | evaluation/cv_results.csv |
| ci_excludes_zero | lower 95 % bootstrap bound of full > 0 | evaluation/cv_results.csv |
| calibrated | pooled 90 % coverage within [0.85, 0.95] | evaluation/calibration.csv |
| backbone_stable | at least 3 genes at selection fraction >= 0.8 for the main program | interpretation/gene_backbone.csv |
| inference_parity | parity_pearson > 0.95 (morphology-only path on annotated centroids versus the training path on the same cells) and `parity_error` empty; an exception raised by the inference path is recorded there and fails the gate | deployment/sanity_check.csv |
| stage_association | PERMANOVA p < 0.05 or any trend q < 0.1, on the predicted vectors; skipped rows do not count | evaluation/stage_association.csv |
| external_validation | at least one validator ran, every validator that ran passed its own stated bar (`passed` in the summary), none failed; a ran validator without a verdict leaves the gate not evaluated | external_validation/validation_summary.json |

On the synthetic fixture the gates test the plumbing, not the biology, and the run says so.

## What a finding reads like

A positive result on the real cohort is one sentence of the form: "From DAPI morphology
alone, the unified model predicts the neighbourhood-scale Wnt and adhesion program with a
per-patient median r of X (95 % CI a to b; ceiling c), above the cell-identity baseline
(Y) and the shuffled baseline (about 0), with 90 % intervals covering Z % of cells; the
program's backbone is the genes G1 to Gk, of which m have established evidence in
myeloma drug resistance." Every symbol in that sentence is a cell in one of the CSVs above.

The honest negative paths are written by the same code: "the model is a cell-type
detector" (full about equal to composition_only), "identity is the bottleneck" (oracle far
above full), "the learned programs are not more predictable than the fixed sets", and
"not detected at this cohort size" for the stage tests.

## What is not a result

- The scores are abundances of gene sets associated with drug resistance in the
  literature, not pathway activity and not a drug-response measurement
  (`docs/research/PATHWAY_INTERPRETATION.md`).
- Stage association is a biological-plausibility check on a 21-patient cohort, not a
  clinical validation. Progression-free survival and treatment-response endpoints are
  data-gated (`clinical_csv`) and absent from the public cohort.
- Attention and attribution maps describe what the model looks at, not why the biology is
  what it is.
