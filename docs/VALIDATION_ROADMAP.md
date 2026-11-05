# From research pipeline to product: what is validated now and what is not

This repository is a research pipeline. A company built on its findings would need to
show, in this order, that the measurement is reproducible (analytical validity), that it
is associated with the clinical state it claims (clinical validity), and that acting on it
changes outcomes (clinical utility). The table says which of these the current code
addresses, with which artefact, and what each later step needs. Nothing below is a
regulatory claim; it is the plan a technical founder would be asked for.

## Analytical validity: does the pipeline measure the same thing every time?

| question | what the code does now | artefact | what is still missing |
|---|---|---|---|
| Reproducible run | seeds everywhere, manifest hashes per component, `environment.json`, deterministic encoder | `run_manifest.json`, `<component>/manifest.json` | a second machine reproducing the numbers to the printed tolerance |
| Same cells, re-measured | split-half reliability of every target, the ceiling in every table | `program_inference/target_reliability.csv` | none for the targets; the image side is covered by the next rows |
| Same slide, re-imaged | perturbation robustness (gamma, blur, rotation, noise, scale) | `external_validation/perturbation_robustness.csv` | real re-scans on a second scanner; multi-site slides |
| Same section, different modality | H&E path through the same encoder (`stain_mode: hematoxylin`), protein bridge | config switch; `external_validation/protein_bridge.csv` when data exist | paired H&E and Xenium sections; a paired protein-imaging dataset (none public for myeloma) |
| Inference equals training | `parity_pearson`: the morphology-only path on the slide's annotated centroids against the training path on the same cells | `deployment/sanity_check.csv` | none |
| Uncertainty is honest | Gaussian and split-conformal intervals, coverage per patient | `evaluation/calibration.csv` | coverage on an independent cohort |

## Clinical validity: is the score associated with the clinical state?

| question | what the code does now | artefact | what is still missing |
|---|---|---|---|
| The score tracks a program, not density | expression-matched random gene sets, split-gene halves | `external_validation/program_specificity.csv` | none at the method level |
| Program burden differs by disease stage | PERMANOVA and Jonckheere-Terpstra trend on patient-level burdens | `evaluation/stage_association.csv` | a cohort large enough to power the test (the public one has 21 patients) |
| Association with progression-free survival | Cox model, data-gated | `evaluation/cox_pfs.csv` when `clinical_csv` is given | a cohort with outcomes; the public cohort has none |
| Transfer to unseen patients and platform | independent-cohort validator, data-gated | `external_validation/independent_cohort.csv` | an independent Xenium or Visium HD cohort (the Visium HD cohort of Ohlstrom et al. 2026 is embargoed as of writing) |
| The genes are the right genes | stability-selected backbone with evidence grades; sensitivity-pointing genes flagged | `interpretation/gene_backbone.csv` | functional validation outside this pipeline |

## Clinical utility: does acting on the score help?

Nothing in this repository addresses utility. It would need a prospective or
retrospective study in which the morphology-only score is available at a treatment
decision and outcomes are compared. The pipeline's contribution to that study is the
score, its uncertainty and its exports (`deployment/`), which are the same files a
pathologist would open.

## Reading the gates as a founder

`RESULTS.json` has nine gates. The first seven (above_chance, shuffled_at_chance,
beyond_identity, ci_excludes_zero, calibrated, backbone_stable, inference_parity) are
analytical and method-level and can pass on the public cohort. `stage_association` is the
only clinical-validity gate the public data can inform, and it is underpowered.
`external_validation` passes when the validators that could run did so without error and
each met its own stated bar (`docs/research/VALIDATION_METHODS.md`); the data-gated
validators stay `skipped` with their reason and do not count against the gate. A run in which every gate the data can inform passes supports a method paper
and a pilot study design, not a clinical claim.
