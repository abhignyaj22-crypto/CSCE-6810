# mmprog: gene-program scores from bone marrow morphology

Supervised deep learning that predicts investigator-defined gene-program
scores for every nucleus in a bone marrow trephine from morphology alone, trained
on public Xenium 5K spatial transcriptomics of myeloma trephines (GSE299193,
Yip et al., Blood 2025). Inference needs an image and nothing else.

The target of most interest is an investigator-defined gene set of Wnt ligands,
receptors and secreted antagonists (WNT5A, WNT5B, WNT10A, ROR2, FZD3, LRP6, FRZB,
DKK1, SFRP1, SFRP2) and adhesion and survival genes. Ohlstrom et al. (2026, preprint)
describe plasma-cell-dense niches enriched for non-canonical Wnt and adhesion-related
expression on Visium HD trephines, with Xenium validation on the Yip et al. data.
Progression-free survival in that preprint is a separate 15-gene signature; that
list is empty here (`signature_15`), and this Xenium cohort has no survival or
drug-response table. The scores are gene-set abundances associated with drug
resistance in the myeloma literature, not a Wnt pathway activity and not a
measured drug response. WNT10A, NCAM1 and BCL2 are flagged because their notes
point toward sensitivity or a better outcome. The Wnt members and the non-Wnt
members are also scored separately. Three other sets (plasma-cell stress
and survival, T-cell exhaustion and regulation, cytotoxicity) are predicted
alongside. The research framing, the verified literature and the gene-level
pathway audit are in `docs/research/`.

## How the v3 pipeline runs

`main.py` is the only entry point and reads like an ordinary program: it builds the configuration and the run context, then calls the twelve components directly, one line each (`ctx = ingest(ctx)`, `ctx = preprocess(ctx)`, ... `ctx = report(ctx)`), and ends by printing the release gates. Each component function is defined at the bottom of its own module under `src/mmprog/components/`, so "go to definition" lands on the component. Every component prints what it does and why, writes its artefacts under `workspace/runs/<run_name>/<component>/` together with a `manifest.json` (config hash, input hashes, wall time, device), and skips itself on re-run when nothing upstream changed. The run directory also gets `environment.json` (interpreter, library and CUDA versions, GPU, git commit, resolved configuration, command line) and `RESULTS.json` (the release gates: pass, fail or not evaluated, each read from one artefact).

```bash
pip install -e ".[dev,geo,clinical]"          # torch: see pytorch.org for the CUDA build you want
python -m pytest -q                            # unit, regression and end-to-end tests (CPU)
python main.py --config configs/synthetic.yaml                         # CPU demo, no download
python main.py --config configs/smoke.yaml                             # 4 trephines
python main.py --config configs/default.yaml --set training.max_epochs=40
python main.py --config configs/h100.yaml                              # one H100: bf16, TF32, compile
python main.py --config configs/default.yaml --only evaluation,reporting   # earlier components load their artefacts
python main.py --config configs/default.yaml --force training              # recompute training and everything after it
python main.py --config configs/default.yaml --dry-run                     # resolved config and what each component would do
python main.py --list-components
bash scripts/release_check.sh                                              # lint, fast tests, synthetic run, release gates
```

| component | what it does | main artefacts |
|---|---|---|
| ingest | GEO manifest, downloads, Xenium loading, count summaries on every cell, image crops | samples.json, adata_raw.h5ad, full_xy.npz |
| preprocess | cell QC, log normalisation, missing-gene mask across samples, nucleus detection | adata.h5ad, gene_mask.npy, nuclei.npz, qc_summary.csv |
| program_inference | anchored factorisation of expression into gene programs seeded by the literature sets, targets at cell and 30 um scales, split-half reliability, lineage labels | program_weights.csv, program_scores.npy, targets.json, Y.npy, target_reliability.csv, panel_coverage.csv |
| feature_extraction | frozen encoder embeddings, g x g patch-block tokens, morphometrics, neighbour graph | emb.f16.npy, grid.f16.npy, morpho.npy, graph.npz, encoder_card.json |
| feature_selection | dead and redundant features dropped; everything else reported | feature_selection.json, selection_report.csv |
| unified_model | builds the token network and writes its specification | model_spec.json |
| training | patient-level 7-fold cross-validation of every variant, three seeds, final fit | folds.json, cv_preds_<variant>.npy, cv_unc_<variant>.npy, final_model.pt, hardware.json, cv_fold_log.csv |
| evaluation | per-patient metrics, reliability adjustment, calibration, decision rules, burden, stage association (PERMANOVA and ordered trend; classical MANOVA optional) | cv_results.csv, cv_results_reliability_adjusted.csv, calibration.csv, decisions.txt, program_burden.csv, stage_association.csv, per_patient.csv |
| interpretation | spatially regularized boosting, attention and integrated-gradient attribution, gene backbone with enrichment, motifs | srb_results.csv, srb_rings.csv, pathway_attribution.csv, gene_backbone.csv, enrichment.csv, motif_enrichment.csv |
| external_validation | modality-agnostic validation: program specificity against expression-matched random gene sets, spatial fidelity, perturbation robustness, independent cohort and protein-imaging bridges when data are given | validation_summary.json, one CSV per validator |
| deployment | morphology-only inference on a training slide, parity check, batch slides, exports | vmap_check.csv, vmap_<slide>.csv, sanity_check.csv, exports/<slide>_qupath.tsv, exports/<slide>_xenium_explorer.csv |
| reporting | model card, figures, QA tiers, release gates, CONTRIBUTIONS.md; every number read from an artefact | model_card.json, CONTRIBUTIONS.md, release_gates.csv, figures/, qa_tier2.csv, qa_tier3.csv; RESULTS.json at the run root |

New in v3, and what each file means:

- `program_inference/program_weights.csv`: one row per learned program (the six seeded sets plus `extra_<i>`), one column per panel gene; the gene weights the targets are projected on. Targets are named `prog_<program>_<scale>_z`; the v2 fixed-set scores are kept as `anchor_<program>_<scale>_z` so the gain from learned weights is measured, not assumed.
- `interpretation/gene_backbone.csv`: genes whose weight survives stability selection over refits on random half-patient subsamples, with the selection fraction, the weight relative to the program maximum and the literature evidence grade from `resources/gene_annotations.json`; a gene whose evidence points toward drug sensitivity is flagged in the same row.
- `interpretation/enrichment.csv`: hypergeometric over-representation of each backbone against the curated gene sets with the measured panel as background, Benjamini-Hochberg q-values.
- `evaluation/calibration.csv`: empirical coverage of the 50 % and 90 % Gaussian intervals and of the split-conformal interval, pooled and per patient, plus an expected calibration error; read the per-patient spread, not the pooled value.
- `interpretation/pathway_attribution.csv`: cross-attention mass of each program token on the centre, inner-ring, outer-ring and neighbour-cell tokens (descriptive, not causal) and integrated-gradient summaries per target.
- `deployment/vmap_*.csv`: one row per detected nucleus with `x, y`, `<target>_pred`, `<target>_std` (temperature-scaled predictive standard deviation) and `lineage_prob_<name>`; the same columns go to QuPath and Xenium Explorer.

New in v3.1:

- `evaluation/stage_association.csv`: PERMANOVA on the patient-level vector of predicted program burdens and a Jonckheere-Terpstra trend test per program along control < MGUS < SMM < MM, each run on the predicted and on the measured vectors, with Benjamini-Hochberg q-values; classical MANOVA only under `evaluation.stage_test: manova` or `both`.
- `external_validation/validation_summary.json` and one CSV per validator that ran: program specificity against expression-matched random gene sets, spatial fidelity (Moran's I and plasma-dense niche contrast, predicted versus measured), perturbation robustness, and the independent-cohort and protein-imaging bridges when their data are given (`docs/research/VALIDATION_METHODS.md`).
- `deployment/sanity_check.csv` carries `parity_pearson`, the agreement between the morphology-only inference path and the training path on the same cells.
- `RESULTS.json` and `reporting/release_gates.csv`: nine release gates read from the artefacts above; `environment.json`: the reproducibility record. What every result means and what a pass looks like: `docs/RESULTS_SPEC.md`; what is validated now and what a product would still need: `docs/VALIDATION_ROADMAP.md`.

Scores are abundances of investigator-defined gene sets associated with drug resistance in the myeloma literature, not pathway activity and not a drug-response measurement (`docs/research/PATHWAY_INTERPRETATION.md`).

The design brief (which published algorithm each component adopts, what is adapted and what is this repository's own choice) is `docs/DESIGN_V3.md`; the software contract is `docs/ARCHITECTURE_V3.md`; the reasons behind every decision, D1 to D36, are in `docs/DECISIONS.md`.

## Repository layout

```
main.py             the entry point: config, context, twelve components called in order, release gates printed last
configs/            default.yaml, smoke.yaml, synthetic.yaml, h100.yaml (sectioned by component)
resources/          gene_programs.json, gene_annotations.json, lineage_markers.json, codex_markers.json, stage_vocabulary.json
src/mmprog/
  config.py         flat Config dataclass; sectioned YAML is flattened on load; typed --set overrides
  logging_utils.py  say(), banner(), timed(), show_table(): prints mirrored to run.log
  resources.py      gene lists, annotations and stage vocabulary; prints per-gene evidence grades
  core/             Context (typed run state), BaseComponent (manifest-hash skip rule), pipeline driver
  components/       one module per component, each ending with the function main.py calls: ingest, preprocess,
                    program_inference, feature_extraction, feature_selection, unified_model, training, evaluation,
                    interpretation, external_validation, deployment, reporting
  data/             manifest (GEO), download, tenx (10x h5 without scanpy), images (OME-TIFF memmap), synthetic, codex
  preprocessing/    qc, spatial_summaries (counts on all cells), nuclei (tiled watershed), stain
  programs/         anchored_nmf (seeded factorisation), targets_builder (learned and anchor-only targets, reliability)
  targets/          programs (scores, lineage labels), reliability (split-half ceiling)
  features/         patches, encoders/registry (resnet50, uni, virchow2, hoptimus0, phikon_v2), embed (EMB and GRID memmaps),
                    patch_grid, morphometrics, feature_space (PCA), selection
  models/           tokens, unified (the token network), heads (Gaussian NLL, InfoNCE, smoothness), spatial_boost (SRB),
                    program_net (v2 reference), losses, neighbours
  training/         data (static padded batches), loop (bf16, TF32, compile, early stopping, temperature), cv (variants, folds)
  evaluation/       metrics, decisions, calibration, clinical, stage_association (PERMANOVA, trend test), manova,
                    transfer, spatial_boost_eval, codex_validation (re-export of validation/protein_bridge)
  interpretation/   attribution (attention mass, integrated gradients), backbone (stability selection), enrichment
  validation/       perturbations, program_specificity, spatial_fidelity, robustness, independent_cohort, protein_bridge
  inference/        virtual_map_v3 (morphology-only inference), export (QuPath, Xenium Explorer), virtual_map (v2)
  analysis/         motifs (byte-pair encoding over neighbourhood lineage sequences)
  reporting/        qa_tiers, figures, model_card, contributions, gates (release gates read from artefacts)
  pipeline.py       the v2 stage pipeline, kept for ingest_one_sample and regression tests
scripts/            release_check.sh (lint, tests, synthetic run, gates), run_pipeline.py (v2 stages), infer_slide.py (v2 inference)
notebooks/          colab_runner.ipynb (v2 runner; v3 runs through main.py)
tests/              unit/ (one file per component), regression/ (v2 parity), e2e/ (main.py on the synthetic fixture), v2 tests
docs/               ARCHITECTURE_V3.md, DESIGN_V3.md, DECISIONS.md, CODE_CHANGES.md, RESULTS_SPEC.md, VALIDATION_ROADMAP.md, DATA.md,
                    ARCHITECTURE.md (v2)
docs/research/      RESEARCH_PROBLEM.md, LITERATURE_VERIFIED.md, PATHWAY_INTERPRETATION.md, METHODS_NOTES.md,
                    VALIDATION_METHODS.md, ALGORITHM_SCOUT_VISION.md, ALGORITHM_SCOUT_PROGRAMS.md
```

## Reading the results honestly

The decision rules (`evaluation/decisions.py`) are applied to the CSV, not to a plot:

- `full` above `shuffled` and above `composition_only`: morphology carries program information beyond cell identity.
- `full` about equal to `composition_only`: the model is a cell-type detector; reported as such.
- `oracle` well above `full`: identity is the bottleneck; the next lever is the encoder, not the head.
- `no_neighbours` about equal to `full`: the neighbour tokens earn nothing on this data (v3 maps this onto the v2 `no_spatial` rule); ablation sentences are also written for `no_program_tokens`, `no_grid`, `no_smooth` and for the learned-versus-fixed target comparison (`anchor_only_targets`). The `align` variant (contrastive alignment switched on; off by default since it sat at chance on the fixture) appears as a column of `cv_results.csv`.

Every correlation is reported next to the attainable ceiling from split-half
reliability, because a correlation on a noisy target is uninterpretable without it.

The run ends with the release gates (`RESULTS.json`, `reporting/release_gates.csv`): nine
checks read from the artefacts, each PASS, FAIL or NOT EVALUATED. A gate whose artefact is
missing is never counted as passed. The meaning of every artefact and gate, and what a
positive and a negative finding read like, is in `docs/RESULTS_SPEC.md`.

## Later phases (data-gated, same code)

Phase 2 Visium HD cohort and cross-platform transfer; Phase 3 clinical endpoints
(PFS Cox, CoMMpass signature concordance); Phase 4 pathology-pretrained encoder
(`encoder: uni`) and stain harmonisation (`stain_mode: hematoxylin`); Phase 5
batch inference over a slide folder with QuPath and Xenium Explorer export.
Each is a config switch; a phase with no data prints its gate and moves on.

## Citation of the data and targets

Data: Yip RKH et al. (2025) Blood 146(15):1837-1849, https://doi.org/10.1182/blood.2025028896 (GSE299207 / GSE299193).
Niche programs that motivate the gene sets: Ohlstrom DJ et al. (2026) bioRxiv, https://doi.org/10.64898/2026.03.23.713195 (preprint).
Gene sets, their corrected labels and per-gene evidence grades are in `resources/gene_programs.json`,
`resources/gene_annotations.json` and `docs/research/PATHWAY_INTERPRETATION.md`.

## License

MIT.
