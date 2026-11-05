# Research problem: estimating plasma-cell niche gene-set abundance from routine trephine morphology in multiple myeloma

Every statement below carries a claim identifier (C1 to C22) from `LITERATURE_VERIFIED.md`,
where the citation and its strength (established, single-study, preprint) are recorded.
Gene-level biology is graded in `PATHWAY_INTERPRETATION.md`.

## 1. The problem

Multiple myeloma plasma cells can survive cytotoxic drugs by adhering to the bone
marrow stroma (cell adhesion-mediated drug resistance). In this literature the
non-canonical Wnt result is adhesion, through ROR2 and WNT5A, not a measured
survival pathway and not a drug-response assay in the training cohort
(PATHWAY_INTERPRETATION.md, sections 2 to 4). T cells inside plasma-cell-dense niches
are reported in a suppressed state with elevated LAG3, and extramedullary disease is
described as TIGIT- and LAG3-dominant and largely T-cell excluded (C9 to C11). Those
claims do not establish response to a T-cell redirecting therapy. A preprint
reports that these states are organised in space: plasma-cell-dense niches in bone marrow
trephines are enriched for non-canonical Wnt gene expression together with
adhesion-related genes, their immune cells carry suppressed transcriptional states with
elevated LAG3, and a 15-gene signature built from spatially selected genes and fitted on
aspirate scRNA-seq predicts progression-free survival (C7, C8, C9; Ohlstrom et al. 2026,
bioRxiv, not peer reviewed). In extramedullary disease the same suppression is
LAG3 and TIGIT dominant and the tumour is largely T-cell excluded (C10, C11; Bingham et al.
2026). With disease evolution, myeloma condenses from scattered cells into dense nodules
with their own ecosystem, and dense architecture at diagnosis predicts worse outcome (C5;
Koops et al. 2026).

These measurements cost thousands of dollars per section and are not part of care. The
bone marrow trephine itself is: every patient has one at diagnosis and most have one at
relapse. The problem this project addresses is whether the nuclear morphology visible in a
routine trephine section carries enough information to estimate the niche gene programs
above, at the scale at which they are organised, so that the spatial biology can be read
from the slide every patient already has.

## 2. Why the question is open

The Xenium 5K study of 21 trephines (C1, C2; Yip et al. 2025) provides matched nuclear
morphology (the DAPI channel) and 5,001-gene expression for every cell, which makes the
question answerable for the first time in marrow. The same study found no universal
myeloma-supporting microenvironment: composition varies between patients and between
regions of one trephine (C3, C4). That heterogeneity is exactly why a per-cell,
morphology-based readout is needed rather than a per-patient score, and it is also why
patient-level cross-validation is the only honest evaluation.

No published model that trains histology-to-expression prediction on bone marrow was
found in the literature search recorded in LITERATURE_VERIFIED.md; the histology-to-
expression methods reviewed by Luo et al. (2024, Briefings in Bioinformatics,
https://doi.org/10.1093/bib/bbae719) are trained on solid tumours and H&E.
Nuclear morphology is also the only channel the public data provides (D2 in DECISIONS.md),
so the project is scoped to DAPI with an explicit stain-harmonisation phase for H&E.

## 3. Hypotheses and the stage that tests each

H1. Nuclear morphology within a 27 um window predicts the neighbourhood-scale score of the
investigator-defined Wnt and adhesion gene set above what cell identity alone explains.
Tested by the full model against the shuffled-label bound and the composition-only
baseline, patient-level cross-validation (stage S4, decision rules in `evaluation/decisions.py`).

H2. The morphological signal attaches to the non-Wnt adhesion and survival genes rather
than to the Wnt components, or the reverse. Tested by scoring `wnt_core` (ten Wnt
members) and `adhesion_survival` (eight non-Wnt members) as separate targets alongside the
investigator-defined 18-gene set (stage S2; PATHWAY_INTERPRETATION.md explains why the Wnt members
cannot be read as pathway activity).

H3. The signal is spatially structured within the patch: it lives in the nucleus itself,
in the immediate neighbours, or in the wider neighbourhood. Tested by spatially regularized
boosting on a grid of patch blocks (C21; Xiang et al. 2009), which yields a spatially
coherent importance map and a centre / inner ring / outer ring summary, against the
AdaBoost control on the same stumps (stage S4b).

H4. Patient-level program vectors differ across control, precursor and myeloma stages,
and the predicted vectors reproduce that difference. Tested by patient-level MANOVA with
Pillai's trace and a patient-level permutation test (stage S5b; METHODS_NOTES.md Q2
explains why cell-level MANOVA is pseudo-replication and why stages are collapsed to
three groups).

H5. Recurrent plasma-cell-dense neighbourhood motifs are enriched for the program, in line
with the plasma-cell-dense niche of C7. Explored by byte-pair encoding over ordered
lineage sequences of each cell's neighbours (stage S5c); exploratory by construction.

H6. The composition the model infers from DAPI agrees with protein-defined composition,
and predicted programs differ between protein-defined plasma-cell-dense niches and the
rest of the marrow. Testable on any CODEX run, because CODEX includes a DAPI channel the
trained model can read directly. No public CODEX and spatial-transcriptomics pair exists
for a myeloma trephine (METHODS_NOTES.md Q1); the public CODEX atlas of Bandyopadhyay et al. 2024 (healthy femoral-head marrow plus
a few AML and lymphoma-staging iliac crest trephines; C17, C18) is the first available test of the composition
head, and the stage is data-gated until a myeloma CODEX section is available (stage S6b).

## 4. How each source shapes the design

| source | what it contributes | where it enters |
|---|---|---|
| Yip et al. 2025, Blood (C1 to C4) | the training data; the finding that niches vary between and within patients | S1 ingest; patient-level CV; per-cell rather than per-patient readout |
| Ohlstrom et al. 2026, bioRxiv (C7 to C9, C22) | the plasma-cell-dense Wnt and CAM-DR niche, LAG3-high immune states, the PFS signature; the Visium HD, Xenium and scRNA-seq integration framework | target gene sets (investigator-defined, motivated by this work); the neighbourhood-scale framing (the 30 um radius is this repository's choice; the preprint works on 16 um Visium HD bins); Phase 2 Visium HD transfer; Phase 3 signature (when published) |
| Bingham et al. 2026, Blood Advances (C10 to C12) | LAG3 and TIGIT dominant suppression, T-cell exclusion, APRIL/BAFF to BCMA signalling in extramedullary disease | t_exhaustion program interpretation; the T-cell exclusion metric in motif analysis |
| Koops et al. 2026, Blood Cancer Discovery (C5, C6, C19) | condensation into dense nodules with evolution; dense architecture predicts outcome; THY1+ stroma inside nodules | density features at three radii; the stage endpoint; the plasma-dense niche definition |
| Pilcher et al. 2026, Nature Cancer (C12 to C14) | immune microenvironment features carry prognosis independent of cytogenetics | Phase 3 clinical endpoints; justification for predicting immune programs, not only plasma-cell programs |
| Rasche et al. 2022, Nature Communications (C15, C16) | spatial genomic heterogeneity concentrates in focal lesions; iliac crest biopsies are comparatively homogeneous | the stated limitation that one iliac trephine samples one site; batch inference over several sites in Phase 5 |
| Li et al. 2026, Nature Communications, SpatialCOC (C20) | continuous spatial representation and cross-omics correction | the neighbourhood-summed target is the kernel-smoothed target in the SpatialCOC sense (uniform disc kernel; a Gaussian kernel is not used because split-half reliability needs integer counts); the composition head is a multi-task auxiliary loss (the features must also explain cell identity); it is only loosely analogous to SpatialCOC's cross-omics correction, whose CCA-based transfer is deferred (METHODS_NOTES.md, Q3.5) |
| Xiang et al. 2009, NeurIPS (C21) | boosting with a spatial regulariser that yields spatially coherent feature selection | `models/spatial_boost.py`, stage S4b |
| Bandyopadhyay et al. 2024, Cell (C17 to C19) | public CODEX atlas of human marrow, including FFPE trephines; THY1+ MSC niches | CODEX bridge (S6b), marker panel in `resources/codex_markers.json` |

## 5. What a positive and a negative result each mean

A positive H1 with H3 showing outer-ring importance means the program is read from the
arrangement of neighbours, which is consistent with a niche-level signal (C7) and argues
for neighbourhood-scale deployment. A positive H1 with centre-only importance means the
plasma cell's own nuclear morphology carries the signal, which is a cell-intrinsic claim and
a different biological story. A negative H1 (full model at the composition-only baseline)
means morphology recovers composition, not a within-type program; that result is still
reported, with the reliability ceiling that bounds it, because it rules out the cheapest
readout and directs effort to the Visium HD and CODEX phases.

## 6. Limits that the design does not remove

- The gene sets are investigator-defined and motivated by a preprint (C7); they are not
  pathway activity scores, three members carry evidence in the opposite direction, and
  no drug-response data exist in the training cohort (PATHWAY_INTERPRETATION.md).
- One trephine per patient and 21 patients: patient-level and slide-level validation
  coincide, MANOVA is at the lower limit of interpretability, and a single iliac site
  undersamples focal lesions (C15).
- DAPI, not H&E; transfer to routine stains is a later, data-gated phase.
- Yip et al. 2025 argues against a universal niche (C3); the project therefore predicts
  per-cell program scores and reports per-patient agreement, and does not assume that
  one niche definition applies to every patient.
