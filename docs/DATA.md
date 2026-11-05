# Data

## Training cohort: GSE299193 (human SubSeries of GSE299207)

Yip RKH et al. (2025) Profiling the spatial architecture of multiple myeloma in
human bone marrow trephine biopsy specimens with spatial transcriptomics. Blood
146(15):1837-1849. https://doi.org/10.1182/blood.2025028896

- 21 human bone marrow trephine biopsies: 4 controls, 2 MGUS, 5 smouldering
  myeloma, 10 newly diagnosed myeloma (plus one relapse sample on a 377-gene
  panel, excluded by default).
- Platform: 10x Xenium Prime 5K human pan-tissue panel (5,001 genes), subcellular.
- Files used per sample (fetched from the GEO supplementary listing at run time):
  `cell_feature_matrix.h5` (cells x genes counts), `cells.csv.gz` or
  `cells.parquet` (centroids in microns), `morphology.ome.tif.gz` (DAPI z-stack,
  13 planes, pyramidal OME-TIFF, PhysicalSizeX 0.2125 um at level 0).
- Not used: `transcripts.parquet`, cell and nucleus boundaries.
- One trephine per patient, so the patient id equals the sample id.

## What the pipeline derives from it

| quantity | how | where |
|---|---|---|
| cell QC | `total_counts >= 60`, `n_genes >= 50` | `preprocessing/qc.py` |
| per-cell program counts | sum of raw counts over program genes in the panel | `preprocessing/spatial_summaries.py` |
| neighbourhood counts | sum over all cells within 30 um (KD-tree, sparse adjacency) | same |
| density | cells within 25, 50, 100 px at level 1 (10.6, 21, 42 um) | same |
| program score | `log1p(1e4 * program / total)` at both scales, cohort z-scored | `targets/programs.py` |
| reliability | binomial split-half, Spearman-Brown | `targets/reliability.py` |
| lineage label | argmax of expression-matched marker scores over 9 lineages | `targets/programs.py` |
| image crop | bounding box of the sample's cells plus a 64 px margin at level 1, z max-projected, uint8, cached | `data/images.py` |

## Later cohorts (configuration only)

- Visium HD trephines (Ohlstrom et al. 2026, preprint; GEO deposit embargoed until publication): add rows with `platform: visium_hd`,
  `cells` = `tissue_positions.parquet`, `scalefactor` as needed. Bins are treated as cells.
- An H&E cohort: set `he_image` in the manifest row and `stain_mode: hematoxylin`.
- Clinical tables: CSV with `patient, pfs_months, pfs_event` (`clinical_csv`), or a
  CoMMpass-style bulk table (`commpass_csv`, controlled access).
- An independent cohort for the `independent_cohort` validator: set
  `validation.validation_cohort_manifest` to a JSON list of ingest rows with local paths
  (Xenium, or Visium HD with bins treated as cells); the final model scores it from the
  images alone and compares with the cohort's own expression (`external_validation/`).
- CODEX / PhenoCycler: a cell-by-marker table (`codex_table`) and the matching DAPI image
  (`codex_dapi`), read by the `protein_bridge` validator of the `external_validation`
  component (the bridge moved out of deployment in v3.1). The public CODEX atlas of Bandyopadhyay et al. 2024 (healthy femoral-head marrow plus a few AML and lymphoma-staging iliac crest trephines)
  (Cell; Figshare+ collection 10.25452/figshare.plus.c.7174914) is the first available test
  of the composition head; no public myeloma CODEX dataset is deposited as of this writing.

## Disk and memory

Full cohort: about 35 to 45 GB of raw downloads (ephemeral), 2 to 6 GB of crops
and 2.6 GB of float16 embeddings (persistent). Peak RAM stays under 12 GB because
whole-slide images are decoded to a disk memmap and only the crop is held, and
embeddings are read chunk by chunk.
