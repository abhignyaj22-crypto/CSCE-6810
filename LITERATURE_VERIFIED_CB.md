# Verified Literature Ledger: Spatial Biology of the Multiple Myeloma Bone Marrow Niche

Verification date: 2026-11-03
Method: Each work was located and checked against primary records (PubMed metadata and PubMed Central full text, the bioRxiv API, GEO records via OmicsDI, and publisher or conference pages). Bibliographic facts labelled "According to PubMed" come from the PubMed MCP tools. Findings are quoted or closely paraphrased from the abstract or the full text; nothing below is inferred. Items that could not be confirmed from a primary source are marked UNVERIFIED with a note on what was tried.

Reading note on sources: for several open access papers the PMC text extraction strips gene symbols that were set in italics (for example "the inhibitory receptor [symbol]"). Where a gene symbol was lost in extraction it is reported only if the abstract or the bioRxiv API supplied it.

---

## Summary table

| # | Work (short) | Status | Venue, year | Peer reviewed | PMID | DOI |
|---|---|---|---|---|---|---|
| 1 | Yip et al., spatial architecture of MM trephines | VERIFIED | Blood 2025;146(15):1837-1849 | Yes | 40643106 | 10.1182/blood.2025028896 |
| 2 | Ohlstrom et al., spatial multi-omics of MM (Wnt, CAM-DR, 15-gene signature) | VERIFIED (preprint; same work as #9) | bioRxiv 2026, posted 2026-03-25, v1 | No (preprint, not yet published) | 41928953 | 10.64898/2026.03.23.713195 |
| 3 | Bingham et al., T-cell-excluded TME in extramedullary MM | VERIFIED | Blood Advances 2026;10(18):6221-6235 | Yes | 42379250 | 10.1182/bloodadvances.2026020500 |
| 4 | Li et al., SpatialCOC | VERIFIED | Nature Communications 2026;17(1) | Yes | 41991905 | 10.1038/s41467-026-71882-2 |
| 5 | Koops et al., topographically defined TME in MM evolution | VERIFIED (platform detail partly unverified) | Blood Cancer Discovery 2026;7(6):999-1016 | Yes | 42678098 | 10.1158/2643-3230.BCD-26-0080 |
| 6 | Rasche et al., spatio-temporal evolution of MM | VERIFIED (published); rs-1177985 identifier UNVERIFIED | Nature Communications 2022;13(1):4517 | Yes | 35922426 | 10.1038/s41467-022-32145-y |
| 7 | Pilcher et al., MM Immune Atlas | VERIFIED | Nature Cancer 2026;7(1):224-246 (DOI carries 2025 prefix; online 2026-01-09) | Yes | 41514053 | 10.1038/s43018-025-01072-4 |
| 8 | Xiang, Xi, Hasson, Ramadge, Boosting with Spatial Regularization | VERIFIED | NIPS 2009 (Advances in NeurIPS 22) | Yes (conference) | n/a | n/a |
| 9 | "Ohlstrom et al. 2026, Visium HD, 15-gene PFS signature" | VERIFIED as the same preprint as #2; "Karimpour-Fard" NOT an author | see #2 | No | 41928953 | 10.64898/2026.03.23.713195 |
| 10 | Bandyopadhyay et al., human bone marrow niche atlas (CODEX) | VERIFIED | Cell 2024;187(12):3120-3140.e29 | Yes | 38714197 | 10.1016/j.cell.2024.04.013 |

---

## 1. Yip et al. 2025, Blood

**Verified citation.** According to PubMed: Yip, R. K. H., Er, J., Qin, L., Nguyen, Q. H., Motyer, A., Rimes, J. S., Light, A., Mishi, R. D., Ling, L., Anttila, C. J. A., Tsui, E., Amann-Zalcenstein, D., Dowling, M. R., Rogers, K. L., Bowden, R., Chen, Y., Harrison, S. J., and Hawkins, E. D. (2025). Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens with spatial transcriptomics. Blood, 146(15), 1837-1849. [DOI 10.1182/blood.2025028896](https://doi.org/10.1182/blood.2025028896). PMID 40643106. PMC12824660. Walter and Eliza Hall Institute (Hawkins lab), with Peter MacCallum Cancer Centre and Royal Melbourne Hospital.

**Status.** Peer reviewed, open access on PMC. The expected citation (Blood 2025, 146(15):1837) is correct.

**Data availability.** GEO SuperSeries GSE299207 (Homo sapiens and Mus musculus, titled "Profiling the spatial architecture of multiple myeloma in human bone marrow trephines with spatial transcriptomics", listed 2025-07-13) with human SubSeries GSE299193 (listed 2025-07-14). Verified via OmicsDI mirrors of GEO. A mouse SubSeries GSE299195 also appears in the same listing but was not opened. The Ohlstrom preprint (#2) independently reports reusing "Xenium data from 10 MM bone marrow biopsies produced by Yip et al."

**Platform and design (from PMC full text).** 10x Genomics Xenium in situ. Pilot on one relapsed trephine with the 377-gene human multi-tissue and cancer panel; main study with the Xenium Prime 5K human pan-tissue and pathways panel (5001 genes). 21 trephines: 4 controls (lymphoma staging, marrow negative), 2 MGUS, 5 smoldering myeloma, 10 newly diagnosed MM. 250,829,044 transcripts in total. Optimized trephine biobanking (neutral buffered formalin, non-decalcified or brief EDTA decalcification) to preserve RNA integrity (DV200). Analysis: Seurat v5, edgeR pseudobulk, SingleR label transfer from a scRNA-seq reference, FICTURE segmentation-free pixel inference at 0.5 um, and scider hexagonal binning (80 um wide, median 55 cells per bin) with unsupervised clustering of bins into spatial "neighborhoods".

**Verified findings.**
- (c, spatial methods) Subcellular spatial transcriptomics of 5001 genes can resolve "major components of the human bone marrow microenvironment and reliably characterize distinct plasma cell populations in samples from healthy, premalignant disease and active myeloma" (abstract).
- (a, plasma-cell niches) "We detected spatially restricted subpopulations of plasma cells in 5 of 10 newly diagnosed myeloma trephines" (abstract). Full text: MM-3, MM-4, MM-5, MM-6, MM-7 and MM-8 showed spatially restricted PC subpopulations, ranging from small scattered colonies to large regions; some subpopulations carried an inflammatory signature.
- (a, niches) "The composition of hematopoietic and stromal microenvironments varied significantly between newly diagnosed myeloma trephines," and also within trephines that had spatially restricted PC subpopulations. "These data are not consistent with the hypothesis that a universal bone marrow microenvironment supports the expansion of malignant plasma cells in myeloma" (abstract).
- (a, stroma) "We did not observe consistent alterations in stromal cell composition across control, MGUS/SM, and MM samples" (full text); stromal subsets (sinusoidal and arterial endothelium, VSMC, MSC, osteoblasts, adipocytes) shifted subtly without a disease-associated trend.
- (b, T cells) "We detected a higher proportion of T cells within MM-1 (neighborhood 2) and MM-4 (neighborhood 2) that correspond to PC clusters that express an inflammatory gene signature, highlighting a possible interplay between inflammatory PCs and the adaptive immune system" (full text). The paper does not report T-cell exhaustion states; the authors explicitly note the 5001-gene panel limits them to "cell composition rather than a more detailed dissection of transcriptional cell states."
- PC expansion in MM versus MGUS/SM (9.74% versus 45.07%) with loss of red blood cells and megakaryocytes at sites of PC expansion.

**Not claimed by this paper.** Wnt signaling, CAM-DR, and exhaustion markers are not analyzed. The stated limitation is that most of the 5001 genes were below the Xenium Prime detection threshold in marrow.

---

## 2 and 9. Ohlstrom et al. 2026, bioRxiv (the Visium HD, Wnt, CAM-DR, 15-gene PFS signature study)

**Verified citation.** According to PubMed: Ohlstrom, D. J., Michaud, M. E., Bakhtiari, M., Vieira Dos Santos, J., Pilcher, W. C., Staub, A., Satpathy, S., Ferguson, K., Mantrala, S., Kim-Schulze, S., Chen, Z., Lonial, S., Kemp, M. L., Sherbenou, D., Lagana, A., Jaye, D. L., Nooka, A. K., Parekh, S., and Bhasin, M. K. (2026). Spatial multi-omics of multiple myeloma uncovers niche-dependent pro-myeloma and immunosuppressive signaling in the bone marrow and extramedullary lesions. bioRxiv. [DOI 10.64898/2026.03.23.713195](https://doi.org/10.64898/2026.03.23.713195). PMID 41928953. PMC13042073.

**bioRxiv API record.** Posted 2026-03-25, version 1, category cancer biology, license CC BY-NC-ND, corresponding author Manoj Bhasin (Emory University), published_doi = "NA". Therefore, as of 2026-11-03, this is still a preprint with no journal version registered with bioRxiv. A PubMed search for an Ohlstrom journal article with this title returned none.

**Lab.** Manoj Bhasin lab (Coulter Department of Biomedical Engineering, Georgia Institute of Technology, and Department of Pediatrics / Biomedical Informatics, Emory School of Medicine; Aflac Cancer and Blood Disorders Center). Collaborators at Icahn School of Medicine at Mount Sinai (Lagana, Kim-Schulze, Chen, Parekh), University of Colorado Anschutz (Sherbenou), Emory Pathology (Jaye) and Emory Hematology (Nooka, Lonial).

**Author-list correction for item 9.** "Karimpour-Fard" does not appear in the author list of this preprint (checked in both the PubMed record and the bioRxiv API record). The attribution "Karimpour-Fard / Bhasin labs" is UNVERIFIED and appears to be incorrect for this work. The expected content (Visium HD, myeloma trephines, 15-gene PFS signature) matches this preprint exactly, so items 2 and 9 are the same work.

**Design (from PMC full text).** Three-layer framework: 10x Visium HD (FFPE, 16 um bins) for whole-transcriptome discovery on 21 MM bone marrow core biopsies (16 at diagnosis, 5 at relapse; 5 to 90% plasma cells; cytogenetics spanning trisomies, t(4;14), t(11;14), gain(1q), del(17p), del(13q)); 252,249 QC-passing 16 um voxels; 10x Xenium data from 10 MM biopsies reused from Yip et al. for single-cell validation; and the MMRF Immune Atlas scRNA-seq (Pilcher et al., 416 biopsies from 336 patients; baseline N = 320, relapse N = 81) for depth and outcomes. Five extramedullary lesions (1 perihepatic soft tissue, 3 lymph nodes, 1 periarticular soft tissue) profiled by Visium HD. Custom voxel annotation and deconvolution with a reference built from public scRNA-seq (including E-MTAB-14010).

**Data availability (from full text).** "The raw and processed data generated in this study have been deposited in the National Center for Biotechnology Information Gene Expression Omnibus and will be made public with publication of this article." The accession is not yet public. The Immune Atlas scRNA-seq is under controlled access at the MMRF VLAB resource; a processed Seurat object "with processed UMI counts and limited metadata" is at Zenodo 10.5281/zenodo.15025566.

**Verified findings.**
- (a, Wnt, plasma-cell niches) "Our approach enabled identification of plasma cell-dense niches enriched for non-canonical Wnt signaling, associated with gene expression supporting cell adhesion mediated drug resistance, inferior progression-free survival, and extramedullary lesions" (abstract). Full text: plasma cells in plasma-cell-predominant neighborhoods upregulated a canonical Wnt antagonist and non-canonical Wnt agonists (gene names lost in extraction); GO:BP "negative regulation of canonical Wnt signaling" and "non-canonical Wnt signaling" had the largest numbers of differentially expressed genes; ligand-to-downstream-gene spatial proximity analysis was used to support paracrine plausibility.
- (a, CAM-DR) "Many of these ligands have been implicated in the promotion of cell adhesion-mediated drug resistance (CAM-DR) in MM"; downstream genes included "promoters of homotypic cell-cell adhesion ... potentially supporting MM cell aggregation." The authors conclude that "spatially restricted tuning of the Wnt signaling pathways may enhance plasma cell survival, chemoresistance, and cell-cell adhesion."
- (b, immune suppression) "Immune cells within these neighborhoods exhibited suppressed transcriptional states, including increased inhibitory receptor expression such as LAG3" (abstract via bioRxiv API; the gene symbol is dropped in the PubMed abstract). Full text: T cells in plasma-rich niches showed "enhanced Treg-associated programs and attenuation of cytotoxic effector drivers"; LAG3 expression in plasma-neighborhood-like T cells was independent of cytogenetics but increased at relapse versus diagnosis (LFC 0.73).
- (b, extramedullary) In EM lesions, T cells in plasma-rich neighborhoods showed "a clear depletion of cytotoxic programs" and depletion of cytotoxicity-supporting cytokines and chemokines; a non-canonical Wnt receptor was enriched in EM lesions but not in marrow.
- (c, prognostic signature) Elastic-net Cox regression on 328 survival-associated spatially differentially expressed genes yielded a 15-gene signature spanning plasma and immune lineages; discovery cohort N = 260, HR = 2.003 (Cox p = 4.4 x 10^-x, exponent lost in extraction; abstract states p < 0.0001); independent validation N = 71, HR = 1.484, p = 0.0262; models adjusted for age, BMI, frontline ASCT, and site-batch.

---

## 3. Bingham et al. 2026, Blood Advances

**Verified citation.** According to PubMed: Bingham, N. E., Boiko, J. R., Jones, D. C., Wong, D., Khong, T., Mithraprabhu, S., Ensbey, K. S., Elz, A. E., Newell, E. W., Spencer, A., and Hill, G. R. (2026). Spatial transcriptomics identifies a suppressive, T-cell-excluded tumor microenvironment in extramedullary myeloma. Blood Advances, 10(18), 6221-6235. [DOI 10.1182/bloodadvances.2026020500](https://doi.org/10.1182/bloodadvances.2026020500). PMID 42379250. PMC13584121. Alfred Health / Monash University (Melbourne) and Fred Hutchinson Cancer Center (Seattle).

**Status.** Peer reviewed, open access on PMC. Issue date 2026-09-22.

**Platform and design (from PMC full text).** Xenium Human 5K Prime panel on FFPE biopsies of hematogenous extramedullary disease only; 8 patients; 500,102 cells; Proseg probabilistic segmentation; median 61,000 cells per sample, median 412 transcripts and 307 genes per cell; Seurat SCTransform, manual annotation, CellPhoneDB v5 for ligand-receptor inference. Niches defined by nearest-neighbor composition clustering.

**Verified findings.**
- (b, immune suppression) "The TME was dominated by macrophages with a suppressive 'M2' phenotype. CD8+ T cells frequently expressed exhaustion markers LAG3 and TIGIT, consistent with a suppressive TME" (abstract). Macrophages averaged 37% of non-tumor cells, T cells 23.7%, fibroblasts about 20%.
- (b, T-cell exclusion) "We identified 3 recurrent TME niches based on immune cell composition: immune excluded, immune suppressed, and immune permissive. The immune-excluded niche comprised the bulk of tumors, reflecting spatial exclusion of immune cells" (abstract). The immune-permissive niche appeared in only 2 of 8 samples.
- (b, checkpoint pattern) TIGIT and LAG3 were more prevalent than PDCD1 (median TIGIT:PDCD1 ratio 1.78, LAG3:PDCD1 ratio 1.62); inhibitory-receptor burden per T cell was low (median 1.22), "suggestive of a suppressive checkpoint landscape rather than a PD-1-dominant, chronic antigen-driven exhaustion."
- (a, niche signaling) "PC modulated the TME through PGE2 and VEGFB while receiving canonical prosurvival signals through CD38, CXCR4, and BCMA" (abstract). Fibroblast-derived ECM in the immune-suppressed niche "may further contribute to T-cell exclusion."
- (a, niche concept) The findings highlight "niche substitution rather than niche independence" in extramedullary disease (abstract); EM plasma cells retain a classic PC transcriptome with no dedifferentiation.

---

## 4. Li et al. 2026, Nature Communications (SpatialCOC)

**Verified citation.** According to PubMed: Li, M., Sun, P., Luo, Y., Zhou, G., Yang, X., Meng, D., and Ye, K. (2026). SpatialCOC: an integrative framework for spatial continuous mapping and cross-omics correction in spatial multi-omics data. Nature Communications, 17(1). [DOI 10.1038/s41467-026-71882-2](https://doi.org/10.1038/s41467-026-71882-2). PMID 41991905. PMC13265925. Xi'an Jiaotong University (Kai Ye lab). Article number not shown in the PubMed record (UNVERIFIED).

**Status.** Peer reviewed, open access, published 2026-04-16.

**What the method does (one paragraph).** SpatialCOC is a two-stage, unsupervised, "top-down" framework for vertical integration of spatial multi-omics (RNA plus protein, or RNA plus chromatin) measured on the same spots. Stage one, the Spatial Continuous Mapping (SCM) module, treats spatial coordinates as prior knowledge and fits an implicit neural representation (a SIREN-style network with three sine-activated layers, optionally with a Gaussian kernel smoothness regularizer) that maps any coordinate to the expression of each omics layer, producing spatially continuous, resolution-independent reconstructions of each modality. Stage two, the Cross-Omics Correction (COC) module, feeds these reconstructions through modality-specific autoencoders and a linear canonical correlation analysis layer, jointly minimizing a reconstruction loss and a correction (correlation-maximizing) loss. The final spot embeddings are the CCA-aligned latent representations, which are used for spatial domain identification (Leiden or mclust) and trajectory inference (PAGA). It was benchmarked against SpatialGlue, COSMOS, Seurat WNN, MultiVI, MultiMAP, SpaGCN and STAGATE on augmented human lymph node data, mouse brain (spatial ATAC-RNA-seq and CUT&Tag-RNA-seq), mouse spleen (SPOTS) and mouse thymus (Stereo-CITE-seq).

**What "cross-omics correction" means technically.** The authors contrast "correction" with the "weighted summation" used by graph-based methods. In COC, each modality's latent embedding is pushed, through a DCCAE-inspired objective, to maximize canonical correlation with the other modality while a reconstruction loss keeps modality-specific signal. Variation in one modality that has no counterpart in the other is treated as modality-specific noise and is removed from the shared embedding, so the modalities "mutually correct" each other rather than being averaged. The paper reports that this also removes batch effects between replicate slices within a single slice without explicit batch modeling (PAGA graph cosine similarity across thymus replicates 0.975 for SpatialCOC versus 0.942 for STAGATE).

**Verified findings (c, spatial methods).**
- Moran's I on mouse brain rose from 0.57 before reconstruction to 0.94 after SCM reconstruction; SpatialCOC mean Moran's I 0.905 versus SpatialGlue 0.785 for domain identification.
- In mouse spleen, SpatialCOC recognized vesicular germinal-centre structure while remaining smooth in macrophage-rich regions; SpaGCN and STAGATE each achieved only one of the two.
- Robust to added Gaussian and dropout noise on the HLN-augmented benchmark where MultiVI collapsed (ARI 0.05 to 0.08).
- Limitation stated by the authors: vertical integration only (all modalities measured on the same cell or spot); horizontal and mosaic integration across sections are future work.

**Relevance caution.** SpatialCOC was not applied to bone marrow or myeloma. Any proposal citing it should present it as a candidate integration method, not as myeloma evidence.

---

## 5. Koops et al. 2026, Blood Cancer Discovery

**Verified citation.** According to PubMed: Koops, M., Bertamini, L., Papazian, N., Korst, C., Budai, M. J., Hoogenboezem, R. M., Sanders, M. A., van der Holt, B., van Duin, M., Broijl, A., Balogh, P., Zweegman, S., Raaijmakers, M. H. G. P., van de Donk, N. W. C. J., Kellermayer, Z., Sonneveld, P., and Cupedo, T. (2026). Spatial profiling identifies a distinct and topographically defined tumor microenvironment that emerges during multiple myeloma evolution. Blood Cancer Discovery, 7(6), 999-1016. [DOI 10.1158/2643-3230.BCD-26-0080](https://doi.org/10.1158/2643-3230.BCD-26-0080). PMID 42678098. Erasmus MC Cancer Institute (Cupedo lab), Amsterdam UMC, University of Pecs, HOVON Foundation.

**Status.** Peer reviewed; PubMed issue date 2026-11-03. No PMC full text yet. Companion GEO record GSE345302 (Homo sapiens, listed 2026-08-28) carries the same title and abstract.

**Platform.** The abstract states "Using spatial transcriptomics and imaging mass cytometry." The specific spatial transcriptomics platform (Visium, Visium HD, Xenium or other) is UNVERIFIED: it is not named in the PubMed abstract or the GEO/OmicsDI listing, and the full text was not accessible. It is not CODEX/PhenoCycler; the imaging modality is imaging mass cytometry.

**Verified findings (all from the abstract).**
- (a, plasma-cell niches) "Myeloma cells initially exist as scattered cells throughout the marrow, yet with disease evolution condense into a spatially defined TME."
- (a, stroma and b, immune) "Dense tumor nodules contain a cellular ecosystem distinct from the surrounding marrow, enriched for macrophages, conventional dendritic cells, and CD8+ T cells, as well as endothelial cells and THY1+ mesenchymal stromal cells."
- (b) "Neutrophil lineage cells are absent from this TME but interact with scattered myeloma cells outside of dense nodules."
- (prognosis) "The presence of the dense tumor architecture at diagnosis confers a worse prognosis."
- Note: this abstract reports CD8+ T-cell enrichment inside nodules; it does not state exhaustion or exclusion. Do not cite it for T-cell exclusion.

**Related but distinct work found during the search (not one of the ten; cited for context only).** Baertsch et al., ASH 2023 oral abstract, "Spatial Dissection of the Bone Marrow Microenvironment in Multiple Myeloma by High Dimensional Multiplex Tissue Imaging": CODEX 60-marker panel, 489 trephines (12 controls, 13 MGUS, 20 SMM, 349 NDMM from a phase III trial, 22 relapsed), about 3.1 million cells; a plasma-cell plus exhausted-CD8 neighborhood was associated with unfavourable prognosis. This is the Heidelberg / Stanford (Weinhold, Raab, Nolan) group and may be the CODEX study the question had in mind. Its journal publication status was not checked.

---

## 6. Rasche et al. 2022, Nature Communications

**Verified citation.** According to PubMed: Rasche, L., Schinke, C., Maura, F., Bauer, M. A., Ashby, C., Deshpande, S., Poos, A. M., Zangari, M., Thanendrarajan, S., Davies, F. E., Walker, B. A., Barlogie, B., Landgren, O., Morgan, G. J., van Rhee, F., and Weinhold, N. (2022). The spatio-temporal evolution of multiple myeloma from baseline to relapse-refractory states. Nature Communications, 13(1), 4517. [DOI 10.1038/s41467-022-32145-y](https://doi.org/10.1038/s41467-022-32145-y). PMID 35922426. PMC9349320. Published 2022-08-03. University of Arkansas for Medical Sciences Myeloma Center and University Hospital Heidelberg (Weinhold).

**Status.** Peer reviewed. The work is published in Nature Communications, not Blood. The Research Square identifier rs-1177985 is UNVERIFIED: a search for the identifier returned no matching record, and the Crossref API was not reachable through the session proxy. Research Square is the preprint service for Nature portfolio "In Review" manuscripts, so an rs- preprint preceding this August 2022 paper is plausible, but the mapping of that specific number to this paper was not confirmed.

**Design.** Spatial-longitudinal whole-exome sequencing of CD138-purified cells, 140 samples (92 iliac crest, 42 CT-guided focal lesion, 4 soft tissue, 2 plasma cell leukemia) from 24 patients over up to 14 years, with PET-CT and DWIBS imaging guidance; melphalan signature SBS-MM1 used as a barcode for single-cell expansions.

**Verified findings.**
- (c, spatial sampling) Paired left and right iliac crest samples were highly homogeneous (median 98% shared mutations), whereas paired iliac crest and focal lesion samples differed (median 79% shared, p = 0.001): "significant spatial genomic heterogeneity is associated with focal lesions."
- Three evolutionary patterns: "relapse driven by a single-cell expansion, competing/co-existing sub-clones, and unique sub-clones at distinct locations" (abstract), the last termed "alternating spatial clonal dominance."
- "In patients with >= 3 focal lesions on positron-emission-tomography at diagnosis, relapse is driven by multiple distinct sub-clones, whereas in other patients, a single-cell expansion is typically seen (p < 0.01)."
- "We show a close phylogenetic relationship between baseline focal lesions and relapse disease, highlighting focal lesions as hotspots of tumor evolution."
- "Resistant sub-clones ... can be hidden over years," so curative therapy must overcome "not only tumor heterogeneity but also dormancy."
- Relevance caution: this is genomic (WES) spatial heterogeneity at the anatomical scale, not tissue-level spatial transcriptomics, and it does not address Wnt, CAM-DR or T cells.

---

## 7. Pilcher et al. 2026, Nature Cancer (MMRF Immune Atlas)

**Verified citation.** According to PubMed: Pilcher, W. C., Yao, L., Gonzalez-Kozlova, E., Pita-Juarez, Y., Karagkouni, D., Acharya, C. R., Michaud, M. E., Hamilton, M., Nanda, S., Song, Y., Sato, K., Wang, J. T., Satpathy, S., Ma, Y., Schulman, J., D'Souza, D., Jayasinghe, R. G., Ohlstrom, D., Ferguson, K. E., Cheloni, G., ..., Mulligan, G., Ding, L., Gnjatic, S., Vlachos, I. S., and Bhasin, M. (2026). A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma. Nature Cancer, 7(1), 224-246. [DOI 10.1038/s43018-025-01072-4](https://doi.org/10.1038/s43018-025-01072-4). PMID 41514053. PMC12858409. Published online 2026-01-09 (the DOI prefix is 2025; the volume/issue is 2026, so "Nature Cancer 2025" in the question should read 2026 for the issue, with online acceptance in late 2025).

**Status.** Peer reviewed, open access on PMC. Multi-site MMRF consortium (Georgia Tech / Emory Bhasin lab, Washington University Ding lab, Mount Sinai Gnjatic lab, BIDMC / Broad Vlachos lab, Mayo Clinic, MMRF).

**Design (from full text).** CD138-negative bone marrow aspirate scRNA-seq from the MMRF CoMMpass study: discovery cohort 263 NDMM participants (1,149,344 cells), validation cohort 74 NDMM participants (247,928 cells); total 1,397,272 cells from 337 participants. Rapid progressors (progression within 18 months) versus non-progressors (durable remission >= 4 years) were compared. Baseline composition: T cells (30.51% CD8, 23.39% CD4), NK 6.82%, B 8.51%, myeloid 12.20%, erythroid 7.87%, plasma cells 9.17%.

**Data availability.** The PMC extraction and the PMC web page retrieved in this session did not include the "Data availability" section, so the paper's own statement is UNVERIFIED from the primary text. Secondary sources: the Ohlstrom preprint states the Immune Atlas scRNA-seq is "available under controlled access at MMRF's VLAB shared resource" with a processed Seurat object at Zenodo 10.5281/zenodo.15025566; a Targeted Oncology commentary (2026-01-21) states the atlas is on the MMRF Virtual Lab portal with a single-cell Shiny app and downloadable formats. The CoMMpass registry is NCT01454297.

**Verified findings.**
- (b, immune dysregulation and cytogenetics) "Cytogenetic risk-based analysis revealed heterogeneous associations with T cells of BM, with 17p13 deletion showing distinct enrichment of a type 1 interferon signature" (abstract). Full text: t(4;14) and 1q21 gain participants were depleted of IFN-responsive CD4 memory populations, while 17p13-deleted participants were enriched for CD4 Teff memory IFN-I cells (p = 0.007).
- (b, T-cell exhaustion) High-risk cytogenetics were associated with T cell dysfunction "measured as a function of the expression of exhaustion or senescence markers" in a differentiated cytotoxic CD8 population; t(4;14) showed higher exhaustion marker expression among CD8 T cells, primarily in GZMK+ early activated effector memory cells.
- (b, senescence) "The disease progression-based analysis revealed the presence of a proinflammatory immune senescence-associated secretory phenotype in rapidly progressing participants" (abstract).
- (a, niche signaling) "Signaling analyses suggested active intercellular communication involving a proliferation-inducing ligand [APRIL] and B cell maturation antigen, potentially promoting tumor growth and survival" (abstract). Full text: APRIL was most strongly associated with a macrophage/monocyte population enriched in rapid progressors; BAFF with IFN-associated monocytes enriched in non-progressors.
- (outcomes) "Using independent discovery and validation cohorts, we demonstrated that integrating immune cell signatures with known tumor cytogenetics and individual characteristics significantly improves stratification for the prediction of survival" (abstract). 32 high-risk participants were non-progressors and 19 standard-risk participants were rapid progressors, motivating immune features beyond cytogenetics.
- Relevance caution: aspirate-based scRNA-seq with no spatial coordinates; Wnt and CAM-DR are not analyzed.

---

## 8. Xiang, Xi, Hasson and Ramadge 2009, NIPS

**Verified citation.** Xiang, Z. J., Xi, Y. T., Hasson, U., and Ramadge, P. J. (2009). Boosting with spatial regularization. In Advances in Neural Information Processing Systems 22 (NIPS 2009), pp. 2107-2115. Verified from the paper's title page in the NeurIPS proceedings PDF (papers.neurips.cc, hash f2217062e9a397a1dca429e7d70bc6ca) and the Princeton Collaborate record.

**Author order.** The paper's title page in the NeurIPS 2009 proceedings PDF and the Princeton Collaborate record list the authors as Zhen James Xiang, Yongxin Taylor Xi, Uri Hasson and Peter J. Ramadge (Advances in Neural Information Processing Systems 22, pp. 2107-2115). The papers.neurips.cc abstract page displays a different order (Xi, Hasson, Ramadge, Xiang), which is a metadata artefact; cite as Xiang et al. 2009.

**Verified content (paraphrase of the abstract).** The paper adds a spatial regularization kernel to the boosting loss, giving an efficient algorithm that places additional weights or priors on base classifiers; it proves a "grouping effect" that encourages selection of spatially local, discriminative base classifiers, motivated by fMRI voxel selection. Not biomedical-spatial-omics; PubMed does not index it. Relevance to a proposal: it is a legitimate precedent for spatially regularized feature selection, which is the methodological analogue of selecting spatially coherent gene or morphology features.

---

## 10. Bandyopadhyay et al. 2024, Cell

**Verified citation.** According to PubMed: Bandyopadhyay, S., Duffy, M. P., Ahn, K. J., Sussman, J. H., Pang, M., Smith, D., Duncan, G., Zhang, I., Huang, J., Lin, Y., Xiong, B., Imtiaz, T., Chen, C.-H., Thadi, A., Chen, C., Xu, J., Reichart, M., Martinez, Z., Diorio, C., Chen, C., Pillai, V., Snaith, O., Oldridge, D., Bhattacharyya, S., Maillard, I., Carroll, M., Nelson, C., Qin, L., and Tan, K. (2024). Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging. Cell, 187(12), 3120-3140.e29. [DOI 10.1016/j.cell.2024.04.013](https://doi.org/10.1016/j.cell.2024.04.013). PMID 38714197. PMC11162340. Preprint: bioRxiv 2024.03.14.585083, PMID 38559168, [DOI 10.1101/2024.03.14.585083](https://doi.org/10.1101/2024.03.14.585083). University of Pennsylvania and Children's Hospital of Philadelphia (Kai Tan and Ling Qin labs).

**Status.** Peer reviewed, open access on PMC. Published 2024-05-06 (online), issue June 2024.

**Marker panel size.** CODEX with a 54-plex panel: 53 antibodies plus DAPI (full text: "we designed and validated a 54-plex panel (53 antibodies + DAPI)"; the abstract and preprint say "53-antibody panel"). A second, smaller ligand-receptor CODEX panel was used for validation with 21 overlapping features.

**Design.** scRNA-seq of 29,325 non-hematopoietic and 53,417 hematopoietic cells from fresh femoral heads (12 of 16 donors aged 52 to 80, hip replacement); whole-slide CODEX on 12 specimens (8 overlapping), over 1.2 million cells (abstract; the preprint says over one million); 32 cell types captured spatially; 15 cellular neighborhoods; applied to 3 diagnostic and 2 post-therapy AML iliac crest biopsies plus 3 negative lymphoma staging marrows.

**Data availability (from full text).** scRNA-seq at NCBI GEO GSE253355 (OmicsDI record confirms the title and 2024-04-24 submission); primary CODEX images and processed CODEX data on FigShare; code on GitHub. Exact FigShare and GitHub URLs were stripped in the text extraction and were not re-verified.

**Verified findings.**
- (c, spatial methods) "We employed co-detection by indexing (CODEX) to spatially profile over 1.2 million cells. We integrated scRNA-seq and CODEX data to link predicted cellular signaling with spatial proximity" (abstract). Label transfer by Seurat RPCA from the atlas to new images.
- (a, niches) "A hyperoxygenated arterio-endosteal neighborhood for early myelopoiesis, and an adipocytic localization for early hematopoietic stem and progenitor cells (HSPCs)" (abstract); adipocytic, not endosteal or perivascular, HSPC niche.
- (a, stroma) Nine transcriptionally distinct non-hematopoietic subtypes, including Adipo-MSC, THY1+ MSC, Fibro-MSC and osteoblast lineages; all MSCs lay near the arterio-endosteal niche, but Adipo-MSCs were near sinusoids while THY1+ MSCs were not.
- (disease application) "We used our CODEX atlas to annotate new images and uncovered mesenchymal stromal cell (MSC) expansion and spatial neighborhoods co-enriched for leukemic blasts and MSCs in acute myeloid leukemia (AML) patient samples" (abstract).
- Generalizability: femoral head neighborhoods correlated with iliac crest staging marrow neighborhoods (median neighborhood R = 0.71).
- Relevance caution: healthy and AML marrow only; no myeloma; no T-cell exhaustion analysis.

---

## Claim ledger

Strength scale: established = supported by two or more peer-reviewed studies or a large peer-reviewed cohort; single-study = one peer-reviewed study; preprint = supported only by a non-peer-reviewed preprint.

| # | Claim a proposal could make | Supporting citation(s) | Strength | Notes and limits |
|---|---|---|---|---|
| C1 | Subcellular spatial transcriptomics (Xenium 5K) is feasible on FFPE bone marrow trephines when fixation and decalcification are optimized. | Yip et al. 2025 | single-study | Bingham et al. 2026 used Xenium 5K on EM soft tissue, not marrow. |
| C2 | Malignant plasma cells in newly diagnosed MM form spatially restricted subpopulations with distinct transcriptional profiles (seen in 5 of 10 NDMM trephines). | Yip et al. 2025 | single-study | Small cohort (10 NDMM). |
| C3 | There is no universal, conserved myeloma-supportive bone marrow microenvironment; hematopoietic and stromal composition varies between and within trephines. | Yip et al. 2025; supported in EM by Bingham et al. 2026 (interpatient heterogeneity, "niche substitution") | established (two peer-reviewed spatial studies, different compartments) | Koops et al. 2026 reports a recurrent dense-nodule ecosystem, so the field is not settled; state both. |
| C4 | Stromal cell composition does not shift consistently from control to MGUS/SMM to MM at the whole-sample level. | Yip et al. 2025 | single-study | Koops et al. 2026 reports THY1+ MSC enrichment inside dense nodules; reconcile at the neighborhood scale. |
| C5 | With disease evolution, myeloma cells condense from scattered cells into dense nodules with a distinct ecosystem (macrophages, cDCs, CD8+ T cells, endothelium, THY1+ MSCs), and dense architecture at diagnosis predicts worse outcome. | Koops et al. 2026 | single-study | Abstract-level evidence only in this ledger; ST platform unverified. |
| C6 | Neutrophil-lineage cells are excluded from dense myeloma nodules but interact with scattered myeloma cells. | Koops et al. 2026 | single-study | |
| C7 | Plasma-cell-dense bone marrow niches are enriched for non-canonical Wnt signaling and for ligands implicated in cell adhesion-mediated drug resistance (CAM-DR). | Ohlstrom et al. 2026 (bioRxiv) | preprint | Gene symbols not recoverable from PMC extraction; cite the preprint for pathway-level claims only. |
| C8 | Spatially derived plasma and immune transcriptional programs predict PFS: a 15-gene signature gives HR 2.00 (discovery N = 260) and HR 1.48 (validation N = 71). | Ohlstrom et al. 2026 (bioRxiv) | preprint | Signature derived on aspirate scRNA-seq after spatial gene selection, not directly on spatial data. |
| C9 | T cells inside plasma-cell-dense niches show suppressed states: elevated LAG3, Treg-like programs, reduced cytotoxic effector programs; LAG3 rises at relapse. | Ohlstrom et al. 2026 (bioRxiv); consistent with Bingham et al. 2026 (LAG3, TIGIT on CD8 T cells in EM) | preprint for marrow; single-study (peer reviewed) for EM | |
| C10 | Extramedullary myeloma has a suppressive, T-cell-excluded TME dominated by M2-like macrophages, with three recurrent niches (immune excluded, immune suppressed, immune permissive) and the excluded niche forming the bulk of tumors. | Bingham et al. 2026 | single-study | 8 patients, hematogenous EMD only. |
| C11 | In EM myeloma the T-cell checkpoint pattern is TIGIT- and LAG3-dominant rather than PD-1-dominant, suggesting microenvironment-driven suppression rather than chronic antigen-driven exhaustion. | Bingham et al. 2026 | single-study | Authors note checkpoint expression alone does not prove dysfunction. |
| C12 | Canonical prosurvival signaling to myeloma cells (APRIL/BAFF to BCMA, CD38, CXCR4) is active in the tumor microenvironment. | Bingham et al. 2026 (EM, spatial ligand-receptor); Pilcher et al. 2026 (marrow aspirate, APRIL and BCMA communication) | established (two peer-reviewed studies, different compartments and modalities) | Both are inferred from expression, not functional assays. |
| C13 | In NDMM, immune microenvironment features (T-cell exhaustion or senescence, type I interferon signatures, an iSASP-like phenotype in rapid progressors) carry prognostic information independent of cytogenetics, and combining them improves survival stratification. | Pilcher et al. 2026 | single-study (large: 337 participants, discovery plus validation) | Not spatial. |
| C14 | High-risk cytogenetic subgroups have distinct immune contexts: t(4;14) with CD8 exhaustion markers; 17p13 deletion with type I interferon enrichment. | Pilcher et al. 2026 | single-study | |
| C15 | Spatial genomic heterogeneity in myeloma is concentrated in focal lesions, which are hotspots of evolution and phylogenetically close to relapse clones; iliac crest biopsies are comparatively homogeneous. | Rasche et al. 2022 | single-study | WES, 24 patients. |
| C16 | Patients with >= 3 PET-positive focal lesions relapse from multiple subclones, others typically from a single-cell expansion. | Rasche et al. 2022 | single-study | p < 0.01, small cohort. |
| C17 | A reference atlas of healthy human bone marrow defines an arterio-endosteal early-myeloid neighborhood and a peri-adipocytic HSPC niche, with nine non-hematopoietic subtypes including THY1+ MSCs. | Bandyopadhyay et al. 2024 | single-study (Cell, 12 donors) | Older donors (52 to 80). |
| C18 | A healthy-marrow CODEX atlas can be used by label transfer to annotate disease marrow and detect stromal remodeling (shown for AML). | Bandyopadhyay et al. 2024 | single-study | Not shown for myeloma. |
| C19 | THY1+ mesenchymal stromal cells are a recurring stromal element of both healthy marrow niches and myeloma dense-nodule ecosystems. | Bandyopadhyay et al. 2024; Koops et al. 2026 | established (two peer-reviewed studies) | Same marker, different diseases; functional link not shown. |
| C20 | Spatial multi-omics layers can be integrated by learning spatially continuous per-modality representations and then aligning modalities by cross-correlation (cross-omics correction), which removes modality-specific noise and batch effects better than weighted-summation graph methods. | Li et al. 2026 (SpatialCOC) | single-study | Benchmarked on brain, spleen, thymus, lymph node; not marrow. |
| C21 | Spatial regularization of boosting encourages selection of spatially local, discriminative features (grouping effect), a precedent for spatially coherent feature selection. | Xiang et al. 2009 | established (standard ML reference) | Non-biomedical. |
| C22 | Integrating sequencing-based whole-transcriptome spatial data (Visium HD), imaging-based single-cell spatial data (Xenium) and clinically annotated scRNA-seq is a workable framework for niche discovery, validation and outcome association in myeloma. | Ohlstrom et al. 2026 (bioRxiv), building on Yip et al. 2025 and Pilcher et al. 2026 | preprint (framework); components peer reviewed | |

---

## Verification summary

Verified from primary sources: items 1, 3, 4, 6 (published version), 7, 8, 10, and the combined 2/9 preprint.

Partially verified or corrected:
- Item 2 and item 9 are one work (Ohlstrom et al., bioRxiv 2026, Bhasin lab). The bioRxiv API shows no journal publication yet. "Karimpour-Fard" is not an author of this preprint.
- Item 5 (Koops et al.) is verified as published in Blood Cancer Discovery; the spatial transcriptomics platform is not named in available records (imaging modality is imaging mass cytometry, not CODEX).
- Item 6: the Nature Communications paper is verified; the Research Square number rs-1177985 could not be matched (search returned nothing, Crossref blocked by the proxy).
- Item 7: published in the 2026 volume of Nature Cancer (online 2026-01-09) with a 2025-prefixed DOI; the paper's own data availability statement was not in the retrieved text (secondary sources point to MMRF VLAB and Zenodo 10.5281/zenodo.15025566).
- Item 8: the proceedings PDF title page and the Princeton Collaborate record give Xiang, Xi, Hasson, Ramadge, pp. 2107-2115; the abstract page's order (Xi, Hasson, Ramadge, Xiang) is a metadata artefact.

Not verifiable in this session: Nature Communications article number for SpatialCOC; FigShare and GitHub URLs for Bandyopadhyay et al.; the Ohlstrom GEO accession (embargoed until publication).

---

## References (APA)

Bandyopadhyay, S., Duffy, M. P., Ahn, K. J., Sussman, J. H., Pang, M., Smith, D., Duncan, G., Zhang, I., Huang, J., Lin, Y., Xiong, B., Imtiaz, T., Chen, C.-H., Thadi, A., Chen, C., Xu, J., Reichart, M., Martinez, Z., Diorio, C., Chen, C., Pillai, V., Snaith, O., Oldridge, D., Bhattacharyya, S., Maillard, I., Carroll, M., Nelson, C., Qin, L., & Tan, K. (2024). Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging. Cell, 187(12), 3120-3140.e29. https://doi.org/10.1016/j.cell.2024.04.013

Bingham, N. E., Boiko, J. R., Jones, D. C., Wong, D., Khong, T., Mithraprabhu, S., Ensbey, K. S., Elz, A. E., Newell, E. W., Spencer, A., & Hill, G. R. (2026). Spatial transcriptomics identifies a suppressive, T-cell-excluded tumor microenvironment in extramedullary myeloma. Blood Advances, 10(18), 6221-6235. https://doi.org/10.1182/bloodadvances.2026020500

Koops, M., Bertamini, L., Papazian, N., Korst, C., Budai, M. J., Hoogenboezem, R. M., Sanders, M. A., van der Holt, B., van Duin, M., Broijl, A., Balogh, P., Zweegman, S., Raaijmakers, M. H. G. P., van de Donk, N. W. C. J., Kellermayer, Z., Sonneveld, P., & Cupedo, T. (2026). Spatial profiling identifies a distinct and topographically defined tumor microenvironment that emerges during multiple myeloma evolution. Blood Cancer Discovery, 7(6), 999-1016. https://doi.org/10.1158/2643-3230.BCD-26-0080

Li, M., Sun, P., Luo, Y., Zhou, G., Yang, X., Meng, D., & Ye, K. (2026). SpatialCOC: An integrative framework for spatial continuous mapping and cross-omics correction in spatial multi-omics data. Nature Communications, 17(1). https://doi.org/10.1038/s41467-026-71882-2

Ohlstrom, D. J., Michaud, M. E., Bakhtiari, M., Vieira Dos Santos, J., Pilcher, W. C., Staub, A., Satpathy, S., Ferguson, K., Mantrala, S., Kim-Schulze, S., Chen, Z., Lonial, S., Kemp, M. L., Sherbenou, D., Lagana, A., Jaye, D. L., Nooka, A. K., Parekh, S., & Bhasin, M. K. (2026). Spatial multi-omics of multiple myeloma uncovers niche-dependent pro-myeloma and immunosuppressive signaling in the bone marrow and extramedullary lesions [Preprint]. bioRxiv. https://doi.org/10.64898/2026.03.23.713195

Pilcher, W. C., Yao, L., Gonzalez-Kozlova, E., Pita-Juarez, Y., Karagkouni, D., Acharya, C. R., Michaud, M. E., Hamilton, M., Nanda, S., Song, Y., Sato, K., Wang, J. T., Satpathy, S., Ma, Y., Schulman, J., D'Souza, D., Jayasinghe, R. G., Ohlstrom, D., Ferguson, K. E., ... Bhasin, M. (2026). A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma. Nature Cancer, 7(1), 224-246. https://doi.org/10.1038/s43018-025-01072-4

Rasche, L., Schinke, C., Maura, F., Bauer, M. A., Ashby, C., Deshpande, S., Poos, A. M., Zangari, M., Thanendrarajan, S., Davies, F. E., Walker, B. A., Barlogie, B., Landgren, O., Morgan, G. J., van Rhee, F., & Weinhold, N. (2022). The spatio-temporal evolution of multiple myeloma from baseline to relapse-refractory states. Nature Communications, 13(1), 4517. https://doi.org/10.1038/s41467-022-32145-y

Xiang, Z. J., Xi, Y. T., Hasson, U., & Ramadge, P. J. (2009). Boosting with spatial regularization. In Advances in Neural Information Processing Systems 22 (pp. 2107-2115). https://papers.neurips.cc/paper_files/paper/2009/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract.html

Yip, R. K. H., Er, J., Qin, L., Nguyen, Q. H., Motyer, A., Rimes, J. S., Light, A., Mishi, R. D., Ling, L., Anttila, C. J. A., Tsui, E., Amann-Zalcenstein, D., Dowling, M. R., Rogers, K. L., Bowden, R., Chen, Y., Harrison, S. J., & Hawkins, E. D. (2025). Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens with spatial transcriptomics. Blood, 146(15), 1837-1849. https://doi.org/10.1182/blood.2025028896

### Context reference (not one of the ten works; found during verification)

Baertsch, M.-A., Brobeil, A., Hickey, J., Haist, M., Poos, A. M., Lu, G., Kuswanto, W., Schuerch, C., Voehringer, H., Huber, W., Mechtersheimer, G., Mueller-Tidow, C., Schirmacher, P., Weisel, K., Fenk, R., Goldschmidt, H., Goltsev, Y., Raab, M. S., Weinhold, N., & Nolan, G. P. (2023, December 9). Spatial dissection of the bone marrow microenvironment in multiple myeloma by high dimensional multiplex tissue imaging [Oral abstract]. 65th ASH Annual Meeting. https://ash.confex.com/ash/2023/webprogram/Paper189255.html

### Data records cited

GEO GSE299207 (SuperSeries) and GSE299193 (human SubSeries): Yip et al. 2025 Xenium data. GEO GSE345302: Koops et al. 2026. GEO GSE253355: Bandyopadhyay et al. 2024 scRNA-seq. Zenodo 10.5281/zenodo.15025566: processed MMRF Immune Atlas Seurat object (as cited by Ohlstrom et al. 2026).
