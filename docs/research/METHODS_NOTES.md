# Methods Notes: Spatial Proteomics Datasets, MANOVA Design, and SpatialCOC

Prepared 2026-11-03. Sources were verified by web retrieval, PubMed (via the PubMed MCP tools) and, where noted, full-text retrieval from PubMed Central. Statements that could not be verified against a primary source are marked as such. Facts attributed to PubMed are marked "According to PubMed" and carry DOI links, as required by the PubMed terms of use.

---

## Q1. Public CODEX / PhenoCycler spatial proteomics datasets of human bone marrow and multiple myeloma

### 1.1 Summary table

| Dataset | Platform | Disease | Samples | Panel | Tissue | Public data | Paired transcriptomics on the same specimen |
|---|---|---|---|---|---|---|---|
| Bandyopadhyay et al. 2024, Cell | CODEX (PhenoCycler-Fusion) | Healthy adult BM; AML; negative staging marrow (NSM) | 12 healthy femoral heads; 5 AML biopsies from 3 patients; 3 NSM | 54-plex (53 antibodies + DAPI) | Healthy: fresh femoral head cores, PFA-fixed, EDTA-decalcified, paraffin-embedded. AML/NSM: clinical FFPE iliac crest trephines (Formical decalcified) | Yes. Figshare+ collection doi:10.25452/figshare.plus.c.7174914 (full multiplexed images, segmentation masks, H&E, annotated Seurat objects with per-cell marker values); scRNA-seq GEO GSE253355 | Partially. scRNA-seq and CODEX overlap on 8 of the healthy femoral heads, but scRNA-seq was from dissociated cells, not from the imaged section. No spatial transcriptomics. |
| Baertsch et al., ASH 2023 oral abstract (Heidelberg/DKFZ/Stanford) | CODEX (Akoya) | MGUS, SMM, NDMM, relapsed MM | 416 TMA cores retained of 489 (12 non-malignant, 13 MGUS, 20 SMM, 349 NDMM, 22 relapsed), ca. 3.1 million cells | 60-marker panel; GRZB, Ki67, PD1, CD206, CD138, CD4, CD8 named | BM trephine biopsies in TMA format (FFPE not stated explicitly) | Not stated; no peer-reviewed paper or data deposit located as of 2026-11-03 | No transcriptomics; paired with FISH (n=309) and WGS (n=212) |
| Soekojo et al., ASH 2024 abstract 1880 (Mayo Clinic) | CODEX | High-risk SMM with paired progression to MM | 6 patients, paired FFPE BM at SMM diagnosis and at progression | 19 markers listed (CD3e, CD4, CD8, CD14, CD20, CD31, CD38, CD45, CD66a, CD68, CD138, HLA-DR, ASCT2, pNRF2, GLUT1, LDHA, ATPA5, CPT1A, citrate synthase) | FFPE bone marrow | Not stated | Not mentioned |
| Roseth et al. 2026, Blood Cancer Discovery | Imaging mass cytometry (IMC, Hyperion), not CODEX | NDMM (n=65), SMM (n=6), MGUS (n=5) | 76 patients, 2 ROIs of 1 mm each, 223,655 annotated cells | 35-antibody panel (includes CD138, CD38, IRF4, CD3, CD4, CD8, CD68, HLA-DR, Ki67, GZMK, HIF1A, IDO, metabolic markers; FOXP3/LAG3/PD-1/SMA/collagen not mentioned in text) | FFPE iliac crest trephines, EDTA decalcified | Code on GitHub (URL stripped in PMC text); raw image deposit not stated in the retrieved text | No |
| Koops et al. 2026, Blood Cancer Discovery (Erasmus MC) | Spatial transcriptomics plus IMC, not CODEX | MGUS/SMM/NDMM evolution (counts not retrieved) | Not retrieved | Not retrieved | Bone marrow (trephine assumed, not confirmed) | GEO GSE345302 (spatial transcriptomics) | Yes, ST and IMC in the same study; whether on the same section was not confirmed |
| Yip et al. 2025, Blood (WEHI) | Xenium Prime 5K (spatial transcriptomics), no multiplexed protein imaging | 4 controls, 2 MGUS, 5 SMM, 10 NDMM | 21 trephines | 5001 genes | FFPE posterior superior iliac spine trephines, nondecalcified or brief EDTA | GEO GSE299193 (human), GSE299195 (mouse) | No protein imaging; CD138 IHC on adjacent sections only |

### 1.2 Bandyopadhyay et al. 2024 (the Tan lab human bone marrow CODEX atlas)

According to PubMed, the peer-reviewed article is Bandyopadhyay S, Duffy MP, Ahn KJ, et al. "Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging." Cell 2024;187(12):3120-3140.e29, [DOI 10.1016/j.cell.2024.04.013](https://doi.org/10.1016/j.cell.2024.04.013) (PMID 38714197, PMC11162340). The preprint is [DOI 10.1101/2024.03.14.585083](https://doi.org/10.1101/2024.03.14.585083) (PMID 38559168). Details below were taken from the PMC full text.

- Samples. Bone marrow was collected from fresh femoral heads of 16 individuals aged 52 to 80 undergoing hip replacement; scRNA-seq on 12, CODEX on 12, with 8 overlapping. The disease vignette adds 3 diagnostic and 2 paired post-therapy AML iliac crest biopsies (3 patients) and 3 negative lymphoma staging marrows (NSM) from the iliac crest. In total 803,132 cells were annotated (32 cell types) in the healthy atlas; over 1.2 million cells overall.
- Tissue. Healthy samples are not trephines: cylindrical cores drilled from femoral heads, PFA-fixed, decalcified 8 days in 0.5 M EDTA, paraffin-embedded. The AML and NSM samples are standard clinical FFPE trephine biopsies (zinc formalin, Formical-4 decalcification), which makes them the closest analogue to myeloma trephines in this dataset.
- Panel. 54-plex (53 antibodies + DAPI). The full list is in Supplementary Tables S4 and S5 of the paper and was not retrievable here. From the main text and legends: CD34, CD38, CD56, CD3E, CD68, CD163, ASMA (alpha smooth muscle actin), CD138 (the Methods note nonspecific CD138 staining, implying it is in the panel; plasma cells were annotated using CD38 and CD79A), CD117, CD19, CD10, CD45, CD45RA, CD90, CD271, SPINK2, MPO, GATA1, CD71, GYPC, VE-cadherin, CD141, CD146, FOXC1, CXCL12, HIF1a, BCL2, Complex IV, NPM1c, PLP1, Vimentin, NaK-ATPase, mast cell tryptase. Not found in the text: CD4, CD8, FOXP3, LAG3, PD-1, collagen (Type I collagen is discussed as a structure but no antibody is named). CD4 and CD8 should be checked in Table S4 before relying on their absence.
- Data access. From the Data and Code Availability section: "Primary imaging data and processed CODEX data are publicly available and can be downloaded from FigShare", collection [doi:10.25452/figshare.plus.c.7174914](https://doi.org/10.25452/figshare.plus.c.7174914), containing "full multiplexed images for the normal bone marrow atlas, AML, NSM samples, corresponding cell segmentation masks and H&E images as well annotated Seurat objects which contain unnormalized and normalized marker expression values for each cell." The individual items are titled "Processed CODEX Data (Seurat Objects)", "Normal Bone Marrow CODEX Images" (figshare+ 25122311), "Normal Bone Marrow CODEX Segmentation Masks", "Normal Bone Marrow H&E Images", "AML and NSM CODEX Images", "AML and NSM CODEX Segmentation Masks", "AML and NSM H&E Images". Format: raw stitched multichannel images plus Seurat objects (cell-by-marker with centroid coordinates, extractable to CSV). scRNA-seq is in GEO GSE253355. Interactive Vitessce viewers are hosted by CHOP. The HuBMAP featured publications page lists this paper and points to the same figshare collection; no bone marrow CODEX dataset was found on the HuBMAP data portal itself by search.
- Pairing. scRNA-seq and CODEX share donors (8 of 12) but not sections; there is no spatial transcriptomics.

### 1.3 "Spatial Profiling Identifies a Distinct and Topographically Defined Tumor Microenvironment that Emerges during Multiple Myeloma Evolution"

According to PubMed, this is Koops M, Bertamini L, Papazian N, et al., Blood Cancer Discovery 2026;7(6):999-1016, [DOI 10.1158/2643-3230.BCD-26-0080](https://doi.org/10.1158/2643-3230.BCD-26-0080) (PMID 42678098; Erasmus MC, Amsterdam UMC, University of Pecs). The abstract states: "Using spatial transcriptomics and imaging mass cytometry, we show that myeloma cells initially exist as scattered cells throughout the marrow, yet with disease evolution condense into a spatially defined TME." The platform is therefore spatial transcriptomics plus IMC, not CODEX or PhenoCycler. The spatial transcriptomics data are deposited in GEO as GSE345302 (OmicsDI record dated 2026-08-28). The specific ST platform, sample counts per stage, and the location of the IMC images could not be retrieved (the AACR article page is not reachable from this session). These should be read from the paper's Data Availability section.

### 1.4 Other multiplexed-imaging myeloma studies

- Baertsch et al., ASH 2023 oral abstract 189255, "Spatial Dissection of the Bone Marrow Microenvironment in Multiple Myeloma By High Dimensional Multiplex Tissue Imaging" (Heidelberg, DKFZ, Stanford; CODEX; 60-marker panel; 416 TMA cores: 12 controls, 13 MGUS, 20 SMM, 349 NDMM, 22 relapsed; ca. 3.1 million cells). This is the largest myeloma CODEX cohort identified. No data deposit or peer-reviewed article was found as of 2026-11-03 (PubMed search for Baertsch/Bartsch with CODEX and myeloma returned no results). The International Myeloma Society Career Development Award page (2021-2022) describes the same project.
- Soekojo et al., ASH 2024 abstract 1880 (Mayo Clinic; CODEX; 6 high-risk SMM patients with paired progression samples; 19 markers including CD138, CD38, CD3e, CD4, CD8, CD68). No data deposit stated.
- According to PubMed, Roseth IA, Hatscher L, Schiller C, et al. "Multiplexed Imaging Reveals Immune-Metabolic Niches in Multiple Myeloma Linked to Progression and Bone Disease." Blood Cancer Discov 2026;7(3):460-478, [DOI 10.1158/2643-3230.BCD-25-0334](https://doi.org/10.1158/2643-3230.BCD-25-0334) (PMID 41734347, PMC13139845). IMC with a 35-antibody panel on FFPE iliac crest trephines from 65 NDMM, 6 SMM and 5 MGUS patients; two 1 mm ROIs per patient; 223,655 annotated cells; analysis scripts on GitHub. This is the closest published analogue to a per-stage multiplexed protein dataset, but it is IMC, not CODEX.
- No PhenoCycler or CODEX myeloma dataset from an "Ankur Sharma" laboratory was found by web search; the search returned only unrelated results. The Schurch laboratory (Tubingen) has a 2026 PhD thesis (Lanzhu Li) describing a 65-plex CODEX panel for human FFPE bone marrow TMAs in MDS (109 patients), not myeloma.

### 1.5 The Yip et al. 2025 dataset (the 21-trephine design used in this project)

According to PubMed, Yip RKH, Er J, Qin L, et al. "Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens with spatial transcriptomics." Blood 2025;146(15):1837-1849, [DOI 10.1182/blood.2025028896](https://doi.org/10.1182/blood.2025028896) (PMID 40643106, PMC12824660). From the PMC full text: Xenium Prime 5K (5001 genes) on 21 trephines (Ctrl-1 to Ctrl-4; MGUS-1, MGUS-2; SM-1 to SM-5; MM-1 to MM-10), posterior superior iliac spine, neutral buffered formalin, nondecalcified or brief EDTA decalcification; 15 to 255 genes detected per cell; H&E was stained on the same slide after Xenium and co-registered in Xenium Explorer. There is no multiplexed protein imaging; plasma cell location was validated by CD138 immunohistochemistry on adjacent sections. GEO: GSE299193 (human) and GSE299195 (mouse).

### 1.6 Conclusion

No public dataset was found in which CODEX (or PhenoCycler) and spatial transcriptomics were acquired on the same myeloma trephine. The closest public resources are: (i) Bandyopadhyay et al. 2024, which provides fully open CODEX images and per-cell tables of healthy femoral-head marrow plus a few AML and NSM trephines, with dissociated scRNA-seq from overlapping donors but no spatial transcriptomics and no myeloma; (ii) Koops et al. 2026, which combines spatial transcriptomics (GEO GSE345302) with IMC in myeloma, but not CODEX, and whose same-section pairing is unconfirmed; (iii) Yip et al. 2025, which provides Xenium 5K on 21 staged trephines with no protein imaging. The largest myeloma CODEX cohort (Baertsch et al.) remains unpublished and undeposited as of this date.

---

## Q2. MANOVA design for the 21-patient, four-program study

Setting: 21 patients (4 control, 2 MGUS, 5 smouldering, 10 myeloma), one trephine each. Per cell, a vector of program scores (wnt_core, adhesion_survival, pc_stress_survival, t_exhaustion, cytotoxicity; the composite wnt_camdr is omitted from the multivariate test because it is a function of its two subsets), measured by Xenium and predicted from DAPI nuclear morphology by cross-validation. The analysis below was written for the original four-program design; the pipeline as implemented collapses stages to control (4), precursor (7) and myeloma (10) and tests five variables, which keeps the error degrees of freedom (18) above the number of variables, and uses 999 patient-level permutations by default.

### 2.1 (a) Patient-level one-way MANOVA on stage

Design facts. Units: 21 patient mean vectors. Groups: g = 4. Dependent variables: p = 4. Hypothesis degrees of freedom: g - 1 = 3. Error degrees of freedom: N - g = 17. Because the pooled error SSCP matrix has 17 degrees of freedom and 17 > p, the pooled matrix is nonsingular and the four classical statistics are computable. However, the MGUS group (n = 2) has fewer cases than dependent variables, so its own covariance matrix is singular.

Assumptions (Tabachnick and Fidell, Using Multivariate Statistics, 7th ed., Pearson, 2019, chapter 7):
1. Independence of units. Satisfied at patient level if each patient contributes one vector.
2. Multivariate normality of the dependent variables within each group. With 21 units, formal tests (Mardia, Henze-Zirkler) have little power and are not decisive; Tabachnick and Fidell treat MANOVA as reasonably robust to modest non-normality when the smallest group has at least about 20 cases, which is not the case here.
3. Homogeneity of variance-covariance matrices. Box's M is the standard test, but it is highly sensitive to non-normality and requires each group's covariance matrix to be estimable (n_j > p). With n = 2 in MGUS, Box's M cannot be computed meaningfully. Tabachnick and Fidell advise that with unequal group sizes, if larger groups have larger variances the test is conservative, and if smaller groups have larger variances the test is liberal; this directional check is more informative here than Box's M.
4. Linearity among DVs and absence of multicollinearity. Program scores derived from overlapping gene sets may be correlated; report the within-group correlation matrix and avoid |r| > 0.9.

Test statistic. Pillai's trace is the recommended statistic when group sizes are unequal and when homogeneity of covariance may be violated. The basis is Olson's Monte Carlo work: Olson CL. "Comparative robustness of six tests in multivariate analysis of variance." Journal of the American Statistical Association 1974;69(348):894-908 (citation confirmed on the IBM SPSS recommended readings page), and the follow-up Olson CL. "On choosing a test statistic in multivariate analysis of variance." Psychological Bulletin 1976;83(4):579-586 (bibliographic details from standard references; full text not retrieved in this session). Olson found Pillai's trace to be the most robust of the four statistics to heterogeneity of covariance and to non-normality, while Roy's largest root was the least robust. Tabachnick and Fidell repeat this recommendation for small and unequal samples.

Minimum sample size. Tabachnick and Fidell state that the number of cases in every cell must exceed the number of dependent variables, and that larger samples are needed for the robustness results to apply (they discuss about 20 cases in the smallest cell for robustness to non-normality). Under these criteria, a 4-group MANOVA with 4 DVs needs at least 5 patients in every group to be estimable per group and at least 20 per group to be interpretable in the usual sense. The present design (4, 2, 5, 10) fails the per-group criterion for MGUS and controls and the robustness criterion for all groups. Power at these sizes is low except for very large effects.

Honest assessment. A four-group patient-level MANOVA with these group sizes is computable but not well-supported. The defensible options are: (i) collapse to three ordered groups, control (n = 4), precursor (MGUS + SMM, n = 7) and myeloma (n = 10), which is the grouping Yip et al. 2025 used for pseudobulk differential expression; (ii) present the MANOVA as exploratory and accompany it with a permutation test that does not rely on normality or on homogeneity of covariance; (iii) pre-register the hypothesis as a trend over ordered stage (a single contrast) rather than an omnibus four-group difference, which uses 1 rather than 3 hypothesis degrees of freedom.

### 2.2 (b) Cell-level MANOVA on hundreds of thousands of cells

Why a naive cell-level MANOVA is invalid. Cells within a trephine share the patient's genotype, disease stage, tissue processing and section, so they are not independent observations. Treating them as independent inflates the error degrees of freedom from about 17 to hundreds of thousands, making the standard error of every group mean far too small and the type I error rate far above nominal. This is pseudoreplication in the sense of Hurlbert (Ecological Monographs 1984;54:187-211). According to PubMed, Zimmerman KD, Espeland MA and Langefeld CD, "A practical solution to pseudoreplication bias in single-cell studies," Nature Communications 2021;12:738, [DOI 10.1038/s41467-021-21038-1](https://doi.org/10.1038/s41467-021-21038-1), document that cells from the same individual are "subsamples or pseudoreplicates", that ignoring this leads to "highly inflated type 1 error rates", and that batch correction for individual does not solve it. The effective sample size for a between-patient contrast is the number of patients, regardless of how many cells are measured.

What to do instead.
1. Patient as the unit (two-stage or pseudobulk). Average the 4-vector within each patient (optionally within plasma cells only, or within a defined compartment), then run the patient-level MANOVA from 2.1. Zimmerman et al. note that pseudobulk aggregation is conservative relative to mixed models but valid.
2. Permutation MANOVA (PERMANOVA) with patient-level permutation. Anderson MJ. "A new method for non-parametric multivariate analysis of variance." Austral Ecology 2001;26:32-46, [DOI 10.1111/j.1442-9993.2001.01070.pp.x](https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x). The pseudo-F statistic is computed from a distance matrix and its null distribution by permuting observations; the only assumption is exchangeability under the null. Anderson explicitly describes nested designs: for an upper-level factor (stage), whole units (patients, with all their cells) are permuted together, so the test is against variation among patients rather than among cells. With cells as rows and patient-level permutation, the permutation distribution is generated by relabelling patients, which gives at most 21!/(4! 2! 5! 10!) distinct assignments (on the order of 10^9), ample for Monte Carlo sampling. Anderson also cautions that PERMANOVA is sensitive to differences in multivariate dispersion as well as location; report a dispersion test (PERMDISP) alongside.
3. Mixed-model MANOVA. Fit a multivariate linear mixed model with a random intercept (and optionally random slopes) for patient, or fit four univariate linear mixed models (one per program) with patient random intercepts and combine with a Holm correction. Zimmerman et al. show that mixed models with a random effect for individual control type I error and are more powerful than pseudobulk. In Python the multivariate version is not available in statsmodels; use R (MCMCglmm, brms) or fit four univariate `statsmodels.regression.mixed_linear_model.MixedLM` models.

### 2.3 (c) Recommended analysis plan

Primary analysis (confirmatory, patient level).
- Unit: patient. Outcome: the 4-vector of mean program scores (measured). Factor: stage, collapsed to control / precursor / myeloma (3 groups) unless the 4-group analysis is explicitly exploratory.
- Statistic: Pillai's trace, reported with the approximate F, degrees of freedom and p-value from `statsmodels.multivariate.manova.MANOVA`.
- Assumption checks reported descriptively: within-group correlation matrix of the four programs; per-group standard deviations; a note on the direction of variance heterogeneity relative to group size; a QQ plot of Mahalanobis distances of residuals.
- Effect size: partial eta squared derived from Pillai's trace, eta_p^2 = V / s, where s = min(p, df_hypothesis). For p = 4 and 3 groups, s = 2; for 4 groups, s = 3. This is the formula used by SPSS GLM and given in Tabachnick and Fidell (chapter 7); for Wilks' lambda the analogous quantity is 1 - lambda^(1/s). Report a bootstrap percentile interval by resampling patients within group (B = 2000), and state that with 21 patients the interval will be wide.
- Secondary: the same analysis on the cross-validated morphology-predicted vectors. Predicted scores are shrunk toward the mean, so their between-stage differences are attenuated; a smaller Pillai's trace on predicted than on measured vectors is expected and should not be read as evidence against the stage effect. Report per-program Pearson or Spearman correlation between measured and predicted patient means (n = 21) as the measure of transfer.

Permutation analysis (sensitivity, no distributional assumptions).
- Statistic: Pillai's trace from the patient-level MANOVA, or Anderson's pseudo-F on Euclidean distances among patient vectors (standardise each program to unit variance first).
- Null distribution: permute stage labels across the 21 patients (this is the "whole units permuted together" rule of Anderson 2001 for an upper-level factor). Use 9999 permutations; p = (b + 1) / (B + 1) where b is the number of permuted statistics at least as large as the observed one.
- Also report PERMDISP (distance to group centroid, tested by ANOVA or permutation) to separate location from dispersion effects.

Reference implementation.

```python
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# df: one row per patient; columns wnt, stress, exhaustion, cytotox, stage
df["stage"] = pd.Categorical(df["stage"], ["control", "precursor", "myeloma"])

mv = MANOVA.from_formula("wnt + stress + exhaustion + cytotox ~ C(stage)", data=df)
res = mv.mv_test()
print(res)  # Intercept block and C(stage) block: Wilks' lambda, Pillai's trace,
            # Hotelling-Lawley trace, Roy's greatest root; columns Value, Num DF,
            # Den DF, F Value, Pr > F

tab = res.results["C(stage)"]["stat"]
V = tab.loc["Pillai's trace", "Value"]
p_obs = tab.loc["Pillai's trace", "Pr > F"]
s = min(4, df["stage"].nunique() - 1)
eta_p2 = V / s

# Patient-level permutation test on Pillai's trace
rng = np.random.default_rng(0)
B = 9999
count = 0
for _ in range(B):
    d = df.copy()
    d["stage"] = rng.permutation(d["stage"].values)
    r = MANOVA.from_formula("wnt + stress + exhaustion + cytotox ~ C(stage)", data=d).mv_test()
    if r.results["C(stage)"]["stat"].loc["Pillai's trace", "Value"] >= V:
        count += 1
p_perm = (count + 1) / (B + 1)
```

For the Anderson pseudo-F alternative, compute the Euclidean distance matrix on standardised patient vectors, SS_total = sum of squared distances over all pairs divided by N, SS_within = sum over same-group pairs divided by n_j summed over groups, SS_among = SS_total - SS_within, pseudo-F = (SS_among / (g - 1)) / (SS_within / (N - g)), and permute patient labels as above. The `skbio.stats.distance.permanova` function implements this directly.

Reporting template. "A one-way MANOVA on patient-level mean program vectors (n = 21; control 4, precursor 7, myeloma 10) gave Pillai's trace V = x.xx, F(df1, df2) = x.xx, p = 0.xx, partial eta squared = 0.xx (bootstrap 95% CI a to b). A permutation test with 9999 patient-level relabellings gave p = 0.xx. Group sizes are below those recommended for MANOVA (Tabachnick and Fidell 2019); results are reported as exploratory. Cell-level tests were not used because cells nested within patients are pseudoreplicates (Zimmerman et al. 2021)."

Additional notes from the Roseth et al. 2026 IMC study (PMC full text): the authors used mixed models with subject-level random effects for cell-level comparisons "thereby accounting for within-subject correlation and preventing type I error inflation due to unmodeled heterogeneity", which is a precedent in the same tissue and disease.


### 2.4 What the v3.1 pipeline runs by default, and why MANOVA was demoted

The user named MANOVA as a reference, not as a requirement. Weighing the options above against the actual design (about 21 patients, four stage groups of sizes 4, 2, 5 and 10, seven neighbourhood-scale program variables), the pipeline now runs two tests that fit the design, and keeps the classical MANOVA only as an opt-in comparison (`evaluation.stage_test: manova` or `both`):

1. PERMANOVA on the patient-level vectors (`evaluation/stage_association.py`, function `permanova`). Pseudo-F from the Euclidean distance matrix of z-scored patient burden vectors, null distribution from `stage_n_perm` permutations of stage labels across patients (the whole-unit permutation Anderson describes for an upper-level factor), with R squared = SS_between / SS_total reported as the effect size. It is defined with more variables than patients, needs no normality, and the classical requirement n > variables + groups that made the synthetic run skip MANOVA does not apply. Verified citation: Anderson MJ. A new method for non-parametric multivariate analysis of variance. Austral Ecology 2001;26(1):32-46, [DOI 10.1111/j.1442-9993.2001.01070.pp.x](https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x). PERMDISP (dispersion) is not implemented; the per-group dispersion caveat from 2.2 still stands and is printed with the result.
2. Jonckheere-Terpstra trend test per program (function `jonckheere_terpstra`), which is option (iii) from 2.1: one ordered contrast control < MGUS < SMM < MM instead of a three-degree-of-freedom omnibus difference. The statistic is the exact sum over ordered group pairs of Mann-Whitney counts (ties count one half); the p-value is two-sided by permutation; Benjamini-Hochberg q-values are computed across the programs tested, and the Spearman correlation of burden with stage rank is printed for continuity with the earlier `stage_trend` summary. Verified citation: Jonckheere AR. A distribution-free k-sample test against ordered alternatives. Biometrika 1954;41(1-2):133-145, [DOI 10.1093/biomet/41.1-2.133](https://doi.org/10.1093/biomet/41.1-2.133). Terpstra TJ. The asymptotic normality and consistency of Kendall's test against trend, when ties are present in one ranking. Indagationes Mathematicae 1952;14:327-333 (citation not re-verified). Verified citation for the multiplicity correction: Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B 1995;57(1):289-300, [DOI 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

Both tests are run twice, on the predicted and on the measured patient vectors, so the reader can see whether a stage effect in the morphology predictions reproduces one that exists in the measured expression. The stage order is the full four-level vocabulary when every stage has at least two patients, and otherwise the control / precursor / myeloma collapse from 2.1 (i), which is stated in the result's `groups` column. Degenerate inputs (fewer than two groups, fewer than three patients) write a row whose note starts with "skipped:" rather than stopping the run. The artefact is `evaluation/stage_association.csv` with columns `vectors, test, program, statistic, p_value, q_value, r2_or_rho, n_patients, n_groups, groups, note`.

Interpretation limit, printed with every result: stage is a proxy for disease progression, not a measurement of drug response, and with these group sizes a non-significant result is absence of evidence, not evidence of absence.

---

## Q3. SpatialCOC

### 3.1 Bibliographic record

According to PubMed: Li M, Sun P, Luo Y, Zhou G, Yang X, Meng D, Ye K. "SpatialCOC: an integrative framework for spatial continuous mapping and cross-omics correction in spatial multi-omics data." Nature Communications 2026;17(1), article 5268, published 16 April 2026, [DOI 10.1038/s41467-026-71882-2](https://doi.org/10.1038/s41467-026-71882-2) (PMID 41991905, PMC13265925; Xi'an Jiaotong University, Kai Ye laboratory). Open access. Code: GitHub xjtu-omics/SpatialCOC (GPL-3.0, Python 3.11, tutorials in Tutorials/Tutorial_[1-5].ipynb); software archived on Zenodo as "xjtu-omics/SpatialCOC: v0.1.0", DOI 10.5281/zenodo.18595579 (cited in the paper's reference list; the Zenodo page could not be fetched from this session); results for figure reproduction on Zenodo DOI 10.5281/zenodo.20115541 (from the readthedocs Reproduction page). Documentation at spatialcoc-tutorials.readthedocs.io. The method details below are taken from the PMC full text and the Nature Communications Methods section.

### 3.2 Problem framing

The paper targets vertical integration of spatial multi-omics: two or more modalities measured on the same spots or cells with shared coordinates (for example spatial ATAC-RNA-seq, CUT&Tag-RNA-seq, SPOTS RNA plus 21 proteins, Stereo-CITE-seq). The authors argue that existing methods (SpatialGlue, COSMOS, STAGATE, SpaGCN, Seurat WNN, MultiVI, MultiMAP) are "bottom-up": they build a neighbourhood graph per modality and fuse by weighted summation or a shared latent space, which captures only local neighbourhoods, is noise sensitive, and can only down-weight a noisy modality rather than remove its noise. SpatialCOC is "top-down": spatial coordinates are treated as prior knowledge for every modality, and modalities then correct one another.

### 3.3 Spatial continuous mapping (SCM)

Inputs: coordinates X in R^(n x 2) and modality matrices Z_1 in R^(n x m_1), Z_2 in R^(n x m_2), where the Z_k may be raw features or reduced embeddings (PCA, LSI) of arbitrary and different dimension.

Model: for each modality k, an implicit neural representation (INR) Phi_k that maps any coordinate x to the expression vector of that modality at x. The network is a SIREN-style network: a linear layer on the coordinates, three sine-activation layers initialised with frequency vector omega_0 = [1, 2, 3] so that each sub-network captures a different spatial frequency band, and a final linear layer. Because the function is continuous in x, neighbouring coordinates produce similar outputs, which yields spatial smoothness without an explicit graph, and the representation is resolution independent, so expression can be queried at unobserved locations.

Loss: a fidelity loss per modality, L_INR_k = || Z_hat_k - K Z_k ||_2, where Z_hat_k = Phi_k(X) and K is an optional Gaussian kernel matrix K(x_i, x_j) = exp(-||x_i - x_j|| / (2 sigma^2)) with sigma = 0.002 in all four datasets (K is the identity when the smoothness regulariser is switched off). The target K Z_k is a kernel-smoothed version of the observed data, so the INR is fitted to spatially denoised expression. The output of SCM is a spatially continuous reconstruction of each modality on the shared coordinates.

### 3.4 Cross-omics correction (COC)

Inputs: the SCM reconstructions Z_hat_1, Z_hat_2.

Model: an autoencoder per modality (encoder with L = 3 layers, batch normalisation on the first layer, mirrored decoder), followed by a linear canonical correlation analysis (CCA) layer. The design is explicitly described as inspired by deep CCA autoencoders (DCCAE).

Loss: two terms are jointly optimised. (1) Reconstruction loss, which requires each decoder to reproduce its own modality so that modality-specific biological structure is retained. (2) Correction loss, which maximises the correlation between the two latent embeddings; formally the CCA objective, maximise tr(W_1^T Sigma_12 W_2) subject to W_k^T (Sigma_kk + r_k I) W_k = I with regularisers r_1, r_2 > 0, solved by the singular value decomposition of Sigma_11^(-1/2) Sigma_12 Sigma_22^(-1/2). The final per-spot embedding is the projection of each decoder output onto the top-k canonical directions, giving an orthogonal, cross-modally aligned representation.

What "correction" means. The authors contrast "correction" with "weighted summation". Weighted summation can only reduce the weight of a noisy modality; the correlation objective retains the components of each modality that covary with the other modality and discards modality-specific variance that does not. The noise being removed is described as RNA "background noise" from uneven cell density, multimodal interference, and batch effects between replicate sections, and the paper reports batch-consistent PAGA trajectories across three thymus replicates as evidence. A stated limitation is that COC requires every modality to be measured on the same cells (vertical integration); it does not handle unpaired or mosaic data.

Training: modular. SCM is trained first per modality; its outputs feed COC, which is trained with reconstruction plus correction loss; the overall objective is the sum of both stages. Ablations on augmented lymph-node data (quadrant, stripe, arc, layered patterns; Gaussian and pepper noise) are used to justify each term.

### 3.5 Which single idea transfers to a morphology-to-transcript model trained on one modality

The project at hand predicts program scores per cell from DAPI nuclear morphology (H&E only after the stain-harmonisation phase), with Xenium program scores available only at training time and only morphology at inference. The transferable idea is the SCM target construction: fit the regression to spatially smoothed program scores, K Z, rather than to raw per-cell scores, using exactly the Gaussian-kernel smoothing that SpatialCOC applies before INR fitting.

Reasons this is the right transfer.
1. The Xenium 5K data behind the targets are sparse (15 to 255 genes detected per cell in Yip et al. 2025), so per-cell program scores are dominated by sampling noise. A model regressed on raw scores learns to predict noise; a model regressed on K Z learns the spatial field of the program. SpatialCOC reports Moran's I rising from 0.57 to 0.94 after SCM reconstruction on the mouse brain data, which is the same denoising effect.
2. It requires no change to the inference path: the morphology model is still trained on one modality's features and predicts a 4-vector per cell; only the target changes. The kernel bandwidth sigma becomes a tuned hyperparameter, set in physical units (micrometres) rather than SpatialCOC's normalised 0.002, and selected by cross-validated correlation between predictions and held-out raw scores (not held-out smoothed scores, to avoid rewarding over-smoothing).
3. It is directly compatible with the MANOVA plan in Q2: the patient-level mean of K Z equals the patient-level mean of Z up to edge effects, so the smoothing does not alter the confirmatory analysis.

The COC idea (a DCCAE-style correction loss between a morphology embedding and a program embedding, with the CCA projection used at inference from the morphology branch alone) is also applicable, because training data are paired per cell. It is the more ambitious transfer and is better suited to a second iteration: it changes the model from direct regression to shared-subspace learning, and the resulting canonical variates are linear combinations of the four programs rather than the programs themselves, which complicates per-program interpretation and the MANOVA on named programs.

A continuous spatial field model (an INR from coordinates to program scores) is not transferable as a predictor, because at inference the field is unknown for a new trephine; it is useful only as the smoother described above or as a post hoc spatial regulariser on predictions within a section.

---

## Sources

PubMed records retrieved in this session (attribution required by PubMed terms of use):
- Bandyopadhyay S et al. Cell 2024. [https://doi.org/10.1016/j.cell.2024.04.013](https://doi.org/10.1016/j.cell.2024.04.013)
- Bandyopadhyay S et al. bioRxiv 2024. [https://doi.org/10.1101/2024.03.14.585083](https://doi.org/10.1101/2024.03.14.585083)
- Koops M et al. Blood Cancer Discov 2026. [https://doi.org/10.1158/2643-3230.BCD-26-0080](https://doi.org/10.1158/2643-3230.BCD-26-0080)
- Roseth IA et al. Blood Cancer Discov 2026. [https://doi.org/10.1158/2643-3230.BCD-25-0334](https://doi.org/10.1158/2643-3230.BCD-25-0334)
- Yip RKH et al. Blood 2025. [https://doi.org/10.1182/blood.2025028896](https://doi.org/10.1182/blood.2025028896)
- Li M et al. Nat Commun 2026 (SpatialCOC). [https://doi.org/10.1038/s41467-026-71882-2](https://doi.org/10.1038/s41467-026-71882-2)
- Zimmerman KD et al. Nat Commun 2021. [https://doi.org/10.1038/s41467-021-21038-1](https://doi.org/10.1038/s41467-021-21038-1)
- Beider K et al. Cancers 2026 (review, retrieved in search, not used). [https://doi.org/10.3390/cancers18193158](https://doi.org/10.3390/cancers18193158)
- Herzberg F et al. Leukemia 2026 (retrieved in search, not used). [https://doi.org/10.1038/s41375-026-03047-5](https://doi.org/10.1038/s41375-026-03047-5)

Web sources:
- Figshare+ dataset "Normal Bone Marrow CODEX Images": https://plus.figshare.com/articles/dataset/Normal_Bone_Marrow_CODEX_Images/25122311 and collection https://doi.org/10.25452/figshare.plus.c.7174914
- HuBMAP featured publications: https://hubmapconsortium.org/featured-publications/
- Baertsch et al. ASH 2023 abstract: https://ash.confex.com/ash/2023/webprogram/Paper189255.html
- Soekojo et al. ASH 2024 abstract: https://ash.confex.com/ash/2024/webprogram/Paper207440.html
- International Myeloma Society award page: https://www.myelomasociety.org/award/deciphering-the-human-plasma-cell-niche-and-its-transformation-in-multiple-myeloma-by-highly-multiplexed-tissue-imaging/
- Li L, PhD thesis, Tubingen 2026: https://publikationen.uni-tuebingen.de/xmlui/handle/10900/181393
- OmicsDI records: GSE345302, GSE299193, GSE299195 (https://www.omicsdi.org/dataset/geo/...)
- 10x Genomics blog on Yip et al.: https://www.10xgenomics.com/blog/xenium-in-situ-generates-the-first-map-of-human-bone-marrow-in-myeloma
- SpatialCOC GitHub: https://github.com/xjtu-omics/SpatialCOC ; docs: https://spatialcoc-tutorials.readthedocs.io
- Anderson MJ 2001, Austral Ecology 26:32-46, PDF: https://www.pelagicos.net/MARS6300/readings/Anderson_2001.pdf
- Olson CL 1974 citation confirmed at https://www.ibm.com/docs/en/SSLVMB_23.0.0/spss/tutorials/glmm_readings.html
- statsmodels MANOVA documentation: https://www.statsmodels.org/v0.14.4/generated/statsmodels.multivariate.manova.MANOVA.html ; worked example: https://psiaims.github.io/CAMIS/python/MANOVA.html
- Tabachnick BG, Fidell LS. Using Multivariate Statistics, 7th ed. Pearson, 2019 (textbook; not retrieved online).
- Hurlbert SH. Pseudoreplication and the design of ecological field experiments. Ecological Monographs 1984;54:187-211 (textbook-level citation; not retrieved online).
