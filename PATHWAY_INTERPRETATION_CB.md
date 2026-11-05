# Pathway interpretation of the four target programs

This document audits the biological interpretation attached to the four gene
programs in `resources/gene_programs.json`. Each claim in the repository is
checked against the primary literature, and the claim is graded as
**established** (multiple independent studies or a review of such studies),
**single-study** (one primary report, not independently replicated in the
sources checked), or **unsupported** (no source found, or the cited source does
not contain the claim). Literature was retrieved from PubMed; DOI links are
given in the reference list.

A summary of required wording changes is in the final section.

---

## 1. Source attribution: two claims that must be withdrawn

Status 2026-11-03: the wording changes in section 6 were applied to README.md,
DECISIONS.md D1 and gene_programs.json; the quotations below are the pre-correction
text, kept for the record.

Two attribution claims in the repository were not supported by the cited work and
have been corrected; the record of the audit follows.

**1.1 Yip et al. (2025) does not report Wnt signalling, cell adhesion-mediated
drug resistance, or progression-free survival.** The repository states, in
`docs/DECISIONS.md` (D1), that "the paper itself reports the plasma-cell-dense
niche enriched for non-canonical Wnt signalling, associated with cell
adhesion-mediated drug resistance and inferior progression-free survival", and
`README.md` repeats that the target program was "reported as a plasma-cell-dense
niche with inferior progression-free survival". The full text of Yip et al.
(2025) contains no mention of Wnt, WNT5A, ROR2, cell adhesion-mediated drug
resistance, or progression-free survival, and the study reports no survival
analysis of any kind. The paper reports the opposite of a conserved resistance
niche: it concludes that the data "are not consistent with the hypothesis that a
universal bone marrow microenvironment supports the expansion of malignant
plasma cells in myeloma", and that distinct plasma cell subpopulations occupy
transcriptionally and compositionally unique neighbourhoods that vary between
and within patients. Status: **unsupported**. The dataset citation (GSE299193,
Xenium 5K, 21 trephines, 7 precursor and 10 newly diagnosed cases) is accurate
and may be retained; the biological claim attached to it may not.

**1.2 The Ohlstrom et al. (2026) preprint is verified.** The Ohlstrom et al. (2026)
preprint is verified (bioRxiv, doi 10.64898/2026.03.23.713195, PMID 41928953; see
LITERATURE_VERIFIED.md items 2 and 9). It reports pathway-level findings
(non-canonical Wnt, CAM-DR-associated ligands, a 15-gene PFS signature) but no
supplementary gene list was used here, so the four program lists remain
investigator-defined sets motivated by that preprint, not reproduced from it.

The consequence of 1.1 and 1.2 together is that the four programs are
**investigator-defined gene sets motivated by the myeloma signalling
literature**, not programs discovered in, or validated against, either cited
dataset. That is a defensible basis for a prediction target, but it is a
different and weaker claim than the one currently in the repository, and no
outcome association can be claimed for the 18-gene set itself: the preprint's
progression-free survival result concerns its own 15-gene signature (C8, preprint
strength), and GSE299193 carries no survival data.

---

## 2. Non-canonical Wnt in the myeloma niche

Non-canonical Wnt signalling proceeds without beta-catenin stabilisation,
through Frizzled receptors acting with alternative co-receptors, principally the
receptor tyrosine kinase-like orphan receptor ROR2, and through downstream
planar cell polarity, calcium, and Rho or JNK effectors.

**2.1 ROR2 is overexpressed in myeloma and mediates adhesion to the bone marrow
through PI3K and AKT.** Frenquelli et al. (2020) identified ROR2 in an unbiased
screen of receptor tyrosine kinases overexpressed in myeloma, reported WNT5A as
the most abundant growth factor in the bone marrow of myeloma patients,
demonstrated in three-dimensional and ex vivo culture that ROR2 is required for
adhesion of myeloma cells to the microenvironment, showed that ROR2 depletion
detaches myeloma cells from the niche in vivo with consequent apoptosis and
delayed disease progression, and found PI3K/AKT and mTOR to be the pathways most
deregulated by ROR2 overexpression, with clinical PI3K inhibitors reducing
adhesion. Status: **single-study** for the complete chain, from a well-powered
primary report with in vitro, ex vivo, and in vivo arms. The ROR2 to adhesion to
PI3K/AKT link in the repository is therefore correctly stated and correctly
cited, and should be labelled as resting on one primary study.

**2.2 The WNT5A/ROR2 axis is compartment-dependent and not uniformly
tumour-supportive.** Bolzoni et al. (2013) reported that ROR2 expression rises
during osteogenic differentiation of bone marrow mesenchymal stromal cells, that
myeloma cells suppress ROR2 in osteoprogenitors by inhibiting WNT5A signalling,
and that activating the WNT5A/ROR2 pathway in stromal cells increases osteogenic
differentiation and blunts the inhibitory effect of myeloma cells. Status:
**single-study**. This matters for interpretation of a neighbourhood score: a
high WNT5A/ROR2 signal in a 30 micrometre neighbourhood may arise from the
plasma cell compartment, where the Frenquelli mechanism applies, or from the
stromal and osteoprogenitor compartment, where the direction of the biology is
the opposite. A count-based neighbourhood score cannot distinguish the two.

**2.3 Non-canonical, Rho-dependent Wnt responses exist in myeloma cells.** Qiang
et al. (2003) showed that Frizzled and LRP5/6 transcripts are present in myeloma
lines, that Wnt3a stabilises beta-catenin in myeloma but not in lymphoma cells,
and that Wnt3a additionally triggers actin cytoskeletal rearrangement through a
second, Rho-dependent pathway. Status: **single-study**, and the evidence is at
the level of the Frizzled and LRP families rather than the specific FZD3 and
LRP6 genes in the program.

---

## 3. Canonical Wnt and drug resistance

**3.1 Canonical Wnt activity in myeloma is ligand-dependent and mediates
proliferation, migration, and drug resistance.** van Andel et al. (2019) review
the evidence that most myelomas display aberrant Wnt/beta-catenin activity
without intrinsic pathway mutations, that this activity is driven by autocrine
and paracrine ligands acting on cells sensitised by genetic and epigenetic
lesions of Wnt regulators, and that the functional consequence includes drug
resistance. Status: **established**, as a review of multiple primary studies.

**3.2 Syndecan-1 presents Wnts and R-spondins and is a component of the myeloma
Wnt signalosome.** Ren et al. (2018) showed by EXT1 knockout and knockdown that
the heparan sulfate chains of syndecan-1 (CD138) mediate autocrine Wnt pathway
activity and Wnt-dependent proliferation, bind Wnts for paracrine signalling
through Frizzled, and bind osteoblast-derived R-spondin for LGR4-dependent
costimulation. Status: **single-study**, mechanistically thorough, and
consistent with 3.1.

**3.3 Canonical Wnt leads through CD44 to stromal adhesion and lenalidomide
resistance.** This link is asserted in the project framing but is not cited
anywhere in the repository. The supporting primary study is Bjorklund et al.
(2014), which reported that lenalidomide-resistant myeloma models overexpress
CD44, a Wnt/beta-catenin transcriptional target, that resistant cells adhere
more strongly to bone marrow stroma and to hyaluronan, that CD44 blockade or
knockdown reduces adhesion and resensitises cells to lenalidomide, that
Wnt/beta-catenin suppression enhances lenalidomide activity, and that
all-trans retinoic acid lowers beta-catenin and CD44 and resensitises resistant
cells in vivo and resensitises primary samples from lenalidomide-refractory
patients ex vivo. Status: **single-study**. This citation must be added, since it
is the only direct evidence for the canonical Wnt to CD44 to CAM-DR to IMiD
resistance chain, and CD44 is not in the program gene list.

**3.4 Wnt inhibition reduces the unfolded protein response and enhances
bortezomib killing.** Spaan et al. (2023) is correctly cited, but it is attached
to the wrong program. It reports a Wnt to unfolded protein response to
proteasome inhibitor sensitivity link, which belongs with `pc_stress_survival`
and in fact provides the clearest mechanistic bridge between Wnt activity and
proteasome inhibitor response available in these sources. Status:
**single-study**, published as a short report.

**3.5 Wnt inhibition raises CD38 and enhances daratumumab efficacy.** Li et al.
(2025) reported that Wnt inhibition lowers STAT3 activity, raises CD38
expression, and enhances daratumumab activity in vitro and in a xenograft model.
Status: **single-study**. Note the direction: Wnt activity reduces sensitivity to
an antibody therapy by lowering target antigen, which is a distinct mechanism
from adhesion-mediated resistance and does not involve the niche.

**3.6 Immunomodulatory drug resistance is mostly cereblon-pathway dependent.**
Teoh et al. (2025) is correctly cited as a review of IMiD resistance, but it
frames cereblon-pathway-dependent mechanisms as the principal contributors and
treats cereblon-independent mechanisms as the emerging minority. The repository
should not use this review as support for adhesion as a primary IMiD resistance
mechanism. Status: **established** as a review, but its emphasis is the opposite
of the use made of it.

**3.7 Wnt antagonist status in myeloma: methylation silences SFRP1 and SFRP2;
DKK1 is secreted and acts on bone.** Jost et al. (2009) found aberrant promoter
methylation in primary patient samples at frequencies of 35.5 percent for SFRP1,
52.6 percent for SFRP2, 1.3 percent for SFRP4, and 6.9 percent for SFRP5, with
SFRP1, SFRP2, and SFRP5 hypermethylation associated with transcriptional
silencing, and concluded that epigenetic silencing of Wnt antagonists is an
early event in myeloma pathogenesis. van Andel et al. (2019) lists promoter
methylation of WIF1, DKK1, DKK3, and SFRP1 through SFRP5 among the lesions that
sensitise myeloma cells to Wnt ligands. DKK1, by contrast, is reported by Tian
et al. (2003) as overexpressed and secreted by myeloma cells in patients with
focal bone lesions, where it inhibits osteoblast differentiation. Status:
**established** for SFRP1 and SFRP2 silencing by methylation, **established** for
DKK1 secretion and the bone phenotype.

This produces a direction problem that the current program does not resolve.
SFRP1 and SFRP2 transcripts are expected to be **low** in malignant plasma cells
because they are methylated, so detected SFRP1 and SFRP2 counts in a myeloma
trephine neighbourhood most plausibly originate from stromal and osteoblastic
cells. Summing them with WNT5A and ROR2 counts in a single unsigned score
conflates ligand abundance, receptor abundance, and antagonist abundance, which
have opposing effects on pathway output. A high `wnt_camdr` score is therefore
not interpretable as high Wnt pathway activity.

---

## 4. The CAM-DR co-program

**4.1 Cell adhesion-mediated drug resistance is an established phenomenon in
myeloma.** Damiano et al. (1999) defined CAM-DR in myeloma, showing that
drug-sensitive 8226 cells adhered to fibronectin through VLA-4 and VLA-5 resist
doxorubicin- and melphalan-induced apoptosis relative to suspension culture,
that this was not explained by reduced drug accumulation or by BCL2 family
upregulation, and that lines selected for drug resistance overexpress VLA-4 and
revert when selection pressure is removed. Anderson (2003) summarises the
broader microenvironment picture, in which adhesion to extracellular matrix and
to stromal cells, together with interleukin-6, vascular endothelial growth
factor, tumour necrosis factor alpha, and insulin-like growth factor 1, supports
growth, survival, drug resistance, and migration, with drug resistance signalled
through PI3K and AKT. Status: **established**.

Two qualifications apply to the mechanism chain as stated in the repository.
First, the founding CAM-DR evidence concerns conventional cytotoxics
(doxorubicin, melphalan), not proteasome inhibitors or immunomodulatory drugs.
The extension to proteasome inhibitors and IMiDs is supported for IMiDs by
Bjorklund et al. (2014) and for proteasome inhibitors by stroma-mediated
mechanisms such as those of Frassanito et al. (2016), but is **single-study** per
drug class rather than established. Second, CAM-DR is mediated principally by
integrins and CD44, none of which are in the program gene list.

**4.2 Eight of the eighteen genes are not Wnt pathway components.** MIF,
IFNAR2, IGF1, VEGFB, NCAM1, CDH2, IRF4, and BCL2 are not Wnt ligands,
receptors, co-receptors, or secreted Wnt modulators. None is a component of the
canonical or non-canonical Wnt cascade, and none is a documented direct
transcriptional target of beta-catenin in plasma cells in the sources reviewed.
Their presence is defensible on a different ground: they encode the adhesion,
survival, and cytokine machinery through which a stromal niche protects plasma
cells, which is the CAM-DR half of the program name. They belong to three
functional subgroups.

*Adhesion.* NCAM1 encodes CD56, an aberrant plasma cell surface marker used in
diagnostic flow cytometry. Okura et al. (2020) reported that **absence** of CD56
was associated with significantly worse overall survival (median 24 versus 60
months). The direction of this association is therefore opposite to a resistance
interpretation, and NCAM1 should not be presented as a resistance gene. CDH2
encodes N-cadherin; Mrozik et al. (2015) reported that elevated N-cadherin is
associated with poor prognosis in newly diagnosed myeloma, that knockdown
reduces adhesion to bone marrow endothelial cells and reduces tumour burden in
the C57BL/KaLwRij model, and that the antagonist ADH-1 reduces tumour burden
when given up front but not in established disease, concluding that N-cadherin
acts principally in extravasation and bone marrow homing rather than in
stromal adhesion. Status: **single-study** each.

*Survival and transcription.* BCL2 is the anti-apoptotic effector on which
t(11;14) myeloma is dependent, which makes that subgroup venetoclax-sensitive;
Chakraborty et al. (2022) review the clinical position, including the excess
mortality signal in the randomised BELLINI trial that was driven by
non-t(11;14) patients. High BCL2 in myeloma therefore predicts **sensitivity**
to a specific agent, not general resistance. IRF4 is the transcription factor to
whose aberrant regulatory network myeloma cells are addicted; Shaffer et al.
(2008) showed by RNA interference screening that IRF4 inhibition is toxic to
myeloma lines regardless of transforming oncogenic mechanism, and identified a
reciprocal IRF4 and MYC autoregulatory circuit. Status: **established** for
both, as plasma cell dependencies rather than as resistance genes.

*Cytokines and growth factors.* MIF provides the strongest non-Wnt CAM-DR
evidence in the program. Zheng et al. (2016) reported that MIF is markedly
elevated in myeloma bone marrow plasma relative to healthy controls, that high
MIF is associated with poorer overall survival across gene expression profile
cohorts, that MIF knockdown impairs adhesion to bone marrow stromal cells and
produces extramedullary tumours in SCID mice, that MIF acts through CXCR4 and
COPS5 to regulate the adhesion molecules ALCAM, ITGAV, and ITGB5, and, directly
on the point at issue, that MIF-deficient myeloma cells are sensitive to
chemotherapy when co-cultured with stromal cells and that the MIF inhibitor
4-IPP sensitises myeloma cells to chemotherapy. Status: **single-study**, and the
closest match in the program to the stated mechanism chain. IGF1 is an
established microenvironment survival and drug resistance cytokine signalling
through PI3K and AKT (Anderson, 2003), and Frassanito et al. (2016) showed that
bortezomib triggers bone marrow fibroblasts from bortezomib-resistant patients
to secrete interleukin-6, interleukin-8, IGF-1, and transforming growth factor
beta, and that these fibroblasts protect plasma cells from bortezomib-induced
apoptosis. Status: **established** for IGF1 in the niche, **single-study** for the
bortezomib-specific fibroblast mechanism. IFNAR2 is a type I interferon receptor
subunit; Walters and Jelinek (2004) reported receptor crosstalk between the
interferon alpha complex and ErbB3 through Tyk2 and Jak1 contributing to
proliferation in one interferon-responsive myeloma line. Status:
**single-study**, and indirect. For VEGFB, no myeloma-specific functional
evidence was found; the myeloma angiogenesis literature concerns VEGFA.

**4.3 Two Wnt ligands lack myeloma-specific support in the expected direction.**
For WNT5B, no myeloma-specific functional study was found; the ligand-level
evidence in myeloma rests on WNT5A. For WNT10A, the only myeloma-specific
evidence found is Restrepo et al. (2022), in which WNT10A forms part of a
three-gene signature, with DUSP1 and ETV7, that predicts **response** to
selinexor through upregulated interferon-mediated apoptotic signalling. That is
a drug-sensitivity biomarker, and its mechanism is interferon signalling rather
than Wnt signalling. Its co-occurrence with IFNAR2 in the same program suggests
that part of this gene set may behave as an interferon module rather than a Wnt
module, which is consistent with the inflammatory plasma cell signatures Yip et
al. (2025) report.

**4.4 FRZB is a bone disease gene in myeloma, not a resistance gene.** FRZB,
also known as sFRP3, appears in the myeloma literature in the context of
myeloma bone disease, where secreted Wnt antagonists impair osteoblast
differentiation (Borset et al., 2019). No drug resistance evidence was found.

---

## 5. What the morphology model can and cannot claim about Wnt

The following limits follow from the evidence above and from the design
decisions already recorded in `docs/DECISIONS.md`.

**Can be claimed, if the cross-validation results support it.** That nuclear
morphology in a DAPI channel carries information about the local abundance of a
defined set of eighteen transcripts, measured as a neighbourhood count score,
beyond what cell type composition alone explains. This is a statement about
predicting a measured quantity, and it is testable against the `shuffled` and
`composition_only` baselines and against the split-half reliability ceiling
already implemented.

**Cannot be claimed.** That the model detects Wnt pathway activity. The program
mixes ligands, receptors, and secreted antagonists whose effects on pathway
output have opposite signs, and the antagonists SFRP1 and SFRP2 are expected to
be epigenetically silenced in the malignant compartment, so their counts
probably report stromal rather than plasma cell biology. An unsigned sum of
these transcripts is not a pathway activity estimate. Pathway activity would
require a signed score, compartment-resolved scoring, or a validated
transcriptional readout such as beta-catenin target genes, none of which is
currently computed.

**Cannot be claimed.** That the model detects drug resistance, or a
drug-resistant niche. No drug exposure, response, or resistance data exist in
GSE299193. Three of the eighteen genes (WNT10A, NCAM1, BCL2) carry evidence
pointing toward drug sensitivity or better outcome rather than resistance. The
mechanism chain is assembled from separate studies in separate model systems,
and no study links this specific eighteen-gene score to any clinical outcome.

**Cannot be claimed.** Any association with progression-free survival. Yip et
al. (2025) performed no survival analysis, and the repository's pipeline
computes a Cox model only if a clinical table is supplied, which the public
dataset does not include.

**Cannot be claimed.** That the eighteen-gene set is a published niche program.
See section 1.2.

**Should be stated explicitly in outputs.** That the program is an
investigator-defined gene set; that the five Wnt-related functional classes
within it have opposing pathway effects; that eight of the eighteen genes are
not Wnt components; and that the resistance interpretation is a hypothesis drawn
from separate mechanistic studies, not a property of the measured target.

---

## 6. Corrected program labels

The current labels and the wording that the evidence supports.

| Program | Current label | Corrected label |
|---|---|---|
| `wnt_camdr` | "Non-canonical Wnt ligands and receptors plus adhesion and survival genes; adhesion to stroma protects plasma cells from proteasome inhibitors and IMiDs (cell adhesion-mediated drug resistance)" | "Investigator-defined set of ten Wnt ligands, receptors, and secreted antagonists and eight niche adhesion, survival, and cytokine genes. The Wnt members have opposing effects on pathway output and the set is not a pathway activity score. Stromal adhesion protects plasma cells from cytotoxic drugs in myeloma models, with per-drug-class evidence from single studies; no drug response data exist in this dataset." |
| `pc_stress_survival` | "Unfolded protein response and mTOR-related survival genes in plasma cells; the stress axis proteasome inhibitors act on" | "Plasma cell stress and survival genes: two mTOR regulators (DEPTOR, DDIT4), one unfolded protein response kinase (EIF2AK3, which encodes PERK), and three cytokine or innate immune receptors (TNFRSF13B, which encodes TACI, plus TLR10 and IL16). The unfolded protein response and mTOR label describes three of the six genes." |
| `t_exhaustion` | "Exhausted and regulatory T-cell genes; an exhausted microenvironment predicts poor response to T-cell redirecting therapy" | "Mixed T cell immunoregulation genes: regulatory T cell markers (FOXP3, IL2RA, CTLA4), one inhibitory checkpoint (LAG3), one costimulatory memory marker (CD27, whose expression is retained rather than lost in exhaustion), and three general immunoregulatory genes (TNFAIP3, CD55, AHR). Only LAG3, and CTLA4 in part, are exhaustion markers. The prognostic significance of regulatory T cells in myeloma is contested." |
| `cytotoxicity` | "Cytotoxic effector genes; the counterweight to exhaustion, used as a control program" | "Cytotoxic lymphocyte effector genes, used as a control program. Fifteen of the seventeen members are T cell or natural killer cell effector or lineage genes; DEFA3 and DEFA4 encode neutrophil alpha-defensins and report granulocyte rather than cytotoxic lymphocyte content." |

### Specific wording to change in existing files

- `docs/DECISIONS.md` D1: delete the clause asserting that Yip et al. report a
  non-canonical Wnt niche associated with cell adhesion-mediated drug resistance
  and inferior progression-free survival. Retain the dataset rationale, which is
  accurate: GSE299193 is the public dataset with matched whole-slide nuclear
  morphology and single-cell expression in myeloma bone marrow, in routine
  diagnostic trephines.
- `README.md`: replace "reported as a plasma-cell-dense niche with inferior
  progression-free survival" with a statement that the program is an
  investigator-defined gene set motivated by the Wnt and niche-adhesion
  literature in myeloma.
- `resources/gene_programs.json` `_source`: remove the unverifiable preprint
  attribution; state that the lineage marker lists follow the Xenium 5K panel
  markers used for cell typing in Yip et al. (2025), which is accurate, and that
  the program lists were assembled for this project.
- `resources/gene_programs.json` `_drug_resistance_link`: move the Spaan et al.
  (2023) citation from `pc_stress_survival` to the Wnt and proteasome inhibitor
  discussion, where it belongs mechanistically, and add Bjorklund et al. (2014),
  Damiano et al. (1999), and Zheng et al. (2016), which are the primary sources
  for the CAM-DR chain and are currently missing.
- Any statement that the model predicts "drug-resistance program scores", as in
  the README title line, should be qualified to "scores of a gene set associated
  with drug resistance in the myeloma literature", since the target itself
  contains no resistance measurement.

### Claims in the repository that are accurate as written

- D5, exclusion of the relapse sample RM-1 on panel size. Yip et al. (2025)
  confirm RM-1 was run on the 377-gene multitissue and cancer panel.
- D7, the transcript sparsity rationale. Yip et al. (2025) report 15 to 255
  genes detected per cell, consistent with the repository's stated 80 to 130
  transcripts per cell and with the need for neighbourhood aggregation.
- D2, that the GEO morphology image is a DAPI stack rather than haematoxylin and
  eosin. Yip et al. (2025) describe separate haematoxylin and eosin imaging on a
  slide scanner after Xenium imaging, co-registered in Xenium Explorer, so the
  Xenium morphology image is indeed the nuclear channel.
- The DEPTOR rationale. Peterson et al. (2009) identified DEPTOR as an mTOR
  inhibitor overexpressed in a subset of myelomas carrying cyclin D1 or D3 or
  c-MAF or MAFB translocations, required for PI3K and AKT activation and for
  survival.
- The gene identity assignments implied for `pc_stress_survival`: EIF2AK3
  encodes PERK, DDIT4 encodes REDD1, and TNFRSF13B encodes TACI, the receptor
  for BAFF and APRIL (Shivakumar and Ansell, 2006).
- The reliability-ceiling discipline in D9 and the baseline and ablation
  discipline in D19. These are methodological rather than biological and are
  unaffected by the findings above.

---

## References

Anderson, K. C. (2003). Moving disease biology from the lab to the clinic.
*Cancer, 97*(3 Suppl), 796-801. https://doi.org/10.1002/cncr.11137

Bjorklund, C. C., Baladandayuthapani, V., Lin, H. Y., Jones, R. J., Kuiatse, I.,
Wang, H., Yang, J., Shah, J. J., Thomas, S. K., Wang, M., Weber, D. M., &
Orlowski, R. Z. (2014). Evidence of a role for CD44 and cell adhesion in
mediating resistance to lenalidomide in multiple myeloma: Therapeutic
implications. *Leukemia, 28*(2), 373-383. https://doi.org/10.1038/leu.2013.174

Bolzoni, M., Donofrio, G., Storti, P., Guasco, D., Toscani, D., Lazzaretti, M.,
Bonomini, S., Agnelli, L., Capocefalo, A., Dalla Palma, B., Neri, A., Nicolini,
F., Lisignoli, G., Russo, F., Colla, S., Aversa, F., & Giuliani, N. (2013).
Myeloma cells inhibit non-canonical Wnt co-receptor Ror2 expression in human
bone marrow osteoprogenitor cells: Effect of Wnt5a/Ror2 pathway activation on
the osteogenic differentiation impairment induced by myeloma cells. *Leukemia,
27*(2), 451-463. https://doi.org/10.1038/leu.2012.190

Borset, M., Sundan, A., Waage, A., & Standal, T. (2020). Why do myeloma patients
have bone disease? A historical perspective. *Blood Reviews, 41*, 100646.
https://doi.org/10.1016/j.blre.2019.100646

Chakraborty, R., Bhutani, D., & Lentzsch, S. (2022). How do we manage t(11;14)
plasma cell disorders with venetoclax? *British Journal of Haematology, 199*(1),
31-39. https://doi.org/10.1111/bjh.18243

Damiano, J. S., Cress, A. E., Hazlehurst, L. A., Shtil, A. A., & Dalton, W. S.
(1999). Cell adhesion mediated drug resistance (CAM-DR): Role of integrins and
resistance to apoptosis in human myeloma cell lines. *Blood, 93*(5), 1658-1667.
https://pubmed.ncbi.nlm.nih.gov/10029595/

Frassanito, M. A., De Veirman, K., Desantis, V., Di Marzo, L., Vergara, D.,
Ruggieri, S., Annese, T., Nico, B., Menu, E., Catacchio, I., Ria, R.,
Racanelli, V., Maffia, M., Angelucci, E., Derudas, D., Fumarulo, R., Dammacco,
F., Ribatti, D., Vanderkerken, K., & Vacca, A. (2016). Halting pro-survival
autophagy by TGFbeta inhibition in bone marrow fibroblasts overcomes bortezomib
resistance in multiple myeloma patients. *Leukemia, 30*(3), 640-648.
https://doi.org/10.1038/leu.2015.289

Frenquelli, M., Caridi, N., Antonini, E., Storti, F., Vigano, V., Gaviraghi, M.,
Occhionorelli, M., Bianchessi, S., Bongiovanni, L., Spinelli, A., Marcatti, M.,
Belloni, D., Ferrero, E., Karki, S., Brambilla, P., Martinelli-Boneschi, F.,
Colla, S., Ponzoni, M., DePinho, R. A., & Tonon, G. (2020). The WNT receptor
ROR2 drives the interaction of multiple myeloma cells with the microenvironment
through AKT activation. *Leukemia, 34*(1), 257-270.
https://doi.org/10.1038/s41375-019-0486-9

Hadjiaggelidou, C., & Katodritou, E. (2021). Regulatory T-cells and multiple
myeloma: Implications in tumor immune biology and treatment. *Journal of
Clinical Medicine, 10*(19), 4588. https://doi.org/10.3390/jcm10194588

Jost, E., Gezer, D., Wilop, S., Suzuki, H., Herman, J. G., Osieka, R., & Galm, O.
(2009). Epigenetic dysregulation of secreted Frizzled-related proteins in
multiple myeloma. *Cancer Letters, 281*(1), 24-31.
https://doi.org/10.1016/j.canlet.2009.02.002

Li, H., Chen, Y., Gregorova, M., Yang, T., Zhang, X., Yu, X., Wang, Z., Hua, H.,
Ye, L., Qi, X., Spaargaren, M., Pals, S. T., & Ren, Z. (2025). Targeting
Wnt/beta-catenin signaling enhances the efficacy of anti-CD38 immunotherapy in
multiple myeloma. *Neoplasia, 70*, 101242.
https://doi.org/10.1016/j.neo.2025.101242

Mrozik, K. M., Cheong, C. M., Hewett, D., Chow, A. W. S., Blaschuk, O. W.,
Zannettino, A. C. W., & Vandyke, K. (2015). Therapeutic targeting of N-cadherin
is an effective treatment for multiple myeloma. *British Journal of
Haematology, 171*(3), 387-399. https://doi.org/10.1111/bjh.13596

Okura, M., Ida, N., & Yamauchi, T. (2020). The clinical significance of CD49e
and CD56 for multiple myeloma in the novel agents era. *Medical Oncology,
37*(11), 103. https://doi.org/10.1007/s12032-020-01423-4

Peterson, T. R., Laplante, M., Thoreen, C. C., Sancak, Y., Kang, S. A., Kuehl,
W. M., Gray, N. S., & Sabatini, D. M. (2009). DEPTOR is an mTOR inhibitor
frequently overexpressed in multiple myeloma cells and required for their
survival. *Cell, 137*(5), 873-886. https://doi.org/10.1016/j.cell.2009.03.046

Qiang, Y. W., Endo, Y., Rubin, J. S., & Rudikoff, S. (2003). Wnt signaling in
B-cell neoplasia. *Oncogene, 22*(10), 1536-1545.
https://doi.org/10.1038/sj.onc.1206239

Rasche, L., & Weinhold, N. (2025). Unlocking myeloma's spatial world at the
micrometer scale. *Blood, 146*(15), 1744-1746.
https://doi.org/10.1182/blood.2025030205

Ren, Z., van Andel, H., de Lau, W., Hartholt, R. B., Maurice, M. M., Clevers, H.,
Kersten, M. J., Spaargaren, M., & Pals, S. T. (2018). Syndecan-1 promotes
Wnt/beta-catenin signaling in multiple myeloma by presenting Wnts and
R-spondins. *Blood, 131*(9), 982-994.
https://doi.org/10.1182/blood-2017-07-797050

Restrepo, P., Bhalla, S., Ghodke-Puranik, Y., Aleman, A., Leshchenko, V.,
Melnekoff, D. T., Agte, S., Jiang, J., Madduri, D., Richter, J., Richard, S.,
Chari, A., Cho, H. J., Jagannath, S., Walker, C. J., Landesman, Y., Lagana, A.,
& Parekh, S. (2022). A three-gene signature predicts response to selinexor in
multiple myeloma. *JCO Precision Oncology, 6*, e2200147.
https://doi.org/10.1200/PO.22.00147

Shaffer, A. L., Emre, N. C. T., Lamy, L., Ngo, V. N., Wright, G., Xiao, W.,
Powell, J., Dave, S., Yu, X., Zhao, H., Zeng, Y., Chen, B., Epstein, J., &
Staudt, L. M. (2008). IRF4 addiction in multiple myeloma. *Nature, 454*(7201),
226-231. https://doi.org/10.1038/nature07064

Shivakumar, L., & Ansell, S. (2006). Targeting B-lymphocyte stimulator/B-cell
activating factor and a proliferation-inducing ligand in hematologic
malignancies. *Clinical Lymphoma and Myeloma, 7*(2), 106-108.
https://doi.org/10.3816/CLM.2006.n.046

Spaan, I., van Nieuwenhuijzen, N., Kimman, T., Rockx-Brouwer, D., Tieland, R. G.,
Maurice, M. M., Minnema, M. C., Raymakers, R. A., & Peperzak, V. (2023). Wnt
inhibitors reduce the unfolded protein response and enhance bortezomib-induced
cell death in multiple myeloma. *Blood Advances, 7*(7), 1103-1107.
https://doi.org/10.1182/bloodadvances.2022008442

Teoh, P. J., Koh, M. Y., Mitsiades, C., Gooding, S., & Chng, W. J. (2025).
Resistance to immunomodulatory drugs in multiple myeloma: The cereblon pathway
and beyond. *Haematologica, 110*(5), 1074-1091.
https://doi.org/10.3324/haematol.2024.285636

Tian, E., Zhan, F., Walker, R., Rasmussen, E., Ma, Y., Barlogie, B., &
Shaughnessy, J. D. (2003). The role of the Wnt-signaling antagonist DKK1 in the
development of osteolytic lesions in multiple myeloma. *New England Journal of
Medicine, 349*(26), 2483-2494. https://doi.org/10.1056/NEJMoa030847

van Andel, H., Kocemba, K. A., Spaargaren, M., & Pals, S. T. (2019). Aberrant
Wnt signaling in multiple myeloma: Molecular mechanisms and targeting options.
*Leukemia, 33*(5), 1063-1075. https://doi.org/10.1038/s41375-019-0404-1

Walters, D. K., & Jelinek, D. F. (2004). A role for Janus kinases in crosstalk
between ErbB3 and the interferon-alpha signaling complex in myeloma cells.
*Oncogene, 23*(6), 1197-1205. https://doi.org/10.1038/sj.onc.1207203

Yip, R. K. H., Er, J., Qin, L., Nguyen, Q. H., Motyer, A., Rimes, J. S., Light,
A., Mishi, R. D., Ling, L., Anttila, C. J. A., Tsui, E., Amann-Zalcenstein, D.,
Dowling, M. R., Rogers, K. L., Bowden, R., Chen, Y., Harrison, S. J., &
Hawkins, E. D. (2025). Profiling the spatial architecture of multiple myeloma in
human bone marrow trephine biopsy specimens with spatial transcriptomics.
*Blood, 146*(15), 1837-1849. https://doi.org/10.1182/blood.2025028896

Zheng, Y., Wang, Q., Li, T., Qian, J., Lu, Y., Li, Y., Bi, E., Reu, F., Qin, Y.,
Drazba, J., Hsi, E., Yang, J., Cai, Z., & Yi, Q. (2016). Role of myeloma-derived
MIF in myeloma cell adhesion to bone marrow and chemotherapy response. *Journal
of the National Cancer Institute, 108*(11), djw131.
https://doi.org/10.1093/jnci/djw131

---

*Literature retrieved from PubMed. Claim grades reflect the sources listed here
and are not an exhaustive review of the myeloma Wnt literature.*
