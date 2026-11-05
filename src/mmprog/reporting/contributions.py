"""CONTRIBUTIONS.md: what the run established, with its numbers, and what it did not.

Every sentence here is generated from a CSV in the run directory, never typed
by hand, so the claims in a slide or report can be traced to a file.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..logging_utils import say
from ..resources import load_annotation_caveats
from ..paths import RunDir


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _external_validation_section(validation: dict, tables: Dict[str, pd.DataFrame], main_target: str) -> List[str]:
    """Section 7a: every validator's status, verdict and headline numbers, read from the summary and tables."""
    lines: List[str] = ["## 7a. External validation (modality-agnostic validators)", ""]
    lines.append("| validator | status | verdict | reason or headline |")
    lines.append("|---|---|---|---|")
    order = ("program_specificity", "spatial_fidelity", "perturbation_robustness", "independent_cohort", "protein_bridge")
    for name in order:
        v = validation.get(name)
        if not isinstance(v, dict):
            continue
        status = str(v.get("status", ""))
        verdict = "pass" if v.get("passed") is True else ("fail" if v.get("passed") is False else ("unjudged" if status == "ran" else ""))
        if status == "ran":
            keys = [k for k in ("median_z", "median_r_target", "median_moran_ratio", "cohens_d_pred", "cohens_d_measured",
                                "min_spearman_main", "worst_perturbation_main", "median_r", "n_matched") if k in v]
            head = ", ".join(f"{k} = {_fmt(v[k]) if isinstance(v[k], (int, float)) else v[k]}" for k in keys)
            bar = v.get("pass_bar")
            head = head + (f" (bar: {bar})" if bar else "")
        else:
            head = str(v.get("reason") or "")
        lines.append(f"| {name} | {status} | {verdict} | {head} |")
    lines.append("")
    spec = tables.get("program_specificity")
    if spec is not None and len(spec) and {"target", "r_target", "null_mean", "z"}.issubset(spec.columns):
        lines.append("Program specificity per target (median over patients): the prediction's correlation with the real "
                     "target against expression-matched random gene sets, and with each random half of the program's genes.")
        lines.append("")
        lines.append("| target | median r_target | median null mean | median z | median r_half_a | median r_half_b |")
        lines.append("|---|---|---|---|---|---|")
        g = spec.groupby("target")
        for t, grp in g:
            lines.append(f"| {t} | {_fmt(grp['r_target'].median())} | {_fmt(grp['null_mean'].median())} | {_fmt(grp['z'].median())} | "
                         f"{_fmt(grp['r_half_a'].median()) if 'r_half_a' in grp else ''} | "
                         f"{_fmt(grp['r_half_b'].median()) if 'r_half_b' in grp else ''} |")
        lines.append("")
    fid = tables.get("spatial_fidelity")
    if fid is not None and len(fid) and "check" in fid.columns:
        mor = fid[fid["check"] == "moran"]
        if len(mor):
            lines.append("Spatial fidelity (median over patients): Moran's I of the predicted and the measured map.")
            lines.append("")
            lines.append("| target | Moran's I predicted | Moran's I measured | ratio |")
            lines.append("|---|---|---|---|")
            for t, grp in mor.groupby("target"):
                lines.append(f"| {t} | {_fmt(grp['moran_pred'].median())} | {_fmt(grp['moran_measured'].median())} | "
                             f"{_fmt(grp['moran_ratio'].median())} |")
            lines.append("")
        nc = fid[fid["check"] == "niche_contrast"]
        if len(nc) and {"cohens_d_pred", "cohens_d_measured"}.issubset(nc.columns):
            lines.append("Plasma-cell-dense niche contrast (Cohen's d, dense versus other neighbourhoods):")
            lines.append("")
            lines.append("| target | d predicted | d measured | sign agreement |")
            lines.append("|---|---|---|---|")
            for _, r in nc.iterrows():
                lines.append(f"| {r['target']} | {_fmt(r['cohens_d_pred'])} | {_fmt(r['cohens_d_measured'])} | {r.get('sign_agreement', '')} |")
            lines.append("")
    rob = tables.get("perturbation_robustness")
    if rob is not None and len(rob) and {"perturbation", "target", "spearman_vs_clean"}.issubset(rob.columns):
        sub = rob[rob["target"] == main_target] if main_target in set(rob["target"]) else rob
        lines.append(f"Perturbation robustness on `{sub['target'].iloc[0]}`: Spearman between perturbed and clean predictions on the same nuclei.")
        lines.append("")
        lines.append("| perturbation | Spearman vs clean | mean absolute shift (z units) |")
        lines.append("|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['perturbation']} | {_fmt(r['spearman_vs_clean'])} | {_fmt(r.get('mean_abs_shift', float('nan')))} |")
        lines.append("")
    coh = tables.get("independent_cohort")
    if coh is not None and len(coh) and {"target", "pearson"}.issubset(coh.columns):
        lines.append("Independent cohort (per target, median over patients):")
        lines.append("")
        lines.append("| target | median Pearson | median shuffled |")
        lines.append("|---|---|---|")
        for t, grp in coh.groupby("target"):
            lines.append(f"| {t} | {_fmt(grp['pearson'].median())} | {_fmt(grp['pearson_shuffled'].median()) if 'pearson_shuffled' in grp else ''} |")
        lines.append("")
    lines.append("A skipped validator is a gap in the evidence, listed with its reason; the pass bars are the validators' own "
                 "stated criteria (docs/research/VALIDATION_METHODS.md) and the `external_validation` release gate reads them.")
    lines.append("")
    return lines


def write_contributions(run: RunDir, cfg: Config, res: pd.DataFrame, res_adj: Optional[pd.DataFrame],
                        reliability_summary: pd.DataFrame, decisions: List[str], verdict: Dict[str, bool],
                        main_target: str, burden_stats: Optional[Tuple[float, float, int]] = None,
                        multiseed: Optional[pd.DataFrame] = None, n_cells: int = 0, n_patients: int = 0,
                        srb_results: Optional[pd.DataFrame] = None, srb_rings: Optional[pd.DataFrame] = None,
                        manova: Optional[pd.DataFrame] = None, motifs: Optional[pd.DataFrame] = None,
                        codex: Optional[dict] = None, calibration: Optional[pd.DataFrame] = None,
                        gene_backbone: Optional[pd.DataFrame] = None, enrichment: Optional[pd.DataFrame] = None,
                        pathway_attr: Optional[pd.DataFrame] = None, gates: Optional[list] = None,
                        stage_assoc: Optional[pd.DataFrame] = None, validation: Optional[dict] = None,
                        validation_tables: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    lines: List[str] = []
    lines.append("# Contributions established by this run")
    lines.append("")
    lines.append(f"Run: `{cfg.run_name}`. Data: {'synthetic fixture' if cfg.demo_mode else cfg.geo_accession}, "
                 f"{n_patients} patients, {n_cells} cells after QC and subsampling. "
                 f"Evaluation: patient-level {cfg.n_folds}-fold cross-validation.")
    lines.append("")
    lines.append("## 1. A measured ceiling for every target")
    lines.append("")
    lines.append("Split-half reliability (binomial thinning, Spearman-Brown corrected) gives the correlation an "
                 "independent re-measurement of the same cells would reach. No image model can be expected to exceed "
                 "its square root.")
    lines.append("")
    lines.append("| target | median r_sb | attainable ceiling sqrt(r_sb) |")
    lines.append("|---|---|---|")
    for t, row in reliability_summary.iterrows():
        lines.append(f"| {t} | {_fmt(row['median'])} | {_fmt(row['ceiling_sqrt'])} |")
    lines.append("")
    lines.append("## 2. Held-out performance against baselines and ablations")
    lines.append("")
    cols = [c for c in res.columns if c != "full_95ci"]
    lines.append("| target | " + " | ".join(cols) + " | full 95% CI |")
    lines.append("|---|" + "---|" * (len(cols) + 1))
    for t, row in res.iterrows():
        lines.append(f"| {t} | " + " | ".join(_fmt(row[c]) for c in cols) + f" | {row.get('full_95ci', '')} |")
    lines.append("")
    if res_adj is not None:
        lines.append("Reliability-adjusted (r divided by sqrt of the median noise ceiling):")
        lines.append("")
        lines.append("| target | " + " | ".join(res_adj.columns) + " |")
        lines.append("|---|" + "---|" * len(res_adj.columns))
        for t, row in res_adj.iterrows():
            lines.append(f"| {t} | " + " | ".join(_fmt(row[c]) for c in res_adj.columns) + " |")
        lines.append("")
    if multiseed is not None and len(multiseed):
        lines.append("Seed stability of the full model (mean and standard deviation of the per-patient median r):")
        lines.append("")
        for t, row in multiseed.iterrows():
            lines.append(f"- {t}: {_fmt(row['mean'])} +/- {_fmt(row['std'])}")
        lines.append("")
    lines.append(f"## 3. Decision rules applied to `{main_target}`")
    lines.append("")
    for d in decisions:
        lines.append(f"- {d}")
    lines.append("")
    lines.append("## 4. What this run supports")
    lines.append("")
    if verdict.get("above_chance") and verdict.get("morphology_beyond_identity"):
        lines.append("- Nuclear morphology carries information about the resistance-associated program beyond what cell "
                     "identity alone explains (full model above both the shuffled-label bound and the composition-only baseline).")
    elif verdict.get("above_chance"):
        lines.append("- The model predicts the program above chance, but not beyond what a cell-type detector would. "
                     "The honest reading is that morphology recovers composition, not a within-type program signal.")
    else:
        lines.append("- The model does not predict the program above the shuffled-label bound on this cohort. "
                     "This is reported as a negative result; the pipeline, the ceiling estimate and the baselines are the contribution.")
    if verdict.get("identity_bottleneck"):
        lines.append("- The oracle gap shows cell identity is the bottleneck; the next lever is the encoder or patch size, not the head.")
    if not verdict.get("spatial_prior_useful", True):
        lines.append("- The spatial prior does not improve held-out correlation on this cohort and can be dropped without loss.")
    if burden_stats is not None:
        rho, p, n = burden_stats
        lines.append(f"- Patient-level program burden versus disease stage: Spearman rho = {rho:.3f}, p = {p:.3g} (n = {n}).")
    lines.append("")
    if srb_results is not None and len(srb_results):
        lines.append("## 5. Where in the neighbourhood the signal lives (spatially regularized boosting)")
        lines.append("")
        lines.append("Boosting over decision stumps on a grid of patch blocks, with the spatial regulariser of "
                     "Xiang et al. (2009); lambda = 0 is the AdaBoost control on the same stumps. Accuracy is the "
                     "per-patient median held-out metric; coherence is Moran's I of the importance map over the block "
                     "grid (about 0 for scattered selection, positive for clustered selection).")
        lines.append("")
        lines.append("| loss | lambda | metric | value | coherence (Moran's I) | active blocks |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in srb_results.iterrows():
            lines.append(f"| {r['loss']} | {r['lam']} | {r['metric']} | {_fmt(r['value'])} | {_fmt(r['coherence_mean'])} | {int(r['n_active_components'])} |")
        lines.append("")
        if srb_rings is not None and len(srb_rings):
            lines.append("Importance by block-centre distance from the nucleus (centre < 5 um, inner 5 to 10 um, outer >= 10 um); "
                         "share of total, and share per block in brackets:")
            lines.append("")
            lines.append("| model | plane | centre | inner ring | outer ring |")
            lines.append("|---|---|---|---|---|")
            for _, r in srb_rings.iterrows():
                cells = []
                for b in ("centre", "inner_ring", "outer_ring"):
                    per = r.get(f"per_block_{b}", float("nan"))
                    cells.append(f"{_fmt(r[b])} ({_fmt(per)})")
                lines.append(f"| {r['model']} | {r['plane']} | " + " | ".join(cells) + " |")
            lines.append("")
    if stage_assoc is not None and len(stage_assoc):
        lines.append("## 6. Association of the predicted program vector with disease stage (patient level)")
        lines.append("")
        lines.append("PERMANOVA (Anderson 2001; Euclidean distance between patient-level burden vectors, permutation "
                     "p-value) asks whether the vector of predicted program burdens differs between stages; the "
                     "Jonckheere-Terpstra trend test asks, per program, whether burden rises along the ordered stages, "
                     "with Benjamini-Hochberg q-values across programs. Both are distribution-free and suited to the "
                     "cohort size. Rows with `vectors = measured` repeat the tests on the measured programs, so the "
                     "predicted and the measured association can be compared directly.")
        lines.append("")
        show_cols = [c for c in ("vectors", "test", "program", "statistic", "p_value", "q_value", "r2_or_rho", "n_patients",
                                 "n_groups", "groups", "note") if c in stage_assoc.columns]
        lines.append("| " + " | ".join(show_cols) + " |")
        lines.append("|---|" * len(show_cols))
        for _, r in stage_assoc.iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in show_cols) + " |")
        lines.append("")
        skipped = stage_assoc["note"].astype(str).str.startswith("skipped") if "note" in stage_assoc.columns else None
        if skipped is not None and skipped.all():
            lines.append("Every stage test was skipped on this run (see the note column); the stage gate is not evaluated.")
        else:
            lines.append("With one trephine per patient the unit of analysis is the patient, so these tests have the "
                         "cohort's patient count as their sample size and are reported as supporting evidence only.")
        lines.append("")
    if manova is not None and len(manova):
        lines.append("## 6a. Classical MANOVA (kept for comparison; `evaluation.stage_test: manova` or `both`)")
        lines.append("")
        cols = [c for c in manova.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|---|" * len(cols))
        for _, r in manova.iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
        n_pat_col = next((c for c in manova.columns if c.endswith("n_patients")), None)
        n_grp_col = next((c for c in manova.columns if c.endswith("n_groups")), None)
        n_pat = int(manova[n_pat_col].dropna().max()) if n_pat_col and manova[n_pat_col].notna().any() else None
        n_grp = int(manova[n_grp_col].dropna().max()) if n_grp_col and manova[n_grp_col].notna().any() else None
        design = (f"with {n_pat} patients in {n_grp} groups the test" if n_pat and n_grp else "the test")
        lines.append("Pillai's trace is reported because it is the most robust MANOVA statistic to unequal group sizes; "
                     "the permutation p-value does not assume multivariate normality. Stages are collapsed to control, precursor and "
                     f"myeloma and the composite set is omitted when its two subsets are present; {design} is at the lower limit "
                     "of interpretability and is reported as supporting evidence only.")
        lines.append("")
    if motifs is not None and len(motifs):
        lines.append("## 7. Neighbourhood motifs enriched for the main program (byte-pair encoding over lineage sequences)")
        lines.append("")
        top = motifs.head(8)
        lines.append("| motif | difference (with minus without) | permutation p | patients with motif | fraction of patients with positive difference |")
        lines.append("|---|---|---|---|---|")
        for _, r in top.iterrows():
            lines.append(f"| {r.get('motif', '')} | {_fmt(r.get('difference', float('nan')))} | {_fmt(r.get('perm_p', float('nan')))} | "
                         f"{r.get('n_patients_with_motif', '')} | {_fmt(r.get('frac_patients_consistent', float('nan')))} |")
        lines.append("")
        lines.append("Motif discovery is exploratory: byte-pair encoding is a tokenisation heuristic applied to ordered "
                     "neighbour sequences, and enrichment is descriptive.")
        lines.append("")
    if validation:
        lines.extend(_external_validation_section(validation, validation_tables or {}, main_target))
    if codex and str(codex.get("status", "")) == "ran":
        lines.append("## 8. CODEX spatial-proteomics bridge")
        lines.append("")
        comp = codex.get("composition", {}) or {}
        lines.append(f"- CODEX cells: {codex.get('n_codex_cells', 'n/a')}; nuclei detected on the DAPI channel: "
                     f"{codex.get('n_detected', 'n/a')}; matched within {codex.get('match_um', 8)} um: {codex.get('n_matched', 'n/a')}")
        if comp:
            lines.append(f"- composition agreement (model head vs protein lineage): CLR Pearson r = {_fmt(comp.get('clr_pearson_r'))}, "
                         f"Aitchison distance = {_fmt(comp.get('aitchison_distance'))}, per-cell agreement = {_fmt(comp.get('cell_agreement'))}")
        contrast = codex.get("niche_contrast") or []
        if contrast:
            lines.append("")
            lines.append("| predicted program | mean in plasma-dense niche | mean elsewhere | Cohen's d |")
            lines.append("|---|---|---|---|")
            for row in contrast:
                lines.append(f"| {row.get('program', '')} | {_fmt(row.get('mean_dense', float('nan')))} | "
                             f"{_fmt(row.get('mean_other', float('nan')))} | {_fmt(row.get('cohens_d', float('nan')))} |")
        if codex.get("caveat"):
            lines.append("")
            lines.append(f"Caveat: {codex['caveat']}.")
        lines.append("")
    if calibration is not None and len(calibration):
        lines.append("## 8a. Calibration of the predictive uncertainty")
        lines.append("")
        lines.append("Empirical coverage of the predictive intervals on held-out patients. A well calibrated "
                     "model covers the target at the nominal level; coverage is reported per patient because "
                     "cells within one patient are not exchangeable with cells of another.")
        lines.append("")
        cols = list(calibration.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|---|" * len(cols))
        for _, r in calibration.head(40).iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    if gene_backbone is not None and len(gene_backbone):
        lines.append("## 8b. Gene backbone of each program")
        lines.append("")
        lines.append("Genes whose learned weight survives stability selection over patient subsamples, with the "
                     "literature evidence grade from resources/gene_annotations.json. A backbone gene whose "
                     "evidence points toward sensitivity rather than resistance is shown with its grade so the "
                     "reader can weigh it.")
        lines.append("")
        cols = [c for c in ("program", "gene", "selection_fraction", "weight_relative_to_max", "evidence_grade", "gene_class") if c in gene_backbone.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|---|" * len(cols))
        prog_col = "program" if "program" in gene_backbone.columns else None
        shown = gene_backbone.groupby(prog_col).head(10) if prog_col else gene_backbone.head(40)
        for _, r in shown.iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    if enrichment is not None and len(enrichment):
        lines.append("## 8c. Pathway annotation of the learned programs")
        lines.append("")
        lines.append("Hypergeometric enrichment of each learned program's backbone against the curated gene sets, "
                     "with the measured panel as background.")
        lines.append("")
        cols = list(enrichment.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|---|" * len(cols))
        for _, r in enrichment.head(40).iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    if pathway_attr is not None and len(pathway_attr):
        lines.append("## 8d. Where the model looks (pathway attribution)")
        lines.append("")
        lines.append("Cross-attention mass that each program token places on the centre patch, the neighbour "
                     "blocks and the neighbour cells, averaged over held-out cells. Attention mass is descriptive, "
                     "not causal.")
        lines.append("")
        cols = list(pathway_attr.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|---|" * len(cols))
        for _, r in pathway_attr.head(40).iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    if gates:
        from .gates import gates_markdown
        lines.append("## 9. Release gates")
        lines.append("")
        lines.append("Each gate is read from one artefact of this run and nothing is recomputed; a gate whose artefact "
                     "or column is absent is NOT EVALUATED, never passed. On the synthetic fixture the gates test the "
                     "plumbing, not the biology.")
        lines.append("")
        lines.extend(gates_markdown(gates))
        n_pass = sum(g.passed is True for g in gates)
        n_fail = sum(g.passed is False for g in gates)
        lines.append("")
        lines.append(f"{n_pass} of {len(gates)} gates passed, {n_fail} failed, {len(gates) - n_pass - n_fail} not evaluated "
                     "(`reporting/release_gates.csv`, `RESULTS.json`).")
        lines.append("")
    lines.append("## 10. What this run does not establish")
    lines.append("")
    if cfg.clinical_csv:
        lines.append(f"- Clinical endpoints were read from `{cfg.clinical_csv}`; see `evaluation/cox_pfs.csv`.")
    else:
        lines.append("- No clinical endpoint (progression-free survival) was tested: no clinical table was supplied.")
    stain = getattr(cfg, "stain_mode", "native")
    if stain == "hematoxylin":
        lines.append("- The morphology channel was harmonised to a hematoxylin-like image (`stain_mode: hematoxylin`); "
                     "transfer to true H&E still needs an H&E cohort.")
    else:
        lines.append(f"- The morphology channel is the native DAPI image (`stain_mode: {stain}`); no H&E slide was used. "
                     "Transfer to H&E requires the stain-harmonisation phase and an H&E cohort.")
    lines.append("- Learned neighbourhood-scale targets average the cell-scale program usage over the cells kept for training "
                 "(after subsampling), while the fixed-set neighbourhood targets were summed at ingest over every QC-passing "
                 "cell; the two target families therefore measure neighbourhoods of different cell density (DESIGN_V3.md D28).")
    lines.append("- The gene sets are investigator-defined gene-set abundances, not pathway activity. Caveats recorded with the "
                 "gene annotations (`resources/gene_annotations.json`, docs/research/PATHWAY_INTERPRETATION.md):")
    for caveat in load_annotation_caveats():
        lines.append(f"  - {caveat}")
    lines.append("- Single trephine per patient: patient-level and slide-level cross-validation coincide.")
    text = "\n".join(lines) + "\n"
    with open(run.art("contributions"), "w") as fh:
        fh.write(text)
    say(f"contributions -> {run.art('contributions')}")
    for ln in lines[:40]:
        say(ln, indent=1)
    return text
