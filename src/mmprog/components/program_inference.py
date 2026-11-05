"""ProgramInference: learn the gene programs, build the targets, measure their ceiling, label lineages.

Order of work and why
1. Print the anchor gene sets with their evidence grades (D24), so the run log says what
   the programs are built from.
2. Fit the anchored NMF once on all QC-passing cells (D28): unsupervised, no morphology,
   no labels. The dependence of the target definition on test-patient expression is the
   same dependence a published gene set carries and is stated in the report.
3. Build targets at the cell and neighbourhood scales for the learned programs and, when
   count summaries exist, for the fixed sets (``anchor_only`` family).
4. Split-half reliability for both families: count thinning on the fixed sets (exact), and
   thinning followed by re-projection for the learned programs.
5. Lineage labels by marker-score argmax (D10) for the composition head.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, section
from ..programs.anchored_nmf import AnchoredNMF
from ..programs.targets_builder import build_targets, learned_reliability, panel_coverage
from ..resources import load_lineages, load_programs, print_program_annotations
from ..targets.programs import score_lineages
from ..targets.reliability import reliability_lookup, reliability_summary, reliability_table


class ProgramInference(BaseComponent):
    name = "program_inference"
    requires = ("adata", "obs", "gene_mask", "images")
    provides = ("programs", "program_weights", "program_scores", "targets", "target_family", "Y", "lineage",
                "lin_names", "LIN", "reliability", "reliability_summary", "rel_lookup", "expression_embedding")
    artefacts = ("program_weights.csv", "program_scores.npy", "targets.json", "Y.npy", "lineage.csv",
                 "target_reliability.csv", "reliability_summary.csv", "panel_coverage.csv", "expression_embedding.npy")
    depends_on = ("preprocess",)
    config_fields: Tuple[str, ...] = ("programs_n_extra", "programs_anchor_prior", "programs_l1_nonanchor", "programs_max_iter",
                                      "programs_fit_cells", "include_anchor_only_targets", "target_spaces", "target_norm",
                                      "reliability_seed", "min_program_genes", "nb_radius_um")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _counts(adata) -> csr_matrix:
        C = adata.layers["counts"] if "counts" in adata.layers else adata.X
        return C.tocsr() if issparse(C) else csr_matrix(C)

    def fit_nmf(self, ctx: Context, cell_index: np.ndarray | None = None) -> AnchoredNMF:
        """Fit the anchored NMF on all cells or on ``cell_index`` (used by stability selection)."""
        cfg = ctx.cfg
        adata = ctx.adata
        mask = np.asarray(ctx.gene_mask, dtype=bool)
        X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
        if cell_index is not None:
            X = X[cell_index]
        anchors = {p: [g for g in genes if g in set(adata.var_names[mask])] for p, genes in ctx.programs.items()}
        anchors = {p: g for p, g in anchors.items() if len(g) >= cfg.min_program_genes}
        nmf = AnchoredNMF(n_extra=cfg.programs_n_extra, anchor_prior=cfg.programs_anchor_prior,
                          l1_nonanchor=cfg.programs_l1_nonanchor, max_iter=cfg.programs_max_iter,
                          fit_cells=cfg.programs_fit_cells, seed=cfg.seed)
        return nmf.fit(X[:, mask], adata.var_names[mask], anchors)

    # ------------------------------------------------------------------ run
    def _run(self, ctx: Context) -> None:
        cfg = ctx.cfg
        adata = ctx.adata
        programs = load_programs()
        ctx.set("programs", programs)
        section("anchor gene sets and evidence grades")
        print_program_annotations(programs)
        coverage = panel_coverage(programs, adata.var_names, ctx.gene_mask)
        coverage.to_csv(self.path(ctx, "panel_coverage.csv"), index=False)

        nmf = self.fit_nmf(ctx)
        # H is stored over the full panel (zeros outside the mask) so downstream code needs no mask
        H_full = nmf.H_.reindex(columns=adata.var_names, fill_value=0.0)
        H_full.to_csv(self.path(ctx, "program_weights.csv"))
        ctx.set("program_weights", H_full)
        self._nmf = nmf

        um_by_sample = ctx.um_per_px_by_sample()
        say("microns per working-level pixel by slide: " + ", ".join(f"{k}={v:.4f}" for k, v in list(um_by_sample.items())[:6])
            + (" ..." if len(um_by_sample) > 6 else ""))
        targets, family, Y, scores, emb, obs = build_targets(adata, ctx.obs, nmf, programs, cfg, ctx.gene_mask, um_by_sample)
        say(f"program scores {scores.shape} (cells x programs x scales); Y {Y.shape}")
        from ..observations import observe
        names = list(getattr(nmf, "program_names_", []))
        examples = []
        if scores.size and names:
            means = scores.reshape(scores.shape[0], -1).mean(axis=0)
            flat_names = [f"{p}/{s}" for p in names for s in cfg.target_spaces][:len(means)]
            examples = [f"{n} mean {m:.3f}" for n, m in list(zip(flat_names, means))[:5]]
        observe(cfg.run_dir, "program_scores",
                before=f"log-normalised expression of {adata.n_vars} genes in {adata.n_obs} cells",
                after=f"scores {tuple(scores.shape)} and model targets {tuple(Y.shape)}",
                note="scores are gene-set abundances, not a measured drug response",
                examples=examples)

        section("reliability of both target families")
        tabs = []
        anchor_programs = {p: g for p, g in programs.items() if f"cnt_{p}" in obs}
        if cfg.include_anchor_only_targets and anchor_programs:
            anchor_tab = reliability_table(obs, anchor_programs, cfg)
            anchor_tab["target"] = anchor_tab["target"].str.replace("^prog_", "anchor_", regex=True)
            tabs.append(anchor_tab)
        tabs.append(learned_reliability(self._counts(adata), nmf, adata.var_names, ctx.gene_mask, obs, cfg, cfg.reliability_seed,
                                        um_by_sample))
        reliability = pd.concat(tabs, ignore_index=True)
        reliability = reliability[reliability["target"].isin(targets)]
        reliability.to_csv(self.path(ctx, "target_reliability.csv"), index=False)
        summary = reliability_summary(reliability)
        summary.to_csv(self.path(ctx, "reliability_summary.csv"))

        section("lineage labels for the composition head")
        adata.obs = obs.set_index(adata.obs.index)
        adata, lin_names = score_lineages(adata, load_lineages(), cfg)
        obs = adata.obs.reset_index(drop=True)
        lineage = obs["lineage"].to_numpy().astype(np.int64)
        LIN = np.zeros((len(obs), len(lin_names)), dtype=np.float32)
        LIN[np.arange(len(obs)), lineage] = 1.0
        obs[["lineage", "lineage_name"]].to_csv(self.path(ctx, "lineage.csv"), index=False)

        np.save(self.path(ctx, "program_scores.npy"), scores)
        np.save(self.path(ctx, "Y.npy"), Y)
        np.save(self.path(ctx, "expression_embedding.npy"), emb)
        with open(self.path(ctx, "targets.json"), "w") as fh:
            json.dump(dict(targets=targets, target_family=family, program_names=nmf.program_names_,
                           anchors=nmf.anchors_, lineages=lin_names, scales=list(cfg.target_spaces),
                           main_target="prog_wnt_camdr_nb_z" if "prog_wnt_camdr_nb_z" in targets else targets[0]), fh, indent=2)
        ctx.set("adata", adata)
        ctx.set("obs", obs)
        ctx.set("program_scores", scores)
        ctx.set("targets", targets)
        ctx.set("target_family", family)
        ctx.set("Y", Y)
        ctx.set("lineage", lineage)
        ctx.set("lin_names", lin_names)
        ctx.set("LIN", LIN)
        ctx.set("reliability", reliability)
        ctx.set("reliability_summary", summary)
        ctx.set("rel_lookup", reliability_lookup(reliability))
        ctx.set("expression_embedding", emb)
        say(f"main target: {json.load(open(self.path(ctx, 'targets.json')))['main_target']}")

    # ------------------------------------------------------------------ load
    def load(self, ctx: Context) -> None:
        meta = json.load(open(self.path(ctx, "targets.json")))
        H = pd.read_csv(self.path(ctx, "program_weights.csv"), index_col=0)
        Y = np.load(self.path(ctx, "Y.npy"))
        scores = np.load(self.path(ctx, "program_scores.npy"))
        emb = np.load(self.path(ctx, "expression_embedding.npy"))
        lin = pd.read_csv(self.path(ctx, "lineage.csv"))
        reliability = pd.read_csv(self.path(ctx, "target_reliability.csv"))
        summary = pd.read_csv(self.path(ctx, "reliability_summary.csv"), index_col=0)
        obs = ctx.obs.copy()
        targets = meta["targets"]
        for j, t in enumerate(targets):
            obs[t] = Y[:, j]
        obs["lineage"] = lin["lineage"].to_numpy()
        obs["lineage_name"] = lin["lineage_name"].to_numpy()
        lineage = obs["lineage"].to_numpy().astype(np.int64)
        LIN = np.zeros((len(obs), len(meta["lineages"])), dtype=np.float32)
        LIN[np.arange(len(obs)), lineage] = 1.0
        ctx.set("programs", load_programs())
        ctx.set("program_weights", H)
        ctx.set("program_scores", scores)
        ctx.set("targets", targets)
        ctx.set("target_family", meta["target_family"])
        ctx.set("Y", Y)
        ctx.set("obs", obs)
        ctx.set("lineage", lineage)
        ctx.set("lin_names", meta["lineages"])
        ctx.set("LIN", LIN)
        ctx.set("reliability", reliability)
        ctx.set("reliability_summary", summary)
        ctx.set("rel_lookup", reliability_lookup(reliability))
        ctx.set("expression_embedding", emb)
        say(f"loaded {len(targets)} targets, {H.shape[0]} programs, {len(meta['lineages'])} lineages from {self.dir(ctx)}")
        if not os.path.exists(self.path(ctx, "manifest.json")):
            say("note: no manifest found; artefacts loaded without hash check")


def infer_gene_programs(ctx: Context) -> Context:
    """Component 3: anchored factorisation into gene programs, cell- and neighbourhood-scale targets, reliability ceilings, lineage labels.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return ProgramInference().run_in(ctx)
