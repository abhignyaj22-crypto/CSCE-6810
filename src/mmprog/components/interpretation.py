"""Interpretation component: where the signal lives, what the model attended to, which genes form the backbone.

Four outputs, each with its own print block and artefact:
1. Spatially regularized boosting over patch blocks (Xiang et al., NIPS 2009, Advances in Neural Information Processing Systems 22; D25),
   lambda = 0 as the AdaBoost control, with the centre / inner / outer ring summary.
2. Cross-attention mass of every program token on token groups, and integrated gradients
   from each target head; both descriptive (``interpretation/attribution.py``).
3. Gene backbone of each program by stability selection over patient-subsampled refits
   of the anchored factorisation, annotated with the literature evidence grades, and
   hypergeometric enrichment of the backbone against the curated gene sets.
4. Neighbourhood motifs by byte-pair encoding (exploratory, ``analysis/motifs.py``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, section, show_table, warn

MAIN_TARGET_PREFERENCE = ["prog_wnt_camdr_nb_z", "anchor_wnt_camdr_nb_z", "prog_wnt_camdr_cell_z"]


def main_target_of(targets: List[str]) -> str:
    for t in MAIN_TARGET_PREFERENCE:
        if t in targets:
            return t
    return targets[0]


class _ArtAdapter:
    """Minimal RunDir look-alike for v2 helpers that call ``run.art(name)``."""

    def __init__(self, base: str, names: Dict[str, str]):
        self.base, self.names = base, names

    def art(self, name: str) -> str:
        return os.path.join(self.base, self.names.get(name, name))

    def done(self, name: str) -> bool:
        return os.path.exists(self.art(name))


def grid_block_features_from_tokens(GRID: np.ndarray, grid: int) -> np.ndarray:
    """(N, 2*g*g) per-block mean and std over the token embedding dims: the token-space analogue
    of the pixel-grid features, used only when the morphology images are not available."""
    G = np.asarray(GRID, dtype=np.float32)
    means = G.mean(-1)
    stds = G.std(-1)
    say(f"grid features from GRID tokens: {G.shape} -> block mean/std over {G.shape[-1]} token dims "
        f"(token-space analogue of the v2 pixel-grid features)")
    return np.concatenate([means, stds], axis=1)


class Interpretation(BaseComponent):
    name = "interpretation"
    # MOR, NB, adata, programs and images are read with guards; EMB is the attribution input and is required
    requires = ("GRID", "obs", "Y", "targets", "SID", "GROUPS", "program_weights", "final_model_path", "EMB")
    provides = ("srb_results", "srb_maps", "srb_rings", "pathway_attr", "gene_backbone", "enrichment", "motifs")
    artefacts = ("srb_results.csv", "srb_maps.npz", "srb_rings.csv", "pathway_attribution.csv", "gene_backbone.csv",
                 "enrichment.csv", "motif_enrichment.csv")
    config_fields = ("srb_enabled", "srb_grid", "srb_lambdas", "srb_radius_um", "srb_rounds", "srb_max_train_cells",
                     "attribution_n_cells", "ig_steps", "backbone_threshold", "backbone_min_fold_fraction",
                     "motifs_enabled", "patch_px", "knn_k", "n_folds", "min_cells_per_patient_metric",
                     "um_per_px_level0", "img_level", "nb_radius_um")
    depends_on = ("training", "program_inference")

    # ------------------------------------------------------------------ steps
    def _spatial_boost(self, ctx: Context) -> None:
        from ..evaluation.spatial_boost_eval import ring_summary, run_spatial_boost_cv
        from ..features.patch_grid import grid_locations_um, patch_grid_features

        section("interpretation 1/4: spatially regularized boosting on patch blocks")
        cfg = ctx.cfg
        grid = int(cfg.srb_grid)
        target = main_target_of(ctx.targets)
        j = ctx.targets.index(target)
        y = np.asarray(ctx.Y[:, j], dtype=np.float64)
        if ctx.images is not None:
            say("morphology images available: pixel-grid block features (exact parity with v2)")
            G = patch_grid_features(ctx.obs, ctx.images, cfg, grid=grid)
        else:
            if np.asarray(ctx.GRID).shape[1] != grid * grid:
                grid = int(round(np.sqrt(np.asarray(ctx.GRID).shape[1])))
                say(f"GRID holds {grid * grid} blocks; srb_grid adjusted to {grid}")
            G = grid_block_features_from_tokens(ctx.GRID, grid)
        locations = grid_locations_um(grid, cfg.patch_px, cfg.um_per_px)
        if not cfg.srb_enabled:
            say("srb_enabled is false: writing empty boosting artefacts")
            ctx.set("srb_results", pd.DataFrame())
            ctx.set("srb_maps", {})
            ctx.set("srb_rings", pd.DataFrame())
        else:
            res, maps, _ = run_spatial_boost_cv(G, locations, y, ctx.SID, ctx.GROUPS, cfg, grid,
                                                lambdas=tuple(cfg.srb_lambdas), radius_um=cfg.srb_radius_um,
                                                n_rounds=cfg.srb_rounds, max_train_cells=cfg.srb_max_train_cells,
                                                seed=cfg.seed)
            block_um = cfg.patch_px / grid * cfg.um_per_px
            rings = []
            for k, m in maps.items():
                r = ring_summary(m, grid, block_um)
                r.insert(0, "model", k)
                rings.append(r)
            ctx.set("srb_results", res.assign(target=target))
            ctx.set("srb_maps", maps)
            ctx.set("srb_rings", pd.concat(rings, ignore_index=True))
            show_table(ctx.srb_rings, title="share of importance by distance from the nucleus:")
        ctx.srb_results.to_csv(self.path(ctx, "srb_results.csv"), index=False)
        np.savez_compressed(self.path(ctx, "srb_maps.npz"), **{k: np.asarray(v) for k, v in ctx.srb_maps.items()})
        ctx.srb_rings.to_csv(self.path(ctx, "srb_rings.csv"), index=False)
        say(f"boosting target {target}; artefacts srb_results.csv, srb_maps.npz, srb_rings.csv")

    def _attribution(self, ctx: Context) -> None:
        from ..inference.virtual_map_v3 import load_checkpoint
        from ..interpretation.attribution import (attention_attribution, default_token_groups, integrated_gradients,
                                                  summarise_attributions)

        section("interpretation 2/4: attention attribution and integrated gradients of the unified model")
        cfg = ctx.cfg
        ck = load_checkpoint(ctx.final_model_path, ctx.device)
        model, spec = ck["model"], ck.get("spec") or {}
        program_names = list(spec.get("program_names") or spec.get("programs") or ctx.program_weights.index)
        grid = int(spec.get("grid", cfg.srb_grid))
        k = int(spec.get("knn_k", cfg.knn_k))
        n = len(ctx.obs)
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(n, min(n, int(cfg.attribution_n_cells)), replace=False))
        xy = ctx.obs[["x", "y"]].to_numpy(dtype=np.float32)
        NB = ctx.NB if ctx.NB is not None else np.repeat(np.arange(n)[:, None], k, axis=1)
        fs = ck.get("fs") or ck.get("feature_space")
        # the model's centre and neighbour inputs are the feature-space vectors used in training
        if fs is not None and ctx.MOR is not None:
            X = np.asarray(fs.transform(ctx.EMB, ctx.MOR), dtype=np.float32)
            say(f"feature space applied to all {n} cells for attribution: {X.shape}")
        else:
            X = np.asarray(ctx.EMB, dtype=np.float32)
        arrays: Dict[str, Any] = dict(
            centre=X[idx], grid=np.asarray(ctx.GRID[idx], dtype=np.float32), nb=X[NB[idx]],
            nb_offset=((xy[NB[idx]] - xy[idx][:, None, :]) * ctx.um_per_px_per_cell()[idx][:, None, None]).astype(np.float32),
            nb_mask=np.ones((len(idx), NB.shape[1]), dtype=bool),
            sid=np.asarray(ctx.SID[idx], dtype=np.int64),
            morpho=np.asarray(ctx.MOR[idx], dtype=np.float32) if ctx.MOR is not None else None,
        )
        if ck.get("flags"):
            arrays["flags"] = ck["flags"]
        groups = default_token_groups(grid, cfg.patch_px / grid * cfg.um_per_px, NB.shape[1])
        # groups refer to the token order [centre, g*g blocks, k neighbours]; a model with fewer tokens
        # (ablation) returns fewer attention columns, so groups are clipped to what it returned.
        lineage_names = ctx.obs["lineage_name"].to_numpy() if "lineage_name" in ctx.obs else None
        try:
            # pass only arrays with at least one element per row; position groups are clipped inside on the first batch
            attr = attention_attribution(model, {k_: v for k_, v in arrays.items() if v is not None}, ctx.targets,
                                         program_names, groups, device=ctx.device, max_cells=len(idx),
                                         lineage_names=lineage_names, seed=cfg.seed)
        except IndexError:
            warn("token groups exceed the attention width of this model (ablation variant?); attribution over the "
                 "available tokens only")
            attr = pd.DataFrame(columns=["program", "group", "lineage", "mass", "n_cells"])
        ig_rows = []
        try:
            small = np.arange(min(len(idx), 256))
            # flags is a dict, not a per-cell array; everything else is sliced to the first cells
            sub = {k_: (v if k_ == "flags" else np.asarray(v)[small]) for k_, v in arrays.items() if v is not None}
            # head column j is ck["targets"][j], not ctx.targets[j]: the head order is a permutation of the
            # context order and ctx.targets also holds the anchor-family columns the network never predicts
            head_targets = list(ck.get("targets") or ctx.targets)
            say(f"integrated gradients for {len(head_targets)} head targets on {len(small)} cells", indent=1)
            for j, t in enumerate(head_targets):
                ig = integrated_gradients(model, sub, j, attr_keys=("centre",), steps=int(cfg.ig_steps), device=ctx.device)
                s = summarise_attributions(ig, top=10)
                s.insert(0, "target", t)
                ig_rows.append(s)
        except Exception as e:
            warn(f"integrated gradients skipped ({type(e).__name__}: {str(e)[:80]})")
        attr["kind"] = "attention_mass"
        if ig_rows:
            ig_df = pd.concat(ig_rows, ignore_index=True)
            ig_df["kind"] = "integrated_gradients"
            attr = pd.concat([attr, ig_df], ignore_index=True)
        ctx.set("pathway_attr", attr)
        attr.to_csv(self.path(ctx, "pathway_attribution.csv"), index=False)
        say(f"pathway attribution: {len(attr)} rows -> pathway_attribution.csv (attention mass is descriptive, not causal)")

    def _backbone(self, ctx: Context) -> None:
        section("interpretation 3/4: gene backbone by stability selection, with enrichment")
        cfg = ctx.cfg
        H_full = ctx.program_weights                            # (P programs x G genes), index = program names
        backbone: Optional[pd.DataFrame] = None
        if ctx.adata is not None:
            try:
                from ..interpretation.backbone import stability_selection
                from .program_inference import ProgramInference
                refit = ProgramInference()

                def fit_fn(cell_index: np.ndarray) -> pd.DataFrame:
                    return refit.fit_nmf(ctx, cell_index).H_

                backbone = stability_selection(fit_fn, ctx.obs, H_full, threshold=cfg.backbone_threshold,
                                               min_fold_fraction=cfg.backbone_min_fold_fraction, seed=cfg.seed)
            except ImportError as e:
                say(f"stability selection unavailable ({e}); backbone from the single factorisation fit")
        else:
            say("no expression matrix in the context (loaded run): backbone from the single factorisation fit")
        if backbone is None or len(backbone) == 0:
            backbone = self._annotate(self._backbone_from_weights(H_full, cfg.backbone_threshold))
        ctx.set("gene_backbone", backbone)
        backbone.to_csv(self.path(ctx, "gene_backbone.csv"), index=False)
        show_table(backbone.sort_values(["program"]).groupby("program").head(6).drop(columns=[c for c in ("note",) if c in backbone]),
                   title="backbone genes per program (top 6):")
        # enrichment of each backbone against the curated sets, with the panel as background
        sets = dict(ctx.programs or {})
        # background = the measured panel genes that pass the gene mask; masked genes carry zero
        # weight and can never enter a backbone, so counting them would inflate M
        gene_mask = getattr(ctx, "gene_mask", None)
        if gene_mask is not None and len(gene_mask) == H_full.shape[1]:
            background = list(H_full.columns[np.asarray(gene_mask, dtype=bool)])
            say(f"enrichment background: {len(background)} panel genes passing the gene mask", indent=1)
        else:
            background = list(H_full.columns)
            say(f"enrichment background: all {len(background)} panel genes (no gene mask available)", indent=1)
        by_program = {p: backbone.loc[backbone.program == p, "gene"].tolist() for p in backbone["program"].unique()}
        if by_program:
            try:
                from ..interpretation.enrichment import hypergeometric_enrichment
                enrich = hypergeometric_enrichment(by_program, sets, background)
            except ImportError:
                enrich = pd.concat([_hypergeometric(g, sets, background).assign(program=p) for p, g in by_program.items()],
                                   ignore_index=True)
        else:
            enrich = pd.DataFrame(columns=["program", "gene_set", "overlap", "p_value"])
        ctx.set("enrichment", enrich)
        enrich.to_csv(self.path(ctx, "enrichment.csv"), index=False)

    @staticmethod
    def _backbone_from_weights(H: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Backbone from one fit: genes whose weight is at least ``threshold`` of the program's maximum."""
        rows = []
        for p in H.index:
            w = H.loc[p].to_numpy(dtype=np.float64)
            norm = w / (np.abs(w).max() + 1e-12)
            for g, v, nv in zip(H.columns, w, norm):
                if nv >= threshold:
                    rows.append(dict(program=p, gene=g, selection_fraction=np.nan, weight_relative_to_max=float(nv),
                                     method="single_fit_threshold"))
        df = pd.DataFrame(rows, columns=["program", "gene", "selection_fraction", "weight_relative_to_max", "method"])
        say(f"backbone from the single fit: {len(df)} gene-program pairs with normalised weight >= {threshold}")
        return df

    @staticmethod
    def _annotate(backbone: pd.DataFrame) -> pd.DataFrame:
        try:
            from ..resources import load_gene_annotations
            ann = load_gene_annotations()
            flat = {g: a for prog in ann.values() for g, a in prog.items()}
        except Exception:
            flat = {}
        backbone = backbone.copy()
        backbone["gene_class"] = [flat.get(g, {}).get("class", "") for g in backbone["gene"]]
        backbone["evidence_grade"] = [flat.get(g, {}).get("evidence_grade", "") for g in backbone["gene"]]
        backbone["is_wnt_component"] = [bool(flat.get(g, {}).get("is_wnt_component", False)) for g in backbone["gene"]]
        n_ann = int((backbone["evidence_grade"] != "").sum())
        say(f"backbone annotated with literature evidence grades for {n_ann}/{len(backbone)} entries "
            "(resources/gene_annotations.json)")
        return backbone

    def _motifs(self, ctx: Context) -> None:
        section("interpretation 4/4: neighbourhood motifs (byte-pair encoding, exploratory)")
        cfg = ctx.cfg
        adapter = _ArtAdapter(self.dir(ctx), {"motifs": "motif_enrichment.csv", "motif_vocab": "motif_vocabulary.json"})
        if not cfg.motifs_enabled or "lineage_name" not in ctx.obs:
            say("motifs_enabled is false or no lineage labels: writing an empty motif table")
            ctx.set("motifs", pd.DataFrame())
            ctx.motifs.to_csv(adapter.art("motifs"), index=False)
            return
        from ..analysis.motifs import run_motif_stage
        target = main_target_of(ctx.targets)
        obs = ctx.obs
        if target not in obs:
            obs = obs.copy()
            obs[target] = ctx.Y[:, ctx.targets.index(target)]
        motifs = run_motif_stage(obs, target, cfg, adapter, nb_table=ctx.NB)
        if motifs is None or len(motifs) == 0:
            motifs = pd.DataFrame()
            motifs.to_csv(adapter.art("motifs"), index=False)
        ctx.set("motifs", motifs)

    # ------------------------------------------------------------------ protocol
    def _run(self, ctx: Context) -> None:
        self._spatial_boost(ctx)
        self._attribution(ctx)
        self._backbone(ctx)
        self._motifs(ctx)
        from ..observations import observe
        bb = ctx.gene_backbone
        examples = []
        if bb is not None and len(bb):
            top = bb.sort_values("weight_relative_to_max", ascending=False).head(5)
            examples = [f"{r.program} {r.gene} relative weight {r.weight_relative_to_max:.2f}" for r in top.itertuples()]
        observe(ctx.cfg.run_dir, "interpretation",
                before="trained model, program weights, and the spatial feature grid",
                after=f"backbone {0 if bb is None else len(bb)} gene-program rows; "
                      f"boosting {0 if ctx.srb_results is None else len(ctx.srb_results)} rows",
                note="a backbone gene survived the weight threshold; the evidence grade says what the literature supports",
                examples=examples)

    def load(self, ctx: Context) -> None:
        ctx.set("srb_results", pd.read_csv(self.path(ctx, "srb_results.csv")))
        ctx.set("srb_maps", dict(np.load(self.path(ctx, "srb_maps.npz"))))
        ctx.set("srb_rings", pd.read_csv(self.path(ctx, "srb_rings.csv")))
        ctx.set("pathway_attr", pd.read_csv(self.path(ctx, "pathway_attribution.csv")))
        ctx.set("gene_backbone", pd.read_csv(self.path(ctx, "gene_backbone.csv")))
        ctx.set("enrichment", pd.read_csv(self.path(ctx, "enrichment.csv")))
        motif_path = self.path(ctx, "motif_enrichment.csv")
        ctx.set("motifs", pd.read_csv(motif_path) if os.path.getsize(motif_path) > 1 else pd.DataFrame())
        say(f"interpretation loaded: boosting {len(ctx.srb_results)} rows, backbone {len(ctx.gene_backbone)} genes, "
            f"motifs {len(ctx.motifs)} rows")


def _hypergeometric(genes: List[str], sets: Dict[str, List[str]], background: List[str]) -> pd.DataFrame:
    """Fallback over-representation test: P(overlap >= k) under sampling without replacement."""
    from scipy.stats import hypergeom
    bg = set(background)
    sel = set(genes) & bg
    rows = []
    for name, members in sets.items():
        m = set(members) & bg
        k = len(sel & m)
        p = float(hypergeom.sf(k - 1, len(bg), len(m), len(sel))) if len(sel) and len(m) else 1.0
        rows.append(dict(gene_set=name, overlap=k, set_size_in_panel=len(m), selected=len(sel), p_value=p,
                         overlap_genes=",".join(sorted(sel & m))))
    df = pd.DataFrame(rows)
    say(f"hypergeometric enrichment of {len(sel)} genes against {len(sets)} sets, background {len(bg)} panel genes")
    return df


def interpret_programs(ctx: Context) -> Context:
    """Component 9: spatially regularized boosting, attribution, the gene backbone of each program.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return Interpretation().run_in(ctx)
