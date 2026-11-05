"""UnifiedModel component: build (not train) the network and write its specification.

The specification holds every dimension the training and deployment code needs to
reconstruct the network from a checkpoint: token dimensions, grid, k, programs and
scales in target order, lineages, slide count, uncertainty kind, and the parameter
count of every block from a dry construction on CPU.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from ..core.component import BaseComponent
from ..core.context import Context
from ..logging_utils import say, warn


def target_layout(targets: List[str], target_family: Dict[str, str] | None) -> Tuple[List[str], List[str], List[int]]:
    """Programs, scales and the column order the heads emit (program-major, scale-minor).

    Targets are named ``prog_<p>_<scale>_z`` (learned family) and ``anchor_<p>_<scale>_z``
    (fixed-set family); the network predicts the learned family, the anchor family is a
    baseline target set trained separately. ``ctx.targets`` may list the learned targets in
    any order; the returned ``target_index`` maps each head output column to its column in
    ``ctx.Y`` so Training can permute once and Evaluation can map back.
    """
    learned = [t for t in targets if (target_family or {}).get(t, "learned") == "learned" and t.startswith("prog_")]
    programs: List[str] = []
    scales: List[str] = []
    for t in learned:
        body = t[len("prog_"):-len("_z")]
        p, _, sc = body.rpartition("_")
        if p not in programs:
            programs.append(p)
        if sc not in scales:
            scales.append(sc)
    ordered = [f"prog_{p}_{s}_z" for p in programs for s in scales]
    missing = [t for t in ordered if t not in targets]
    if missing:
        raise ValueError(f"learned targets are not a full program x scale grid; missing {missing}")
    target_index = [targets.index(t) for t in ordered]
    return programs, scales, target_index


def build_model(spec: Dict, device: str):
    from ..models.unified import UnifiedProgramModel
    model = UnifiedProgramModel(spec).to(device)
    return model


class UnifiedModel(BaseComponent):
    name = "unified_model"
    requires: Tuple[str, ...] = ("EMB", "GRID", "MOR", "targets", "program_weights", "lin_names", "n_samples")
    provides: Tuple[str, ...] = ("model_spec",)
    artefacts: Tuple[str, ...] = ("model_spec.json",)
    depends_on: Tuple[str, ...] = ("feature_extraction", "program_inference", "feature_selection")
    config_fields: Tuple[str, ...] = ("d_model", "n_layers", "n_heads", "head_dim", "dropout", "n_fourier", "fourier_scale",
                                      "slide_emb_dim", "uncertainty", "temperature_scaling", "lambda_comp", "lambda_smooth",
                                      "lambda_align", "knn_k", "grid", "hidden")

    def _run(self, ctx: Context) -> None:
        cfg = ctx.cfg
        programs, scales, target_index = target_layout(ctx.targets, ctx.target_family)
        pw = ctx.program_weights                                   # DataFrame: programs x genes or genes x programs
        if all(p in pw.index for p in programs):
            init = pw.loc[programs].to_numpy(dtype=np.float32)    # (P, G)
        elif all(p in pw.columns for p in programs):
            init = pw[programs].to_numpy(dtype=np.float32).T      # (P, G)
        else:
            raise ValueError(f"program_weights has neither rows nor columns for {programs}")
        grid_tokens = int(ctx.GRID.shape[1])
        g = int(round(grid_tokens ** 0.5))
        if g * g != grid_tokens:
            raise ValueError(f"GRID has {grid_tokens} tokens, not a square grid")
        if g != cfg.grid:
            warn(f"model grid set to {g} (encoder feature map) instead of configured {cfg.grid}")
        spec = dict(
            d_model=int(cfg.d_model), n_layers=int(cfg.n_layers), n_heads=int(cfg.n_heads), head_dim=int(cfg.head_dim),
            dropout=float(cfg.dropout), n_fourier=int(cfg.n_fourier), fourier_scale=float(cfg.fourier_scale),
            slide_emb_dim=int(cfg.slide_emb_dim), grid=g, k=int(cfg.knn_k), n_programs=len(programs), n_scales=len(scales),
            programs=programs, scales=scales, targets=[f"prog_{p}_{s}_z" for p in programs for s in scales],
            target_index=target_index,
            n_lineages=len(ctx.lin_names), lineages=list(ctx.lin_names), n_samples=int(ctx.n_samples),
            d_centre=int(ctx.EMB.shape[1]), d_grid=int(ctx.GRID.shape[2]), n_genes=int(init.shape[1]),
            uncertainty=str(cfg.uncertainty), temperature_scaling=bool(cfg.temperature_scaling),
            d_align=len(programs), mlp_hidden=int(cfg.hidden), program_init_weights=init.tolist(),
            lambda_comp=float(cfg.lambda_comp), lambda_smooth=float(cfg.lambda_smooth), lambda_align=float(cfg.lambda_align),
            encoder=ctx.encoder_card.get("name") if ctx.encoder_card else cfg.encoder,
        )
        model = build_model(spec, "cpu")
        model.describe()
        spec["parameter_counts"] = model.block_parameter_counts()
        spec["attention_mask"] = ("morphology tokens attend to all tokens; each neighbour token attends to itself and the "
                                  "centre token; program tokens cross-attend to all morphology and neighbour tokens")
        say(f"targets predicted by the network ({len(spec['targets'])}): {spec['targets']}")
        say(f"head column -> ctx.Y column: {target_index}")
        say(f"d_centre={spec['d_centre']} d_grid={spec['d_grid']} grid={g} k={spec['k']} lineages={spec['n_lineages']} "
            f"slides={spec['n_samples']} genes={spec['n_genes']}")
        with open(self.path(ctx, "model_spec.json"), "w") as fh:
            json.dump(spec, fh, indent=2)
        ctx.set("model_spec", spec)

    def load(self, ctx: Context) -> None:
        with open(self.path(ctx, "model_spec.json")) as fh:
            spec = json.load(fh)
        ctx.set("model_spec", spec)
        say(f"loaded model spec: {spec['parameter_counts']['total']:,} parameters, targets {len(spec['targets'])}")


def build_unified_model(ctx: Context) -> Context:
    """Component 6: the token network (centre, grid blocks, neighbours, program tokens) and its heads, written as a specification.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return UnifiedModel().run_in(ctx)
