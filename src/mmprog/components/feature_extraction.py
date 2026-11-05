"""FeatureExtraction component: encoder embeddings, grid tokens, morphometrics, neighbour graph.

Four outputs, all indexed by ``ctx.obs`` rows: EMB (pooled encoder embedding), GRID
(g x g block tokens), MOR (hand-made morphometrics, D13), and the neighbour graph
(NB, NB_DIST in microns, slide-normalised XY, slide ids SID, patient ids GROUPS).
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..core.component import BaseComponent
from ..core.context import Context
from ..features.embed import check_embeddings
from ..features.embed import extract_features as embed_cells
from ..features.encoders import get_encoder
from ..features.morphometrics import MORPHO_NAMES, morphometrics
from ..logging_utils import describe_array, say, timed
from ..models.neighbours import knn_table, norm_xy


class FeatureExtraction(BaseComponent):
    name = "feature_extraction"
    requires: Tuple[str, ...] = ("obs", "images", "adata")
    provides: Tuple[str, ...] = ("EMB", "GRID", "MOR", "morpho_names", "NB", "NB_DIST", "XY", "SID", "GROUPS",
                                 "n_samples", "encoder_card")
    artefacts: Tuple[str, ...] = ("emb.f16.npy", "grid.f16.npy", "morpho.npy", "graph.npz", "encoder_card.json")
    depends_on: Tuple[str, ...] = ("preprocess",)
    config_fields: Tuple[str, ...] = ("encoder", "encoder_token_env", "commercial_use", "patch_px", "grid", "memmap_dir", "emb_batch",
                                      "emb_dtype", "density_radii_px", "knn_k", "num_workers", "pin_memory",
                                      "persistent_workers", "prefetch_factor", "channels_last", "compile_encoder",
                                      "amp_enabled")

    # ------------------------------------------------------------------
    def _graph(self, ctx: Context) -> None:
        obs = ctx.obs
        nb = knn_table(obs, ctx.cfg.knn_k)
        xy_px = obs[["x", "y"]].to_numpy(dtype=np.float32)
        d_px = np.linalg.norm(xy_px[nb] - xy_px[:, None, :], axis=-1)
        um_cell = ctx.um_per_px_per_cell()                       # each slide's own pixel size (OME metadata at ingest)
        nb_dist = (d_px * um_cell[:, None]).astype(np.float32)
        say(f"neighbour distances in microns use per-slide pixel sizes ({len(set(np.round(um_cell, 5)))} distinct values)")
        sid = pd.Categorical(obs["sample"]).codes.astype(np.int64)
        ctx.set("NB", nb)
        ctx.set("NB_DIST", nb_dist)
        ctx.set("XY", norm_xy(obs))
        ctx.set("SID", sid)
        ctx.set("GROUPS", obs["patient"].astype(str).to_numpy())
        ctx.set("n_samples", int(sid.max()) + 1)
        say(f"neighbour graph: k={ctx.cfg.knn_k}, median neighbour distance "
            f"{np.median(nb_dist):.1f} um (max {nb_dist.max():.1f} um), {ctx.n_samples} slides, "
            f"{len(np.unique(ctx.GROUPS))} patients")
        np.savez_compressed(self.path(ctx, "graph.npz"), NB=nb, NB_DIST=nb_dist, XY=ctx.XY, SID=sid,
                            GROUPS=ctx.GROUPS, n_samples=ctx.n_samples)

    def _run(self, ctx: Context) -> None:
        cfg = ctx.cfg
        say(f"inputs: obs {ctx.obs.shape}, {ctx.obs['sample'].nunique()} slides; encoder '{cfg.encoder}'")
        enc, card = get_encoder(cfg.encoder, ctx.device, cfg)
        with timed("EMB and GRID extraction"):
            emb, grid = embed_cells(ctx.obs, ctx.images, enc, card, cfg, ctx.device, self.dir(ctx))
        del enc
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        check_embeddings(emb, len(ctx.obs), "EMB")
        check_embeddings(grid, len(ctx.obs), "GRID")
        ctx.set("EMB", emb)
        ctx.set("GRID", grid)
        card["grid_tokens"] = int(grid.shape[1])
        card["grid_dim"] = int(grid.shape[2])
        with open(self.path(ctx, "encoder_card.json"), "w") as fh:
            json.dump(card, fh, indent=2, default=str)
        ctx.set("encoder_card", card)

        with timed("morphometrics"):
            mor = morphometrics(ctx.obs, ctx.images, cfg)
        np.save(self.path(ctx, "morpho.npy"), mor)
        ctx.set("MOR", mor)
        ctx.set("morpho_names", MORPHO_NAMES(cfg))
        describe_array("MOR", mor)
        self._graph(ctx)
        from ..observations import observe
        observe(cfg.run_dir, "feature_extraction",
                before=f"crops and {len(ctx.obs)} cell coordinates",
                after=f"EMB {tuple(emb.shape)}, GRID {tuple(grid.shape)}, MOR {tuple(mor.shape)}, "
                      f"neighbours k={ctx.NB.shape[1]}",
                note=f"encoder {card.get('name')}; neighbour distances are in microns",
                examples=[f"slide {s}: {int((ctx.obs['sample'] == s).sum())} cells"
                          for s in list(ctx.obs['sample'].unique())[:5]])

    def load(self, ctx: Context) -> None:
        emb = np.load(self.path(ctx, "emb.f16.npy"), mmap_mode="r")
        grid = np.load(self.path(ctx, "grid.f16.npy"), mmap_mode="r")
        check_embeddings(emb, len(ctx.obs), "EMB")
        check_embeddings(grid, len(ctx.obs), "GRID")
        ctx.set("EMB", emb)
        ctx.set("GRID", grid)
        ctx.set("MOR", np.load(self.path(ctx, "morpho.npy")))
        ctx.set("morpho_names", MORPHO_NAMES(ctx.cfg))
        g = np.load(self.path(ctx, "graph.npz"), allow_pickle=True)
        ctx.set("NB", g["NB"])
        ctx.set("NB_DIST", g["NB_DIST"])
        ctx.set("XY", g["XY"])
        ctx.set("SID", g["SID"])
        ctx.set("GROUPS", g["GROUPS"].astype(str))
        ctx.set("n_samples", int(g["n_samples"]))
        with open(self.path(ctx, "encoder_card.json")) as fh:
            ctx.set("encoder_card", json.load(fh))
        say(f"loaded EMB {emb.shape}, GRID {grid.shape}, MOR {ctx.MOR.shape}, graph k={ctx.NB.shape[1]}, "
            f"encoder {ctx.encoder_card['name']} (weights {ctx.encoder_card.get('weight_hash')})")
        if not os.path.exists(self.path(ctx, "graph.npz")):
            raise FileNotFoundError("graph.npz missing")


def extract_features(ctx: Context) -> Context:
    """Component 4: frozen encoder embeddings, patch-grid tokens, morphometrics, neighbour graph.

    The one line ``main.py`` calls; the class above carries the skip rule, the manifest and
    the artefact contract. Runs under ``ctx.plan`` (``--only`` / ``--force``).
    """
    return FeatureExtraction().run_in(ctx)
