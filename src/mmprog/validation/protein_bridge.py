"""Protein-imaging bridge (CODEX / PhenoCycler), data-gated.

The model is applied to the DAPI channel of a protein-imaging section exactly as to any
new slide; detected nuclei are matched to the protein cell table within 8 um, which gives
every nucleus a protein-defined lineage and a protein-defined plasma-cell-dense flag. Two
questions follow (``evaluation/codex_validation.py`` holds the statistics): composition
agreement between the model's composition head and the protein lineages, and the
contrast of predicted programs between plasma-dense niches and the rest.

No public dataset pairs CODEX with Xenium on the same myeloma trephine, so the bridge
runs only when ``codex_table`` and ``codex_dapi`` are set; the code path is exercised on
a synthetic fixture in the tests.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from ..logging_utils import say, timed


def protein_bridge(cfg, checkpoint: Dict[str, Any], device: str, targets: Sequence[str], lin_names: Sequence[str],
                   vmap_fn: Callable, out_dir: str, match_um: float = 8.0) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Return ``(result dict in the v2 codex_validation.json shape, niche-contrast table or None)``."""
    from ..data.codex import assign_lineage_from_markers, codex_to_obs, load_codex_table, load_marker_mapping, match_markers
    from ..data.images import load_image
    from ..evaluation.codex_validation import (composition_agreement, dapi_native_um_per_px, niche_contrast,
                                               resample_to_working_frame)

    mapping = load_marker_mapping()
    table = load_codex_table(cfg.codex_table, um_per_px_codex=cfg.codex_um_per_px)
    matched = match_markers(list(table.columns), mapping)
    lineage = assign_lineage_from_markers(table, mapping, matched)
    cobs = codex_to_obs(table, lineage, cfg, sample="CODEX", mapping=mapping, matched=matched)
    native = dapi_native_um_per_px(cfg.codex_dapi, cfg.codex_dapi_um_per_px)
    say(f"protein-imaging DAPI {cfg.codex_dapi}: native {native:.4f} um/px, resampled to working {cfg.um_per_px:.4f} um/px")
    img = resample_to_working_frame(load_image(cfg.codex_dapi, 0, None, cfg.z_project, cfg.stain_mode), native, cfg.um_per_px)
    with timed("morphology-only inference on the protein-imaging DAPI channel"):
        vmap = vmap_fn(img, checkpoint, cfg, device, "CODEX", None, um_per_px=float(cfg.um_per_px))   # resampled to the working frame
    if len(vmap) == 0:
        return dict(status="aborted", reason="no nuclei detected on the protein-imaging DAPI image"), None
    d, j = KDTree(cobs[["x", "y"]].to_numpy(dtype=np.float32)).query(vmap[["x", "y"]].to_numpy(dtype=np.float32), k=1)
    ok = d[:, 0] < match_um * cfg.px_per_um
    say(f"matched {int(ok.sum())}/{len(vmap)} detected nuclei to protein-imaging cells within {match_um} um")
    result: Dict[str, Any] = dict(status="ran", n_codex_cells=int(len(cobs)), n_detected=int(len(vmap)),
                                  n_matched=int(ok.sum()), match_um=match_um)
    contrast: Optional[pd.DataFrame] = None
    if ok.sum() < 10:
        result["status"] = "insufficient_matches"
        result["reason"] = f"only {int(ok.sum())} nuclei matched within {match_um} um"
    else:
        m = vmap.loc[ok].copy()
        m["codex_lineage"] = cobs["lineage_name"].to_numpy()[j[ok, 0]]
        m["plasma_dense"] = cobs["plasma_dense"].to_numpy()[j[ok, 0]]
        prob_cols: List[str] = [c for c in m.columns if c.startswith("lineage_prob_")]
        if prob_cols:
            m["pred_lineage"] = [c.replace("lineage_prob_", "") for c in np.array(prob_cols)[m[prob_cols].to_numpy().argmax(1)]]
            result["composition"] = composition_agreement(m["pred_lineage"].to_numpy(), m["codex_lineage"].to_numpy(), list(lin_names))
        else:
            say("model has no composition head: composition agreement skipped")
        pred_cols = [t.replace("_z", "_pred") for t in targets if t.replace("_z", "_pred") in m]
        contrast = niche_contrast(m, pred_cols)
        result["niche_contrast"] = contrast.to_dict("records")
        m.to_csv(os.path.join(out_dir, "protein_bridge_matched_nuclei.csv"), index=False)
    with open(os.path.join(out_dir, "protein_bridge.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    say(f"protein bridge: {result['status']} -> protein_bridge.json")
    return result, contrast
