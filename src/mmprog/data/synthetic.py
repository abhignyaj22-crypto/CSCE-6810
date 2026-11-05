"""Synthetic fixture: six small "trephines" with planted spatial structure.

The fixture exists so that every stage of the pipeline can be exercised on a
laptop in minutes, with no download and no GPU. It is NOT a simulation of
myeloma biology; it only guarantees that there is a learnable signal between
morphology and the program targets:

* each slide has four circular "lesions"; the background inside them is darker;
* plasma cells are enriched inside lesions and carry extra ``wnt_camdr``
  transcripts there; T cells inside lesions carry extra ``t_exhaustion``;
* nuclei are drawn as discs whose radius and darkness depend on lineage, so
  the image carries cell-type information the way real morphology does.

Disease stage controls the plasma-cell fraction (control 3%, SMM 20%, MM 60%).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..config import Config
from ..logging_utils import say
from ..preprocessing.spatial_summaries import attach_summaries, count_summaries

# lineage -> (nuclear radius px, intensity change on a light background)
_STYLE = {"plasma": (7, -90), "T": (3, -70), "NK": (3, -70), "B": (3, -60), "granulocyte": (4, -40),
          "monocyte": (5, -45), "erythroid": (3, -20), "stromal": (6, -15), "megakaryocyte": (9, -50)}
_PC_FRACTION = {"control": 0.03, "MGUS": 0.08, "SMM": 0.20, "MM": 0.60}


def _draw_slide(rng: np.random.Generator, xy: np.ndarray, lineage: np.ndarray,
                centers: np.ndarray, blob_r: float, img_size: int) -> np.ndarray:
    img = rng.normal(150, 12, size=(img_size, img_size)).astype(np.float32)
    yy, xx = np.mgrid[0:img_size, 0:img_size]
    for c in centers:
        inside = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < blob_r ** 2
        img[inside] += rng.normal(-25, 15, size=inside.sum())
    for (cx, cy), lin in zip(xy, lineage):
        r, dv = _STYLE[lin]
        cx, cy = int(cx), int(cy)
        ys, xs = slice(max(cy - r, 0), cy + r + 1), slice(max(cx - r, 0), cx + r + 1)
        sub = img[ys, xs]
        gy, gx = np.mgrid[ys, xs]
        disc = (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
        sub[disc] += dv + rng.normal(0, 8, size=disc.sum())
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo(programs: Dict[str, List[str]], lineages: Dict[str, List[str]], cfg: Config,
              n_patients: int = 6, cells_per: int = 3000, img_size: int = 1024,
              seed: int = 7) -> Tuple[ad.AnnData, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Build the synthetic cohort.

    Returns ``(adata, images, manifest_rows)``. ``adata.obs`` carries
    ``lineage_true`` and all count summaries (``cnt_*``, ``nb_*``, ``density_*``).
    """
    rng = np.random.default_rng(seed)
    stages = (["control", "control", "SMM", "SMM", "MM", "MM"] * 3)[:n_patients]
    genes = sorted(set(g for p in programs.values() for g in p) | set(g for p in lineages.values() for g in p))
    gene_index = {g: k for k, g in enumerate(genes)}
    other_lineages = [name for name in lineages if name != "plasma" and name in _STYLE]
    blob_r = img_size * 120 / 1024

    adatas, images, manifest = [], {}, []
    say(f"synthetic fixture: {n_patients} patients x {cells_per} cells, {len(genes)} genes, {img_size}px slides")
    for i, stage in enumerate(stages):
        pid = f"DEMO_P{i}"
        pc_frac = _PC_FRACTION[stage]
        xy = rng.uniform(32, img_size - 32, size=(cells_per, 2))
        centers = rng.uniform(100, img_size - 100, size=(4, 2))
        dist = np.min(np.linalg.norm(xy[:, None, :] - centers[None], axis=-1), axis=1)
        in_blob = dist < blob_r
        is_pc = (rng.uniform(size=cells_per) < pc_frac) | (in_blob & (rng.uniform(size=cells_per) < pc_frac * 1.5))
        lineage = np.where(is_pc, "plasma", rng.choice(other_lineages, size=cells_per))

        X = rng.poisson(0.7, size=(cells_per, len(genes))).astype(np.float32)
        for lin, markers in lineages.items():
            m = lineage == lin
            for g in markers:
                X[m, gene_index[g]] += rng.poisson(3.0, size=m.sum())
        # Planted program signal: Wnt/CAM-DR in plasma cells inside lesions, exhaustion in T cells inside lesions.
        wnt_cells = is_pc & in_blob
        for g in programs.get("wnt_camdr", []):
            X[wnt_cells, gene_index[g]] += rng.poisson(4.0, size=wnt_cells.sum())
        exh_cells = (lineage == "T") & in_blob
        for g in programs.get("t_exhaustion", []):
            X[exh_cells, gene_index[g]] += rng.poisson(3.0, size=exh_cells.sum())

        obs = pd.DataFrame({"x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32),
                            "sample": pid, "patient": pid, "stage": stage, "lineage_true": lineage,
                            "cohort": "demo", "platform": "xenium"},
                           index=[f"{pid}_c{j}" for j in range(cells_per)])
        a = ad.AnnData(csr_matrix(X), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
        summaries = count_summaries(a, programs, cfg)
        a = attach_summaries(a, summaries, a)
        adatas.append(a)
        images[pid] = _draw_slide(rng, xy, lineage, centers, blob_r, img_size)
        manifest.append(dict(sample=pid, patient=pid, stage=stage, title=f"synthetic {stage}", cohort="demo",
                             platform="xenium", supp=[], h5=None, cells=None, image=None, he_image=None,
                             um_per_px_l0=cfg.um_per_px_level0))
        say(f"{pid} {stage}: {is_pc.sum()} plasma cells ({100 * is_pc.mean():.0f}%), "
            f"{wnt_cells.sum()} carry the planted wnt_camdr signal, {exh_cells.sum()} T cells exhausted", indent=1)

    adata = ad.concat(adatas)
    adata.obs_names_make_unique()
    say(f"synthetic cohort: {adata.n_obs} cells x {adata.n_vars} genes, "
        f"stages {adata.obs.groupby('sample')['stage'].first().to_dict()}")
    return adata, images, manifest
