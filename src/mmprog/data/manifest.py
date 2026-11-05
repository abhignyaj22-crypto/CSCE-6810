"""Sample manifest built from GEO metadata at run time.

Nothing about sample ids or file names is hard-coded. The GEO record is read
with GEOparse, human samples are kept, the disease stage is parsed from the
sample title with the vocabulary in ``resources/stage_vocabulary.json``, and
the supplementary-file URLs are stored on each row so that a wiped Colab disk
can be re-populated without touching GEO's metadata service again.

Manifest row schema (one row per slide)::

    sample, patient, stage, title, cohort, platform, supp (list of URLs),
    h5, cells, image, he_image, um_per_px_l0
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd

from ..config import Config
from ..logging_utils import say, show_table
from ..resources import load_stage_vocabulary

ManifestRow = Dict[str, Any]


def _stage_from_title(title: str, vocabulary: List[tuple]) -> str:
    low = title.lower()
    for key, stage in vocabulary:
        if key in low:
            return stage
    return "unknown"


def build_manifest_from_geo(cfg: Config) -> pd.DataFrame:
    """Query GEO for ``cfg.geo_accession`` and return one row per human sample."""
    import GEOparse  # optional dependency; only needed in real-data mode

    say(f"building manifest from GEO {cfg.geo_accession} (destdir {cfg.local_data})")
    gse = GEOparse.get_GEO(geo=cfg.geo_accession, destdir=cfg.local_data, silent=True)
    vocabulary, _ = load_stage_vocabulary()
    rows: List[ManifestRow] = []
    for gsm_id, gsm in sorted(gse.gsms.items()):
        title = gsm.metadata.get("title", [""])[0]
        organism = " ".join(gsm.metadata.get("organism_ch1", []))
        if "sapiens" not in organism:
            say(f"skipped non-human sample {gsm_id} ({organism})", indent=1)
            continue
        if cfg.exclude_title_regex and re.search(cfg.exclude_title_regex, title, re.I):
            say(f"excluded by title regex {cfg.exclude_title_regex!r}: {gsm_id} {title}", indent=1)
            continue
        supp = [v for k, vs in gsm.metadata.items() if k.startswith("supplementary_file")
                for v in vs if v.startswith(("ftp", "http"))]
        rows.append(dict(sample=gsm_id, patient=gsm_id, stage=_stage_from_title(title, vocabulary),
                         title=title, cohort="xenium_yip", platform="xenium", supp=supp,
                         h5=None, cells=None, image=None, he_image=None, um_per_px_l0=cfg.um_per_px_level0))
    mdf = pd.DataFrame(rows)
    say(f"manifest: {len(mdf)} human samples, stages: {mdf['stage'].value_counts().to_dict() if len(mdf) else {}}")
    if len(mdf):
        show_table(mdf[["sample", "stage", "title"]], max_rows=50)
    return mdf


def select_samples(mdf: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply ``cfg.sample_limit``: one sample per stage first, then fill in GEO order."""
    if not cfg.sample_limit or len(mdf) <= cfg.sample_limit:
        say(f"using all {len(mdf)} samples")
        return mdf.reset_index(drop=True)
    first_per_stage = mdf.groupby("stage").head(1)
    chosen = pd.concat([first_per_stage, mdf]).drop_duplicates("sample").head(cfg.sample_limit)
    say(f"sample_limit={cfg.sample_limit}: chose {chosen['sample'].tolist()} "
        f"(stages {chosen['stage'].tolist()})")
    return chosen.reset_index(drop=True)


def save_manifest(rows: List[ManifestRow], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    say(f"manifest with {len(rows)} rows written to {path}")


def load_manifest(path: str) -> List[ManifestRow]:
    with open(path) as fh:
        rows = json.load(fh)
    say(f"manifest with {len(rows)} rows loaded from {path}")
    return rows
