"""Gene programs, lineage markers and stage vocabulary, loaded from ``resources/``."""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

from .config import RESOURCE_DIR
from .logging_utils import say


def load_programs(include_signature: bool = True) -> Dict[str, List[str]]:
    with open(os.path.join(RESOURCE_DIR, "gene_programs.json")) as fh:
        meta = json.load(fh)
    programs = dict(meta["programs"])
    if include_signature and meta.get("signature_15"):
        programs["signature_15"] = list(meta["signature_15"])
    say(f"loaded {len(programs)} gene programs: " + ", ".join(f"{k} ({len(v)} genes)" for k, v in programs.items()))
    return programs


def load_program_notes() -> Dict[str, str]:
    with open(os.path.join(RESOURCE_DIR, "gene_programs.json")) as fh:
        return json.load(fh).get("_drug_resistance_link", {})


def load_program_labels() -> Dict[str, str]:
    with open(os.path.join(RESOURCE_DIR, "gene_programs.json")) as fh:
        return json.load(fh).get("_labels", {})


def load_gene_annotations() -> Dict[str, Dict[str, dict]]:
    """Per-gene class, pathway role, myeloma evidence and evidence grade (see docs/research/PATHWAY_INTERPRETATION.md)."""
    path = os.path.join(RESOURCE_DIR, "gene_annotations.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh).get("programs", {})


def load_annotation_caveats() -> List[str]:
    """The interpretation caveats recorded with the gene annotations, so reports read them instead of retyping them."""
    path = os.path.join(RESOURCE_DIR, "gene_annotations.json")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [str(c) for c in json.load(fh).get("_caveats", [])]


def print_program_annotations(programs: Dict[str, List[str]]) -> None:
    """Print every program gene with its annotation and evidence grade, so a run log states what is being predicted."""
    ann = load_gene_annotations()
    labels = load_program_labels()
    flat = {g: a for prog in ann.values() for g, a in prog.items()}
    for name, genes in programs.items():
        say(f"program {name} ({len(genes)} genes): {labels.get(name, '')}")
        grades = {}
        for g in genes:
            a = flat.get(g)
            if a is None:
                say(f"{g:<10} no annotation on file", indent=2)
                continue
            grades[a.get("evidence_grade", "unknown")] = grades.get(a.get("evidence_grade", "unknown"), 0) + 1
            say(f"{g:<10} {a.get('class', ''):<28} grade={a.get('evidence_grade', '?'):<13} {a.get('pathway_role', '')[:110]}", indent=2)
        if grades:
            say("evidence grades: " + ", ".join(f"{k}={v}" for k, v in sorted(grades.items())), indent=2)


def load_lineages() -> Dict[str, List[str]]:
    with open(os.path.join(RESOURCE_DIR, "lineage_markers.json")) as fh:
        lin = json.load(fh)["lineages"]
    say(f"loaded {len(lin)} lineage marker sets: " + ", ".join(lin))
    return lin


def load_stage_vocabulary() -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    with open(os.path.join(RESOURCE_DIR, "stage_vocabulary.json")) as fh:
        v = json.load(fh)
    return [tuple(x) for x in v["stage_from_title"]], v["stage_order"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
