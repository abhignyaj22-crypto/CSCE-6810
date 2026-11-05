"""Run configuration.

One dataclass holds every knob of the pipeline. A YAML file sets the values, the
command line can override any of them, and the resolved configuration is written
to the run directory so a result can always be traced back to the exact settings
that produced it.

Design rule: nothing about sample ids, file names or gene lists lives here. Gene
lists come from ``resources/``, sample ids come from GEO at run time.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCE_DIR = os.path.join(REPO_ROOT, "resources")


@dataclass
class Config:
    run_name: str = "gse299193_full"
    root: str = "./workspace"
    local_data: str = "./workspace/raw"    # raw GEO downloads (tens of GB); safe to wipe
    retain_raw_images: bool = True         # False: delete each morphology file after its crop is cached
    seed: int = 7
    demo_mode: bool = False
    demo_n_patients: int = 6
    demo_cells_per: int = 3000
    demo_img_size: int = 1024
    phases: Tuple[int, ...] = (1,)
    geo_accession: str = "GSE299193"
    sample_limit: Optional[int] = None
    exclude_title_regex: str = "relapse"
    min_panel_genes: int = 1000
    platform: str = "xenium"
    um_per_px_level0: float = 0.2125
    img_level: int = 1
    z_project: str = "max"
    stain_mode: str = "native"
    min_transcripts: int = 60
    min_genes: int = 50
    max_cells_per_sample: Optional[int] = 30000
    nb_radius_um: float = 30.0
    target_spaces: Tuple[str, ...] = ("cell", "nb")
    target_norm: str = "cohort"
    reliability_seed: int = 11
    min_program_genes: int = 3
    encoder: str = "resnet50"
    patch_px: int = 64
    emb_batch: int = 512
    emb_dtype: str = "float16"
    pca_dim: int = 128
    use_morpho: bool = True
    density_radii_px: Tuple[int, ...] = (25, 50, 100)
    selection_enabled: bool = True
    selection_variance_floor: float = 1e-06
    selection_corr_threshold: float = 0.98
    selection_report_top: int = 20
    hidden: int = 256
    dropout: float = 0.3
    knn_k: int = 6
    lambda_smooth: float = 0.1
    lambda_comp: float = 0.5
    n_fourier: int = 32
    fourier_scale: float = 6.0
    slide_emb_dim: int = 16
    batch_cells: int = 2048
    max_epochs: int = 40
    patience: int = 6
    lr: float = 0.001
    weight_decay: float = 0.001
    n_folds: int = 7
    seeds: Tuple[int, ...] = (7, 8, 9)
    cv_variants: Tuple[str, ...] = ("full", "no_spatial", "no_comp", "no_morpho", "shuffled", "oracle")
    n_boot: int = 200
    min_cells_per_patient_metric: int = 30
    srb_enabled: bool = True
    srb_grid: int = 8
    srb_lambdas: Tuple[float, ...] = (0.0, 0.1, 0.5)
    srb_radius_um: float = 5.0
    srb_rounds: int = 120
    srb_max_train_cells: int = 40000
    motifs_enabled: bool = True
    # ---- clinical / optional phases ----
    clinical_csv: Optional[str] = None
    commpass_csv: Optional[str] = None
    slide_dir: Optional[str] = None
    min_plasma_for_burden: int = 50
    # CODEX / PhenoCycler bridge (data-gated): a cell-by-marker table (CSV or parquet with x, y and
    # marker intensity columns) and the matching DAPI image of the same section. Both must be set.
    codex_table: Optional[str] = None
    codex_dapi: Optional[str] = None
    codex_um_per_px: Optional[float] = None    # scale of the CODEX table's x, y if they are in pixels; None = microns
    codex_dapi_um_per_px: Optional[float] = None   # native pixel size of the CODEX DAPI image; None = read from the TIFF
                                                   # resolution tags or OME metadata, and raise if neither is present

    # ---- v3: program inference (anchored NMF, Spectra-style seeded factorisation) ----
    programs_n_extra: int = 2                  # unseeded programs learned alongside the seeded ones
    programs_anchor_prior: float = 1.0         # prior weight on anchor genes (pulls seeded programs toward their sets)
    programs_l1_nonanchor: float = 0.05        # L1 shrinkage on non-anchor gene weights
    programs_max_iter: int = 200
    programs_fit_cells: int = 100000           # cells sampled for the factorisation fit
    include_anchor_only_targets: bool = True   # keep the v2 fixed-set targets as the 'anchor_only' target family

    # ---- v3: features / encoder registry ----
    encoder_token_env: str = "HF_TOKEN"
    commercial_use: bool = False               # refuse non-commercial-licensed encoders when true
    grid: int = 8                              # grid x grid block tokens per patch
    memmap_dir: str = ""                       # empty: EMB and GRID sit in the component dir
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    channels_last: bool = True
    compile_encoder: bool = False

    # ---- v3: unified model ----
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    head_dim: int = 64
    uncertainty: str = "gaussian"              # heteroscedastic Gaussian head; "none" for plain MSE
    nll_beta: float = 0.5                      # beta-NLL weighting (0 = plain NLL); see models/heads.py
    temperature_scaling: bool = True
    lambda_align: float = 0.0                  # TANGLE-style contrastive alignment; off by default (D31), 'align' variant uses lambda_align_on
    lambda_align_on: float = 0.1
    model_variants: Tuple[str, ...] = ("full", "no_neighbours", "no_grid", "no_program_tokens", "no_comp", "no_smooth", "align", "shuffled", "oracle", "anchor_only_targets")

    # ---- v3: training on one GPU ----
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    tf32: bool = True
    compile_model: bool = False
    compile_mode: str = "default"
    cuda_graphs: bool = False
    grad_clip: float = 1.0
    warmup_steps: int = 100
    coverage_levels: Tuple[float, ...] = (0.5, 0.9)
    conformal_alpha: float = 0.1
    stage_test: str = "permanova"
    stage_n_perm: int = 999
    manova_n_perm: int = 999
    validation_n_random_sets: int = 200
    validation_perturbations: Tuple[str, ...] = ("gamma_0.8", "gamma_1.25", "blur_1.0", "rot90", "noise_0.02", "scale_0.9")
    validation_cohort_manifest: Optional[str] = None
    attribution_n_cells: int = 20000
    ig_steps: int = 32
    backbone_threshold: float = 0.05
    backbone_min_fold_fraction: float = 0.8
    device: str = "auto"

    @property
    def um_per_px(self) -> float:
        return self.um_per_px_level0 * (2 ** self.img_level)

    @property
    def px_per_um(self) -> float:
        return 1.0 / self.um_per_px

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root, "runs", self.run_name)

    @property
    def crop_dir(self) -> str:
        return os.path.join(self.root, "crops")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        return config_from_json(path)

    def validate(self) -> None:
        assert self.target_norm in ("cohort", "slide"), "target_norm must be cohort or slide"
        assert self.z_project in ("max", "mid"), "z_project must be max or mid"
        assert self.stain_mode in ("native", "hematoxylin")
        assert all(s in ("cell", "nb") for s in self.target_spaces)
        assert set(self.cv_variants) <= {"no_morpho", "full", "no_spatial", "no_comp", "oracle", "shuffled"}, (
            f"unknown cv variant in {self.cv_variants}")
        assert self.pca_dim > 0 and self.patch_px >= 16
        assert self.n_folds >= 2

    def describe(self) -> str:
        lines = ["Resolved configuration:"]
        for f in dataclasses.fields(self):
            lines.append(f"  {f.name:<32} = {getattr(self, f.name)!r}")
        lines.append(f"  {'um_per_px (working level)':<32} = {self.um_per_px:.4f}")
        lines.append(f"  {'run_dir':<32} = {self.run_dir}")
        return "\n".join(lines)


_TUPLE_FIELDS = frozenset({
    "coverage_levels", "seeds", "target_spaces", "density_radii_px", "cv_variants", "phases",
    "validation_perturbations", "model_variants", "srb_lambdas",
})
_PATH_FIELDS = frozenset({
    "root", "clinical_csv", "codex_dapi", "validation_cohort_manifest", "codex_table",
    "commpass_csv", "slide_dir", "local_data",
})
_STR_FIELDS = {f.name for f in dataclasses.fields(Config) if f.type in ("str", "Optional[str]")}
_INT_FIELDS = {f.name for f in dataclasses.fields(Config) if f.type in ("int", "Optional[int]")}
_FLOAT_FIELDS = {f.name for f in dataclasses.fields(Config) if f.type == "float"}
_BOOL_FIELDS = {f.name for f in dataclasses.fields(Config) if f.type == "bool"}
_NULL_WORDS = {"", "null", "none", "~"}


def _coerce(name: str, value: Any) -> Any:
    """Normalise a value already parsed from YAML/JSON into the field's type."""
    if name in _TUPLE_FIELDS and value is not None:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)
    if name in _PATH_FIELDS and isinstance(value, str):
        return os.path.expanduser(value)
    return value


def _parse_override(name: str, raw: str) -> Any:
    """Parse one ``key=value`` override by the dataclass annotation, not by YAML guessing.

    YAML would turn ``run_name=001`` into an int and ``run_name=2025-01-01`` into a date;
    a string field must keep the text exactly as typed (surrounding quotes stripped).
    """
    text = raw.strip()
    if name in _STR_FIELDS:
        if text.lower() in _NULL_WORDS and name not in {"run_name", "root", "local_data", "geo_accession"}:
            return None
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]
        return text
    if name in _BOOL_FIELDS:
        low = text.lower()
        if low in {"true", "yes", "1", "on"}:
            return True
        if low in {"false", "no", "0", "off"}:
            return False
        raise ValueError(f"{name} expects a boolean, got {raw!r}")
    if name in _INT_FIELDS:
        return None if text.lower() in _NULL_WORDS else int(text)
    if name in _FLOAT_FIELDS:
        return float(text)
    if name in _TUPLE_FIELDS:
        # accept YAML lists ([a, b]) and bare comma-separated lists (a,b)
        parsed = yaml.safe_load(text) if text.startswith("[") else [t.strip() for t in text.split(",") if t.strip()]
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
        # numeric tuples (seeds, phases, density radii) hold ints; the others hold names
        if name in {"seeds", "phases", "density_radii_px"}:
            return [int(v) for v in parsed]
        if name in ("srb_lambdas", "coverage_levels"):
            return [float(v) for v in parsed]
        return [str(v) for v in parsed]
    return yaml.safe_load(text)


CONFIG_SECTIONS = (
    "run", "ingest", "preprocess", "programs", "features", "selection", "model", "training",
    "evaluation", "interpretation", "validation", "deployment", "reporting",
)


def flatten_sections(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in CONFIG_SECTIONS and isinstance(v, dict):
            for kk, vv in v.items():
                out[kk] = vv
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Sequence[str]] = None) -> Config:
    values: Dict[str, Any] = {}
    if path:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        values.update(flatten_sections(raw))
        print(f"[config] loaded {path} ({len(values)} keys)")
    known = {f.name for f in dataclasses.fields(Config)}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.split(".")[-1]
        k = k.strip()
        if k not in known:
            raise ValueError(f"unknown config key in override: {k!r}")
        values[k] = _parse_override(k, v)
        print(f"[config] override {k} = {values[k]!r} ({type(values[k]).__name__})")
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**{k: _coerce(k, v) for k, v in values.items()})
    cfg.validate()
    return cfg


def config_from_json(path: str) -> Config:
    with open(path) as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(Config)}
    dropped = sorted(set(raw) - known)
    if dropped:
        print(f"[config] {path}: dropping unknown keys {dropped}")
    values = {k: _coerce(k, v) for k, v in raw.items() if k in known}
    values["device"] = "auto"
    cfg = Config(**values)
    cfg.validate()
    print(f"[config] restored from {path}")
    return cfg


def resolve_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    cfg.device = dev
    return dev


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)
    print(f"[config] written to {path}")
