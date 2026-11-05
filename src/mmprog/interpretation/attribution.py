"""Attribution for the unified model: cross-attention mass and integrated gradients.

Both quantities are descriptive. Attention mass says where a program token looked
(centre block, inner or outer grid ring, neighbour cells); it is not a causal
measure of what changed the prediction. Integrated gradients (Sundararajan et al.,
ICML 2017; venue not verified in docs/research) attribute the prediction of one target to the input features along the
straight path from a baseline, and satisfy completeness (attributions sum to the
difference between the prediction at the input and at the baseline), which is the
property the test checks.

Model interface assumed here (shared with ``inference/virtual_map_v3.py``): the
model is called with one dict ``batch`` of tensors and returns a dict with at least
``mean`` (B, T) and, when it has program tokens, ``attn`` (B, P, n_tokens). Keys of
``batch``: ``centre`` (B, D_enc), ``grid`` (B, g*g, D_grid), ``nb`` (B, k, D_enc),
``nb_offset`` (B, k, 2) in microns, ``nb_mask`` (B, k) bool, ``sid`` (B,), ``morpho``
(B, M) and optionally ``features`` (B, F) when a feature space is attached.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import say, timed


def call_model(model: Any, batch: Dict[str, Any], need_attn: bool = True) -> Dict[str, Any]:
    """Call the model on one batch dict.

    ``UnifiedProgramModel.forward(centre_emb, grid_tokens, nb_emb, nb_offsets_um, sid, variant_flags, need_attn)``
    takes positional tensors, so the dict keys are mapped onto that signature; a model that
    accepts the dict directly (test doubles) is called with it unchanged.
    """
    if getattr(model, "accepts_batch_dict", False):
        return model(batch)
    try:
        return model(batch["centre"], batch["grid"], batch.get("nb"), batch.get("nb_offset"), batch["sid"],
                     variant_flags=batch.get("flags"), need_attn=need_attn)
    except TypeError:
        return model(batch)


def _to_tensor(value: Any, device: str):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return torch.as_tensor(arr, device=device)
    if np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def batch_slice(arrays: Dict[str, Any], idx: np.ndarray, device: str) -> Dict[str, Any]:
    """Slice every array of ``arrays`` by ``idx`` and move to tensors on ``device``."""
    out = {}
    for k, v in arrays.items():
        if v is None:
            continue
        if k == "flags":
            out[k] = v
            continue
        out[k] = _to_tensor(np.asarray(v)[idx], device)
    return out


def default_token_groups(grid: int, block_um: float, k: int, has_centre: bool = True) -> Dict[str, np.ndarray]:
    """Token index groups for a token order [centre, g*g grid blocks, k neighbours].

    Grid blocks are grouped by their block-centre distance from the nucleus (same bands as
    ``evaluation.spatial_boost_eval.ring_summary``): centre < 5 um, inner 5 to 10 um, outer >= 10 um.
    """
    offset = 1 if has_centre else 0
    c = (grid - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    radius = np.sqrt((ii - c) ** 2 + (jj - c) ** 2).ravel() * block_um
    groups: Dict[str, np.ndarray] = {}
    if has_centre:
        groups["centre_token"] = np.array([0])
    groups["grid_centre"] = offset + np.where(radius < 5.0)[0]
    groups["grid_inner_ring"] = offset + np.where((radius >= 5.0) & (radius < 10.0))[0]
    groups["grid_outer_ring"] = offset + np.where(radius >= 10.0)[0]
    if k > 0:
        groups["neighbour_cells"] = offset + grid * grid + np.arange(k)
    return {g: idx for g, idx in groups.items() if len(idx) > 0}


def attention_attribution(model: Any, arrays: Dict[str, Any], targets: Sequence[str], program_names: Sequence[str],
                          token_groups: Dict[str, np.ndarray], device: str = "cpu", max_cells: int = 20000,
                          batch_size: int = 512, lineage_names: Optional[np.ndarray] = None,
                          seed: int = 0) -> pd.DataFrame:
    """Mean cross-attention mass of each program token on each token group.

    Returns one row per (program, group) over all cells and, when ``lineage_names`` is
    given, one row per (program, group, lineage). Masses within a program sum to one when
    the groups partition the tokens. Attention mass is descriptive, not causal.
    """
    import torch

    n = len(next(v for k, v in arrays.items() if k != "flags" and v is not None))
    rng = np.random.default_rng(seed)
    idx_all = np.sort(rng.choice(n, min(n, max_cells), replace=False))
    say(f"attention attribution: {len(idx_all)} cells, {len(program_names)} program tokens, groups {list(token_groups)}")
    sums: Dict[Any, np.ndarray] = {}      # key (lineage or "all") -> (P, n_groups) summed mass
    counts: Dict[Any, int] = {}
    model.eval()
    with torch.no_grad(), timed("attention attribution"):
        for b in range(0, len(idx_all), batch_size):
            idx = idx_all[b:b + batch_size]
            out = call_model(model, batch_slice(arrays, idx, device))
            attn = out.get("attn")
            if attn is None:
                say("model returned no attention (no program tokens); attribution skipped")
                return pd.DataFrame(columns=["program", "group", "lineage", "mass"])
            attn = attn.detach().float().cpu().numpy()          # (B, P, n_tokens)
            group_mass = np.stack([attn[:, :, g].sum(-1) for g in token_groups.values()], axis=-1)  # (B, P, G)
            keys = ["all"] * len(idx) if lineage_names is None else list(lineage_names[idx])
            for key in set(keys) | {"all"}:
                m = np.ones(len(idx), dtype=bool) if key == "all" else (np.asarray(keys) == key)
                sums[key] = sums.get(key, 0) + group_mass[m].sum(0)
                counts[key] = counts.get(key, 0) + int(m.sum())
    rows = []
    for key, s in sums.items():
        mean = s / max(counts[key], 1)
        for p, pname in enumerate(program_names):
            for g, gname in enumerate(token_groups):
                rows.append(dict(program=pname, group=gname, lineage=key, mass=float(mean[p, g]), n_cells=counts[key]))
    df = pd.DataFrame(rows)
    overall = df[df.lineage == "all"].pivot(index="program", columns="group", values="mass")
    say("attention mass per program token (all cells):")
    for line in overall.round(3).to_string().splitlines():
        say(line, indent=1)
    return df


def integrated_gradients(model: Any, inputs: Dict[str, Any], target_idx: int, attr_keys: Sequence[str] = ("centre",),
                         baseline: Optional[Dict[str, Any]] = None, steps: int = 32, device: str = "cpu",
                         output_key: str = "mean") -> Dict[str, np.ndarray]:
    """Integrated gradients of ``out[output_key][:, target_idx]`` with respect to ``inputs[key]`` for each key.

    The baseline defaults to zeros for the attributed keys and to the input itself for
    all other keys (they are held fixed). Returns attributions with the shape of each
    attributed input. Completeness holds up to the Riemann approximation error of ``steps``.
    """
    import torch

    model.eval()
    base_batch = {k: (v if k == "flags" else _to_tensor(v, device)) for k, v in inputs.items() if v is not None}
    base_vals = {}
    for k in attr_keys:
        base_vals[k] = (_to_tensor(baseline[k], device) if baseline and k in baseline
                        else torch.zeros_like(base_batch[k]))
    totals = {k: torch.zeros_like(base_batch[k]) for k in attr_keys}
    for s in range(1, steps + 1):
        alpha = s / steps
        batch = dict(base_batch)
        for k in attr_keys:
            batch[k] = (base_vals[k] + alpha * (base_batch[k] - base_vals[k])).detach().requires_grad_(True)
        out = call_model(model, batch)
        score = out[output_key][:, target_idx].sum()
        grads = torch.autograd.grad(score, [batch[k] for k in attr_keys], allow_unused=True)
        for k, g in zip(attr_keys, grads):
            if g is not None:
                totals[k] = totals[k] + g
    attributions = {k: ((base_batch[k] - base_vals[k]) * totals[k] / steps).detach().cpu().numpy() for k in attr_keys}
    with torch.no_grad():
        f_x = call_model(model, base_batch)[output_key][:, target_idx]
        zero_batch = dict(base_batch)
        for k in attr_keys:
            zero_batch[k] = base_vals[k]
        f_0 = call_model(model, zero_batch)[output_key][:, target_idx]
    total_attr = sum(a.reshape(len(a), -1).sum(1) for a in attributions.values())
    gap = float(np.abs(total_attr - (f_x - f_0).cpu().numpy()).mean())
    say(f"integrated gradients for target {target_idx}: keys {list(attr_keys)}, {steps} steps, "
        f"mean completeness gap {gap:.4f}")
    return attributions


def summarise_attributions(attr: Dict[str, np.ndarray], names: Optional[Dict[str, List[str]]] = None,
                           top: int = 10) -> pd.DataFrame:
    """Mean absolute attribution per input dimension, top entries per key."""
    rows = []
    for key, a in attr.items():
        flat = a.reshape(len(a), -1)
        mean_abs = np.abs(flat).mean(0)
        order = np.argsort(mean_abs)[::-1][:top]
        for rank, j in enumerate(order):
            label = names[key][j] if names and key in names and j < len(names[key]) else str(j)
            rows.append(dict(input=key, dim=int(j), name=label, mean_abs_attribution=float(mean_abs[j]), rank=rank + 1))
    return pd.DataFrame(rows)
