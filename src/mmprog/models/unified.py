"""The unified program model: one network over morphology, neighbourhood and program tokens.

Tokens
    morphology   1 centre token + g*g block tokens (MorphologyTokens)
    neighbours   k tokens, one per nearest cell (NeighbourTokens)
    programs     P learned queries initialised from gene weights (ProgramTokens)

Blocks
    n_layers of self-attention over [morphology | neighbours] with a boolean mask: every
    morphology token attends to all morphology tokens and all neighbours; a neighbour
    token attends to itself and to the centre token only, so neighbours do not mix with
    each other through the patch. Then one cross-attention layer in which program tokens
    are the queries and the attended morphology and neighbour tokens are keys and values
    (program tokens as queries over morphology and neighbour tokens, in the spirit of the
    pathway tokens of SurvPath, Jaume et al., CVPR 2024, and MMP, Song et al., ICML 2024,
    both verified at abstract level; the attention direction is this repository's choice).
    Attention is ``F.scaled_dot_product_attention`` with head_dim fixed by config; when no
    mask is needed (the no_neighbours variant, and the program cross-attention) ``attn_mask``
    is None so the fused flash kernel
    is eligible (docs/ARCHITECTURE_V3.md section 8).

Heads
    program scores     per program a Linear(d_model, n_scales) on its attended token,
                       ordered to match ctx.targets (program-major, scale-minor);
    log-variance       same layout (heteroscedastic Gaussian, D30), or None;
    composition        Linear on the pooled centre token (multi-task, D15);
    morph_pooled       the pooled centre token projected to d_model for the alignment loss.

Variants (flags passed at forward time, so one weight set serves every ablation)
    no_neighbours, no_grid, no_program_tokens (skips every token and attention layer and
    feeds the raw centre feature vector, the v2 ProgramNet input, into a v2-shaped 2-layer
    MLP head, so the variant measures what the token architecture adds; the v2 ProgramNet
    with its coordinate prior remains the regression reference in tests/regression),
    no_comp. ``no_smooth`` lives in the loss.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging_utils import say
from .tokens import MorphologyTokens, NeighbourTokens, ProgramTokens


class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout: float):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q = nn.Linear(d_model, inner)
        self.kv = nn.Linear(d_model, 2 * inner)
        self.out = nn.Linear(inner, d_model)
        self.dropout = dropout

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        b, nq, _ = x_q.shape
        nk = x_kv.shape[1]
        q = self.q(x_q).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x_kv).view(b, nk, 2, self.n_heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = None if mask is None else mask[:, None, :, :]            # (B, 1, nq, nk) bool, True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = self.out(out.transpose(1, 2).reshape(b, nq, -1))
        weights = None
        if need_weights:
            scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            weights = scores.softmax(-1).mean(1)                                  # (B, nq, nk) averaged over heads
        return out, weights


class _SelfBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_heads, head_dim, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, mask)[0]
        return x + self.ff(self.n2(x))


def block_mask(n_morph: int, k: int, batch: int, device) -> Optional[torch.Tensor]:
    """(B, n, n) boolean mask. Morphology tokens see everything; neighbour j sees itself and the centre token."""
    if k == 0:
        return None
    n = n_morph + k
    m = torch.zeros(n, n, dtype=torch.bool, device=device)
    m[:n_morph, :] = True
    idx = torch.arange(k, device=device)
    m[n_morph + idx, n_morph + idx] = True
    m[n_morph:, 0] = True
    return m.unsqueeze(0).expand(batch, -1, -1)


class UnifiedProgramModel(nn.Module):
    def __init__(self, spec: Dict):
        super().__init__()
        self.spec = dict(spec)
        d = int(spec["d_model"])
        self.d_model = d
        self.grid = int(spec["grid"])
        self.k = int(spec["k"])
        self.n_programs = int(spec["n_programs"])
        self.n_scales = int(spec["n_scales"])
        self.n_targets = self.n_programs * self.n_scales
        self.uncertainty = spec.get("uncertainty", "gaussian")

        self.morph = MorphologyTokens(int(spec["d_centre"]), int(spec["d_grid"]), d, self.grid)
        self.nbr = NeighbourTokens(int(spec["d_centre"]), d, int(spec["n_fourier"]), float(spec["fourier_scale"]),
                                   int(spec["slide_emb_dim"]), int(spec["n_samples"]))
        init_w = spec.get("program_init_weights")
        self.prog = ProgramTokens(self.n_programs, int(spec["n_genes"]), d,
                                  None if init_w is None else np.asarray(init_w, dtype=np.float32))
        self.layers = nn.ModuleList([_SelfBlock(d, int(spec["n_heads"]), int(spec["head_dim"]), float(spec["dropout"]))
                                     for _ in range(int(spec["n_layers"]))])
        self.cross_norm_q = nn.LayerNorm(d)
        self.cross_norm_kv = nn.LayerNorm(d)
        self.cross = _Attention(d, int(spec["n_heads"]), int(spec["head_dim"]), float(spec["dropout"]))
        self.cross_ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        self.score_heads = nn.ModuleList([nn.Linear(d, self.n_scales) for _ in range(self.n_programs)])
        self.logvar_heads = nn.ModuleList([nn.Linear(d, self.n_scales) for _ in range(self.n_programs)]) \
            if self.uncertainty == "gaussian" else None
        self.comp_head = nn.Linear(d, int(spec["n_lineages"]))
        self.align_proj = nn.Linear(d, int(spec.get("d_align", d)))

        # v2-shaped MLP head for the no_program_tokens variant: the raw centre feature vector (PCA embedding plus
        # morphometrics, exactly the v2 ProgramNet input) -> 2-layer MLP -> targets. The attention layers are
        # skipped in that branch, so the variant measures what the token architecture adds over the v2 head.
        hidden = int(spec.get("mlp_hidden", 256))
        self.mlp = nn.Sequential(nn.Linear(int(spec["d_centre"]), hidden), nn.GELU(), nn.Dropout(float(spec["dropout"])),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(float(spec["dropout"])))
        self.mlp_head = nn.Linear(hidden, self.n_targets)
        self.mlp_logvar = nn.Linear(hidden, self.n_targets) if self.uncertainty == "gaussian" else None
        self.mlp_comp = nn.Linear(hidden, int(spec["n_lineages"]))
        self.align_proj_mlp = nn.Linear(hidden, int(spec.get("d_align", d)))

    # ------------------------------------------------------------------
    def unseen_slide_index(self) -> int:
        return self.nbr.unseen_slide_index()

    def forward(self, centre_emb: torch.Tensor, grid_tokens: torch.Tensor, nb_emb: Optional[torch.Tensor],
                nb_offsets_um: Optional[torch.Tensor], sid: torch.Tensor,
                variant_flags: Optional[Dict[str, bool]] = None, need_attn: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        flags = variant_flags or {}
        # Two spellings are accepted: the trainer passes use_<part>=False, tests may pass no_<part>=True.
        use_grid = flags.get("use_grid", not flags.get("no_grid", False)) and grid_tokens is not None
        use_nb = (flags.get("use_neighbours", not flags.get("no_neighbours", False))
                  and nb_emb is not None and nb_emb.shape[1] > 0)
        b = centre_emb.shape[0]

        if (not flags.get("use_program_tokens", not flags.get("no_program_tokens", False))):
            # v2 path: no tokens, no attention; the centre feature vector goes straight into the MLP head
            h = self.mlp(centre_emb)
            out: Dict[str, Optional[torch.Tensor]] = dict(attn=None, log_var=None)
            out["mean"] = self.mlp_head(h)
            out["log_var"] = self.mlp_logvar(h) if self.mlp_logvar is not None else None
            out["comp_logits"] = None if not flags.get("use_comp", not flags.get("no_comp", False)) else self.mlp_comp(h)
            out["morph_pooled"] = self.align_proj_mlp(h)
            return out

        tokens = self.morph(centre_emb, grid_tokens, use_grid=use_grid)                  # (B, n_m, d)
        n_m = tokens.shape[1]
        k = 0
        if use_nb:
            nb_tok = self.nbr(nb_emb, nb_offsets_um, sid)                                 # (B, k, d)
            tokens = torch.cat([tokens, nb_tok], dim=1)
            k = nb_tok.shape[1]
        mask = block_mask(n_m, k, b, tokens.device)
        for layer in self.layers:
            tokens = layer(tokens, mask)
        centre = tokens[:, 0]
        out: Dict[str, Optional[torch.Tensor]] = dict(morph_pooled=self.align_proj(centre), attn=None, log_var=None)

        q = self.cross_norm_q(self.prog(b))                                                # (B, P, d)
        attended, weights = self.cross(q, self.cross_norm_kv(tokens), None, need_weights=need_attn)
        pt = q + attended
        pt = pt + self.cross_ff(pt)
        mean = torch.stack([self.score_heads[p](pt[:, p]) for p in range(self.n_programs)], dim=1)   # (B, P, S)
        out["mean"] = mean.reshape(b, self.n_targets)
        if self.logvar_heads is not None:
            lv = torch.stack([self.logvar_heads[p](pt[:, p]) for p in range(self.n_programs)], dim=1)
            out["log_var"] = lv.reshape(b, self.n_targets)
        out["comp_logits"] = None if not flags.get("use_comp", not flags.get("no_comp", False)) else self.comp_head(centre)
        out["attn"] = weights
        out["n_morph_tokens"] = torch.tensor(n_m)
        return out

    # ------------------------------------------------------------------
    def block_parameter_counts(self) -> Dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return dict(morphology_tokens=count(self.morph), neighbour_tokens=count(self.nbr), program_tokens=count(self.prog),
                    self_attention=count(self.layers), cross_attention=count(self.cross) + count(self.cross_ff)
                    + count(self.cross_norm_q) + count(self.cross_norm_kv),
                    score_heads=count(self.score_heads) + (count(self.logvar_heads) if self.logvar_heads is not None else 0),
                    composition_head=count(self.comp_head), align_proj=count(self.align_proj),
                    v2_mlp_variant=count(self.mlp) + count(self.mlp_head) + count(self.mlp_comp) + count(self.align_proj_mlp)
                    + (count(self.mlp_logvar) if self.mlp_logvar is not None else 0),
                    total=count(self))

    def describe(self) -> None:
        say(f"UnifiedProgramModel: d_model={self.d_model} grid={self.grid} k={self.k} programs={self.n_programs} "
            f"scales={self.n_scales} targets={self.n_targets} uncertainty={self.uncertainty}")
        for name, n in self.block_parameter_counts().items():
            say(f"{name:<20} {n:>12,} params", indent=1)
