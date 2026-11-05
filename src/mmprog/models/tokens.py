"""Token builders for the unified model: morphology blocks, neighbour cells, program queries.

Morphology tokens   one centre token (pooled patch embedding) plus g*g block tokens from
                    the encoder feature map, each projected to d_model with a learned
                    2-D block position embedding (own choice; see DESIGN_V3).
Neighbour tokens    the k nearest cells' centre embeddings projected to d_model, plus
                    random Fourier features of the relative offset (dx, dy) in microns
                    (D16) and a learned slide embedding whose last index is the zero slot
                    for an unseen slide (as in v2).
Program tokens      one learned query per program, initialised by projecting the program's
                    gene-weight vector through a gene-to-token linear map so the query
                    carries the gene content of the program from the first step
                    (own choice, in the spirit of pathway tokens in SurvPath, CVPR 2024).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MorphologyTokens(nn.Module):
    def __init__(self, d_in_centre: int, d_in_grid: int, d_model: int, grid: int):
        super().__init__()
        self.grid = grid
        self.centre_proj = nn.Linear(d_in_centre, d_model)
        self.block_proj = nn.Linear(d_in_grid, d_model)
        self.centre_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.row_pos = nn.Parameter(torch.randn(grid, d_model) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(grid, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, centre: torch.Tensor, blocks: torch.Tensor, use_grid: bool = True) -> torch.Tensor:
        """centre (B, D_c), blocks (B, g*g, D_g) -> (B, 1 + g*g, d_model) or (B, 1, d_model)."""
        c = self.centre_proj(centre).unsqueeze(1) + self.centre_pos
        if not use_grid:
            return self.norm(c)
        g = self.grid
        pos = (self.row_pos[:, None, :] + self.col_pos[None, :, :]).reshape(1, g * g, -1)
        b = self.block_proj(blocks) + pos
        return self.norm(torch.cat([c, b], dim=1))


class NeighbourTokens(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_fourier: int, fourier_scale: float, slide_emb_dim: int, n_samples: int):
        super().__init__()
        self.n_samples = n_samples
        self.proj = nn.Linear(d_in, d_model)
        # offsets are in microns; the scale divides them so a 30 um neighbourhood spans about one period
        self.register_buffer("B", torch.randn(2, n_fourier) * (fourier_scale / 30.0))
        self.offset_proj = nn.Linear(2 * n_fourier, d_model)
        self.semb = nn.Embedding(n_samples + 1, slide_emb_dim)
        nn.init.zeros_(self.semb.weight)
        self.slide_proj = nn.Linear(slide_emb_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def unseen_slide_index(self) -> int:
        return self.n_samples

    def forward(self, nb_emb: torch.Tensor, nb_offsets_um: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        """nb_emb (B, k, D), nb_offsets_um (B, k, 2), sid (B,) -> (B, k, d_model)."""
        ff = nb_offsets_um @ self.B
        offset = self.offset_proj(torch.cat([torch.sin(ff), torch.cos(ff)], dim=-1))
        slide = self.slide_proj(self.semb(sid)).unsqueeze(1)
        return self.norm(self.proj(nb_emb) + offset + slide)


class ProgramTokens(nn.Module):
    def __init__(self, n_programs: int, n_genes: int, d_model: int, init_weights: np.ndarray | None = None):
        super().__init__()
        self.gene_to_token = nn.Linear(n_genes, d_model)
        if init_weights is not None:
            w = torch.as_tensor(np.asarray(init_weights, dtype=np.float32))
            assert w.shape == (n_programs, n_genes), f"init_weights {tuple(w.shape)} != ({n_programs}, {n_genes})"
            w = w / (w.norm(dim=1, keepdim=True) + 1e-6)
            self.register_buffer("gene_weights", w)
        else:
            self.register_buffer("gene_weights", torch.zeros(n_programs, n_genes))
        self.query_residual = nn.Parameter(torch.randn(n_programs, d_model) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        q = self.gene_to_token(self.gene_weights) + self.query_residual      # (P, d_model)
        return q.unsqueeze(0).expand(batch_size, -1, -1)
