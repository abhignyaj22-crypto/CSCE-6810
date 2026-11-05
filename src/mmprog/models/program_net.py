"""ProgramNet: program scores from morphology features plus a shared spatial prior.

    y = f(features) + g(fourier(x, y), slide_embedding)

- ``f``: two-layer MLP on [PCA(embedding), standardised morphometrics].
- ``g``: random Fourier features of within-slide normalised coordinates,
  concatenated with a learned per-slide embedding, through a small MLP. The
  block is shared across slides, so the parameter count does not grow with the
  cohort. The embedding row with index ``n_samples`` is reserved for an unseen
  slide and is never trained, so it stays at zero: on a new slide ``g`` reduces
  to the smooth spatial residual learned from the training cohort.
- composition head: lineage logits from the same hidden state. This is the
  cross-omics consistency idea repurposed as a cell-identity auxiliary task.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..logging_utils import say


class ProgramNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_targets: int,
        n_lin: int,
        n_samples: int,
        hidden: int = 256,
        dropout: float = 0.3,
        use_spatial: bool = True,
        use_comp: bool = True,
        n_fourier: int = 32,
        fourier_scale: float = 6.0,
        slide_emb_dim: int = 16,
    ):
        super().__init__()
        self.d_in, self.n_targets, self.n_lin, self.n_samples = d_in, n_targets, n_lin, n_samples
        self.use_spatial, self.use_comp = use_spatial, use_comp

        self.f = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_targets)
        self.comp = nn.Linear(hidden, n_lin) if use_comp else None

        if use_spatial:
            # fixed random projection: sin/cos of (xy @ B) gives a smooth, multi-scale basis
            self.register_buffer("B", torch.randn(2, n_fourier) * fourier_scale)
            self.semb = nn.Embedding(n_samples + 1, slide_emb_dim)
            nn.init.zeros_(self.semb.weight)
            self.g = nn.Sequential(
                nn.Linear(2 * n_fourier + slide_emb_dim, 64), nn.GELU(), nn.Linear(64, n_targets)
            )
            # start as a zero residual so the feature branch is learned first
            nn.init.zeros_(self.g[-1].weight)
            nn.init.zeros_(self.g[-1].bias)

    def forward(
        self, x: torch.Tensor, xy: torch.Tensor, sid: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.f(x)
        y = self.head(h)
        if self.use_spatial:
            ff = xy @ self.B
            spatial_in = torch.cat([torch.sin(ff), torch.cos(ff), self.semb(sid)], dim=1)
            y = y + self.g(spatial_in)
        comp_logits = self.comp(h) if self.comp is not None else None
        return y, comp_logits

    def unseen_slide_index(self) -> int:
        """Slide id to use at inference on a slide the model has never seen."""
        return self.n_samples

    def describe(self) -> None:
        """Print the parameter count of every block."""
        def count(mod: Optional[nn.Module]) -> int:
            return 0 if mod is None else sum(p.numel() for p in mod.parameters())

        total = sum(p.numel() for p in self.parameters())
        say(f"ProgramNet: d_in={self.d_in} targets={self.n_targets} lineages={self.n_lin} slides={self.n_samples}")
        say(f"feature MLP f: {count(self.f):,} params", indent=1)
        say(f"program head: {count(self.head):,} params", indent=1)
        say(f"composition head: {count(self.comp):,} params ({'on' if self.use_comp else 'off'})", indent=1)
        if self.use_spatial:
            say(f"spatial prior g + slide embedding: {count(self.g) + count(self.semb):,} params", indent=1)
        else:
            say("spatial prior: off", indent=1)
        say(f"total trainable: {total:,}", indent=1)
