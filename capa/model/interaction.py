"""Donor-recipient cross-attention interaction network."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionInteraction(nn.Module):
    """Model immunological conflict between donor and recipient HLA via cross-attention.

    The donor allele matrix attends to the recipient allele matrix (and vice versa),
    producing a compact interaction representation.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of input ESM-2 allele embeddings.
    interaction_dim : int
        Output dimensionality after projection.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of stacked cross-attention layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        interaction_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

        # Donor → recipient cross-attention
        self.donor_to_recip = nn.ModuleList(
            [
                nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        # Recipient → donor cross-attention
        self.recip_to_donor = nn.ModuleList(
            [
                nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Project concatenated interaction to interaction_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, interaction_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim * 2, interaction_dim),
        )

    def forward(self, donor: Tensor, recipient: Tensor) -> Tensor:
        """Compute interaction features from donor and recipient allele matrices.

        Parameters
        ----------
        donor : Tensor
            Donor allele embeddings of shape ``(batch, n_loci, embedding_dim)``.
        recipient : Tensor
            Recipient allele embeddings of shape ``(batch, n_loci, embedding_dim)``.

        Returns
        -------
        Tensor
            Interaction feature vector of shape ``(batch, interaction_dim)``.
        """
        d, r = donor, recipient
        for d2r_attn, r2d_attn in zip(self.donor_to_recip, self.recip_to_donor):
            d_out, _ = d2r_attn(query=d, key=r, value=r)
            r_out, _ = r2d_attn(query=r, key=d, value=d)
            d = self.layer_norm(d + self.dropout(d_out))
            r = self.layer_norm(r + self.dropout(r_out))

        # Global mean-pool over loci
        d_pooled = d.mean(dim=1)  # (batch, embedding_dim)
        r_pooled = r.mean(dim=1)

        combined = torch.cat([d_pooled, r_pooled], dim=-1)
        return self.projection(combined)  # (batch, interaction_dim)
