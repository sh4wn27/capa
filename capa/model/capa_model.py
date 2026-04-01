"""Full CAPA model: embedding → interaction → survival."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from capa.model.interaction import CrossAttentionInteraction
from capa.model.survival import DeepHitHead


class CAPAModel(nn.Module):
    """End-to-end CAPA model for competing-risks survival prediction.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of ESM-2 HLA allele embeddings.
    n_loci : int
        Number of HLA loci considered per subject.
    clinical_dim : int
        Dimensionality of the clinical covariate vector.
    interaction_dim : int
        Output dimensionality of the cross-attention interaction network.
    num_events : int
        Number of competing events.
    time_bins : int
        Number of discrete time bins for the DeepHit head.
    num_heads : int
        Number of attention heads in the interaction network.
    num_layers : int
        Number of cross-attention layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        n_loci: int = 5,
        clinical_dim: int = 32,
        interaction_dim: int = 128,
        num_events: int = 3,
        time_bins: int = 100,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_loci = n_loci
        self.embedding_dim = embedding_dim

        self.interaction = CrossAttentionInteraction(
            embedding_dim=embedding_dim,
            interaction_dim=interaction_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        combined_dim = interaction_dim + clinical_dim
        self.survival_head = DeepHitHead(
            input_dim=combined_dim,
            num_events=num_events,
            time_bins=time_bins,
            dropout=dropout,
        )

    def forward(
        self,
        donor_embeddings: Tensor,
        recipient_embeddings: Tensor,
        clinical_features: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        donor_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        recipient_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        clinical_features : Tensor
            Shape ``(batch, clinical_dim)``.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, num_events, time_bins)``.
        """
        interaction_feats = self.interaction(donor_embeddings, recipient_embeddings)
        combined = torch.cat([interaction_feats, clinical_features], dim=-1)
        return self.survival_head(combined)
