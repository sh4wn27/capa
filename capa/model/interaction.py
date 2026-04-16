"""Donor-recipient HLA interaction networks.

Two implementations are provided:

* :class:`CrossAttentionInteraction` — the main model: bidirectional multi-head
  cross-attention between donor and recipient allele matrices, with extractable
  per-head attention weights for interpretability.

* :class:`DiffMLPInteraction` — a simple ablation baseline that concatenates
  the mean donor embedding, mean recipient embedding, and their element-wise
  absolute difference, then passes through an MLP.

Both modules share the same call signature ``forward(donor, recipient) -> Tensor``
and output a vector of shape ``(batch, interaction_dim)``.

Attention weight access
-----------------------
After a forward pass through :class:`CrossAttentionInteraction`, the attention
weights from each layer and direction are stored in
:attr:`CrossAttentionInteraction.last_attn_weights`::

    model = CrossAttentionInteraction(...)
    out = model(donor, recipient)
    weights = model.last_attn_weights
    # weights.donor_to_recip[layer_idx] → Tensor (batch, n_loci_q, n_loci_kv)
    # weights.recip_to_donor[layer_idx] → Tensor (batch, n_loci_q, n_loci_kv)

Thread safety: ``last_attn_weights`` is overwritten on every forward call.
Do not share a single module instance across threads.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class AttentionWeights(NamedTuple):
    """Attention weight tensors from one forward pass.

    Attributes
    ----------
    donor_to_recip : list[Tensor]
        Per-layer attention weights for the donor→recipient direction.
        Each tensor has shape ``(batch, n_loci_donor, n_loci_recip)``
        (averaged over heads).
    recip_to_donor : list[Tensor]
        Per-layer attention weights for the recipient→donor direction.
        Each tensor has shape ``(batch, n_loci_recip, n_loci_donor)``
        (averaged over heads).
    """

    donor_to_recip: list[Tensor]
    recip_to_donor: list[Tensor]


# ---------------------------------------------------------------------------
# Helper: single bidirectional cross-attention block
# ---------------------------------------------------------------------------

class _CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention block with residual connections.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimensionality (must be divisible by *num_heads*).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability applied inside attention and after residual add.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d2r = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.r2d = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_d = nn.LayerNorm(embedding_dim)
        self.norm_r = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, donor: Tensor, recip: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one bidirectional cross-attention step.

        Parameters
        ----------
        donor : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        recip : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.

        Returns
        -------
        donor_out : Tensor
            Updated donor representation, same shape as *donor*.
        recip_out : Tensor
            Updated recipient representation, same shape as *recip*.
        w_d2r : Tensor
            Averaged attention weights, shape ``(batch, n_loci_donor, n_loci_recip)``.
        w_r2d : Tensor
            Averaged attention weights, shape ``(batch, n_loci_recip, n_loci_donor)``.
        """
        d_attn, w_d2r = self.d2r(
            query=donor,
            key=recip,
            value=recip,
            need_weights=True,
            average_attn_weights=True,
        )
        r_attn, w_r2d = self.r2d(
            query=recip,
            key=donor,
            value=donor,
            need_weights=True,
            average_attn_weights=True,
        )
        donor_out = self.norm_d(donor + self.drop(d_attn))
        recip_out = self.norm_r(recip + self.drop(r_attn))
        return donor_out, recip_out, w_d2r, w_r2d


# ---------------------------------------------------------------------------
# Main model: CrossAttentionInteraction
# ---------------------------------------------------------------------------

class CrossAttentionInteraction(nn.Module):
    """Bidirectional multi-head cross-attention interaction network.

    The donor allele matrix attends to the recipient allele matrix and vice
    versa over *num_layers* stacked blocks.  After the final block both
    sequences are mean-pooled over the loci dimension and concatenated before
    projection to *interaction_dim*.

    Attention weights from every layer are stored in
    :attr:`last_attn_weights` after each forward pass for interpretability.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of input ESM-2 allele embeddings.  Default: 1280.
    interaction_dim : int
        Output dimensionality of the interaction feature vector.  Default: 128.
    num_heads : int
        Number of attention heads.  *embedding_dim* must be divisible by this.
    num_layers : int
        Number of stacked bidirectional cross-attention blocks.
    dropout : float
        Dropout probability used in attention and projection layers.
    use_pos_embed : bool
        If ``True``, add a learned per-locus positional embedding (dim 32)
        to each allele vector before the first cross-attention block.
        The embedding table supports up to 16 loci; actual *n_loci* is
        determined at runtime.  Default: ``False``.

    Raises
    ------
    ValueError
        If *embedding_dim* is not divisible by *num_heads*.
    """

    #: Maximum number of loci supported by the positional embedding table.
    MAX_LOCI: int = 16

    def __init__(
        self,
        embedding_dim: int = 1280,
        interaction_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pos_embed: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self._embedding_dim = embedding_dim
        self._interaction_dim = interaction_dim
        self._use_pos_embed = use_pos_embed

        self.blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # V2: optional learned per-locus positional embedding projected to
        # embedding_dim so it can be added to the allele embeddings.
        if use_pos_embed:
            _POS_DIM = 32
            self.pos_embed = nn.Embedding(self.MAX_LOCI, _POS_DIM)
            self.pos_proj = nn.Linear(_POS_DIM, embedding_dim, bias=False)
        else:
            self.pos_embed = None  # type: ignore[assignment]
            self.pos_proj = None   # type: ignore[assignment]

        # Project concatenated pooled vectors → interaction_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, interaction_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim * 4, interaction_dim),
        )

        # Populated by forward(); None until first call
        self._last_attn_weights: AttentionWeights | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Input embedding dimensionality."""
        return self._embedding_dim

    @property
    def interaction_dim(self) -> int:
        """Output interaction feature dimensionality."""
        return self._interaction_dim

    @property
    def last_attn_weights(self) -> AttentionWeights | None:
        """Attention weights from the most recent forward pass, or ``None``.

        Each entry in the lists has shape ``(batch, n_loci_q, n_loci_kv)``,
        averaged over attention heads.

        This is ``None`` before the first forward call.
        """
        return self._last_attn_weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, donor: Tensor, recipient: Tensor) -> Tensor:
        """Compute interaction features via bidirectional cross-attention.

        Parameters
        ----------
        donor : Tensor
            Donor allele embeddings, shape ``(batch, n_loci, embedding_dim)``.
        recipient : Tensor
            Recipient allele embeddings, shape ``(batch, n_loci, embedding_dim)``.

        Returns
        -------
        Tensor
            Interaction feature vector, shape ``(batch, interaction_dim)``.
        """
        d, r = donor, recipient

        # Add learned per-locus positional bias before the first block
        if self._use_pos_embed and self.pos_embed is not None and self.pos_proj is not None:
            n_loci = d.size(1)
            positions = torch.arange(n_loci, device=d.device)  # (n_loci,)
            pos_bias = self.pos_proj(self.pos_embed(positions))  # (n_loci, embedding_dim)
            d = d + pos_bias.unsqueeze(0)  # broadcast over batch
            r = r + pos_bias.unsqueeze(0)

        d2r_weights: list[Tensor] = []
        r2d_weights: list[Tensor] = []

        for block in self.blocks:
            d, r, w_d2r, w_r2d = block(d, r)
            d2r_weights.append(w_d2r)
            r2d_weights.append(w_r2d)

        self._last_attn_weights = AttentionWeights(
            donor_to_recip=d2r_weights,
            recip_to_donor=r2d_weights,
        )

        # Global mean-pool over loci → (batch, embedding_dim) each
        d_pooled = d.mean(dim=1)
        r_pooled = r.mean(dim=1)

        combined = torch.cat([d_pooled, r_pooled], dim=-1)  # (batch, 2 * embedding_dim)
        return self.projection(combined)  # (batch, interaction_dim)


# ---------------------------------------------------------------------------
# Ablation baseline: DiffMLPInteraction
# ---------------------------------------------------------------------------

class DiffMLPInteraction(nn.Module):
    """Element-wise difference + MLP baseline for ablation studies.

    Computes a fixed relational feature vector from the mean-pooled donor and
    recipient embeddings::

        features = [donor_mean | recipient_mean | |donor_mean − recipient_mean|]

    This concatenation (3 × *embedding_dim*) is passed through a two-layer MLP
    to produce the interaction vector.  No attention mechanism is used, so
    :attr:`last_attn_weights` is always ``None``.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of input ESM-2 allele embeddings.  Default: 1280.
    interaction_dim : int
        Output dimensionality of the interaction feature vector.  Default: 128.
    hidden_dim : int | None
        Hidden layer width.  Defaults to ``interaction_dim * 4``.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        interaction_dim: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._interaction_dim = interaction_dim

        hidden = hidden_dim if hidden_dim is not None else interaction_dim * 4
        # Input: [donor | recipient | |donor - recipient|] → 3 × embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, interaction_dim),
        )

    # ------------------------------------------------------------------
    # Properties (mirror CrossAttentionInteraction interface)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Input embedding dimensionality."""
        return self._embedding_dim

    @property
    def interaction_dim(self) -> int:
        """Output interaction feature dimensionality."""
        return self._interaction_dim

    @property
    def last_attn_weights(self) -> None:
        """Always ``None`` — this baseline has no attention mechanism."""
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, donor: Tensor, recipient: Tensor) -> Tensor:
        """Compute interaction features via element-wise difference + MLP.

        Parameters
        ----------
        donor : Tensor
            Donor allele embeddings, shape ``(batch, n_loci, embedding_dim)``.
        recipient : Tensor
            Recipient allele embeddings, shape ``(batch, n_loci, embedding_dim)``.

        Returns
        -------
        Tensor
            Interaction feature vector, shape ``(batch, interaction_dim)``.
        """
        d = donor.mean(dim=1)    # (batch, embedding_dim)
        r = recipient.mean(dim=1)
        diff = (d - r).abs()     # element-wise |donor − recipient|
        combined = torch.cat([d, r, diff], dim=-1)  # (batch, 3 * embedding_dim)
        return self.mlp(combined)  # (batch, interaction_dim)
