"""DeepHit competing-risks survival head."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DeepHitHead(nn.Module):
    """MLP survival head outputting a joint distribution over event types and times.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    num_events : int
        Number of competing events (e.g. 3 for GvHD, relapse, TRM).
    time_bins : int
        Number of discrete time bins.
    hidden_dim : int
        Hidden layer width.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_events: int = 3,
        time_bins: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_events = num_events
        self.time_bins = time_bins

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Separate output head per event
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, time_bins) for _ in range(num_events)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits for the joint (event × time) distribution.

        Parameters
        ----------
        x : Tensor
            Feature vector of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, num_events, time_bins)``.
            Apply softmax over the flattened space for probabilities.
        """
        h = self.shared(x)
        return nn.functional.stack([head(h) for head in self.heads], dim=1)  # type: ignore[attr-defined]
