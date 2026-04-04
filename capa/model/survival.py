"""Competing-risks survival heads for the CAPA model.

Two implementations are provided with a shared interface:

* :class:`DeepHitHead` — joint distribution over ``(event_type × time_bin)``
  via a softmax over the flattened space (Lee et al., 2018).

* :class:`CauseSpecificHazardHead` — separate discrete sub-hazard function per
  event type, combined via the discrete competing-risks product formula to give
  valid cumulative incidence functions (CIF) guaranteed to stay in ``[0, 1]``
  and be monotone non-decreasing.

Shared interface
----------------
Both heads expose:

``forward(x) -> Tensor``
    Returns the *raw* model output (logits for DeepHit, sub-hazards for CSH).
    Shape: ``(batch, num_events, time_bins)``.

``cif(x) -> Tensor``
    Returns cumulative incidence functions in ``[0, 1]``.
    Shape: ``(batch, num_events, time_bins)``.
    Each entry ``CIF[b, k, t]`` is the predicted probability that subject *b*
    experiences event *k* at or before time bin *t*.

Discrete competing-risks recap
-------------------------------
Let time be discretised into bins ``t = 0, 1, ..., T-1``.

*DeepHit*: models the joint distribution directly::

    p(T=t, K=k) = softmax_over_all_(k,t)(logits)[k, t]
    CIF_k(t)    = Σ_{s≤t} p(T=s, K=k)

*Cause-specific hazard*: models sub-hazards h_k(t) ∈ (0,1) at each t,
with Σ_k h_k(t) < 1 enforced via a (K+1)-way softmax where the extra class
represents "no event at this step"::

    overall survival:  S(t) = Π_{s≤t} [1 − Σ_k h_k(s)]
    CIF_k(t)        = Σ_{s≤t} h_k(s) · S(s−1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------

def _make_shared_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# DeepHit head
# ---------------------------------------------------------------------------

class DeepHitHead(nn.Module):
    """Joint-distribution competing-risks survival head (DeepHit).

    Models ``p(T=t, K=k)`` as a softmax over the flattened
    ``(num_events × time_bins)`` space.  Each event's CIF is the
    cumulative sum of its marginal probability mass up to time *t*.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    num_events : int
        Number of competing events (e.g. 3 for GvHD, relapse, TRM).
    time_bins : int
        Number of discrete time bins.
    hidden_dim : int
        Width of the two hidden layers in the shared backbone.
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

        self.shared = _make_shared_mlp(input_dim, hidden_dim, dropout)
        # One linear head per event; outputs are concatenated, then softmaxed jointly
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, time_bins) for _ in range(num_events)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute unnormalised logits over the joint (event × time) space.

        Parameters
        ----------
        x : Tensor
            Feature vector of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, num_events, time_bins)``.
            To obtain probabilities, apply softmax over the *flattened*
            last two dimensions (use :meth:`cif` for ready-to-use CIFs).
        """
        h = self.shared(x)
        return torch.stack([head(h) for head in self.heads], dim=1)

    def cif(self, x: Tensor) -> Tensor:
        """Compute cumulative incidence functions from the input features.

        Parameters
        ----------
        x : Tensor
            Feature vector of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            CIF tensor of shape ``(batch, num_events, time_bins)`` with values
            in ``[0, 1]``.  Each row is monotone non-decreasing over time bins.
        """
        logits = self.forward(x)
        batch = logits.shape[0]
        # Softmax over the *joint* (event × time) space
        joint = F.softmax(logits.view(batch, -1), dim=-1).view(
            batch, self.num_events, self.time_bins
        )
        return torch.cumsum(joint, dim=2)


# ---------------------------------------------------------------------------
# Cause-specific hazard head
# ---------------------------------------------------------------------------

class CauseSpecificHazardHead(nn.Module):
    """Cause-specific discrete sub-hazard competing-risks survival head.

    Models a separate sub-hazard function ``h_k(t)`` per competing event.
    At each time step the model outputs a ``(num_events + 1)``-way softmax,
    where the extra class represents "no event at this step", guaranteeing
    ``Σ_k h_k(t) ≤ 1`` for all ``t``.

    The cumulative incidence for event *k* is then::

        S(t)     = Π_{s ≤ t} no_event(s)          # overall survival
        CIF_k(t) = Σ_{s ≤ t} h_k(s) · S(s − 1)   # cause-k CIF

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    num_events : int
        Number of competing events.
    time_bins : int
        Number of discrete time bins.
    hidden_dim : int
        Width of the two hidden layers in the shared backbone.
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

        self.shared = _make_shared_mlp(input_dim, hidden_dim, dropout)
        # One head per event; the "no-event" class is computed implicitly
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, time_bins) for _ in range(num_events)]
        )
        # "No-event" head (Σ_k h_k(t) < 1 enforced via joint softmax)
        self.no_event_head = nn.Linear(hidden_dim, time_bins)

    def forward(self, x: Tensor) -> Tensor:
        """Compute cause-specific sub-hazard functions.

        At each time bin, the ``(num_events + 1)``-way softmax is applied
        over the event classes (including the implicit "no event" class),
        ensuring the sub-hazards sum to a value strictly less than 1.

        Parameters
        ----------
        x : Tensor
            Feature vector of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Sub-hazard tensor of shape ``(batch, num_events, time_bins)``
            with values in ``(0, 1)`` and ``hazards.sum(dim=1) < 1`` at each
            time bin.
        """
        h = self.shared(x)
        # Event logits: (batch, num_events, time_bins)
        event_logits = torch.stack([head(h) for head in self.heads], dim=1)
        # No-event logits: (batch, 1, time_bins)
        no_event_logits = self.no_event_head(h).unsqueeze(1)
        # Joint logits: (batch, num_events + 1, time_bins)
        all_logits = torch.cat([event_logits, no_event_logits], dim=1)
        # Softmax over the event dimension at each time step
        probs = F.softmax(all_logits, dim=1)  # (batch, num_events + 1, time_bins)
        # Return only the K event sub-hazards; no_event is probs[:, -1, :]
        return probs[:, : self.num_events, :]

    def cif(self, x: Tensor) -> Tensor:
        """Compute cumulative incidence functions.

        Parameters
        ----------
        x : Tensor
            Feature vector of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            CIF tensor of shape ``(batch, num_events, time_bins)`` with values
            in ``[0, 1]``.  Each row is monotone non-decreasing over time bins
            and ``CIF.sum(dim=1)`` is non-decreasing and bounded by 1.
        """
        hazards = self.forward(x)  # (batch, num_events, time_bins)
        return hazards_to_cif(hazards)


# ---------------------------------------------------------------------------
# Utility: convert cause-specific hazards → CIF
# ---------------------------------------------------------------------------

def hazards_to_cif(hazards: Tensor) -> Tensor:
    """Convert cause-specific sub-hazards to cumulative incidence functions.

    Implements the standard discrete competing-risks formula::

        no_event(t) = 1 − Σ_k h_k(t)
        S(t)        = Π_{s ≤ t} no_event(s)
        CIF_k(t)    = Σ_{s ≤ t} h_k(s) · S(s − 1)

    Parameters
    ----------
    hazards : Tensor
        Sub-hazard tensor of shape ``(batch, num_events, time_bins)`` with
        values in ``(0, 1)`` and ``hazards.sum(dim=1) ≤ 1``.

    Returns
    -------
    Tensor
        CIF tensor of shape ``(batch, num_events, time_bins)``.
    """
    batch, num_events, time_bins = hazards.shape
    eps = 1e-7

    # Overall "no-event" probability at each time step
    no_event = (1.0 - hazards.sum(dim=1)).clamp(min=eps)  # (batch, time_bins)

    # Overall survival S(t) = Π_{s≤t} no_event(s)
    # Computed via cumulative product along time dimension
    S = torch.cumprod(no_event, dim=1)  # (batch, time_bins)

    # S(t−1): prepend 1.0, drop the last column
    ones = torch.ones(batch, 1, device=hazards.device, dtype=hazards.dtype)
    S_prev = torch.cat([ones, S[:, :-1]], dim=1)  # (batch, time_bins)

    # CIF_k(t) = Σ_{s≤t} h_k(s) · S(s−1)
    increments = hazards * S_prev.unsqueeze(1)  # (batch, num_events, time_bins)
    return torch.cumsum(increments, dim=2)
