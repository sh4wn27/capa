"""Competing risks loss functions for DeepHit training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def deephit_loss(
    logits: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    alpha: float = 0.5,
    sigma: float = 0.1,
) -> Tensor:
    """Compute the DeepHit combined loss (log-likelihood + ranking).

    Parameters
    ----------
    logits : Tensor
        Raw model output of shape ``(batch, num_events, time_bins)``.
        Will be softmax-normalised over the joint (event × time) space.
    event_times : Tensor
        Observed event/censoring time indices of shape ``(batch,)``, dtype long.
    event_types : Tensor
        Event type indicator of shape ``(batch,)``.
        0 = censored, 1..K = cause-specific event.
    alpha : float
        Weight for the ranking loss term (0 = log-likelihood only).
    sigma : float
        Bandwidth for the ranking loss exponential kernel.

    Returns
    -------
    Tensor
        Scalar combined loss.
    """
    batch, num_events, time_bins = logits.shape

    # Joint distribution over (event type, time): softmax over flattened space
    joint = F.softmax(logits.view(batch, -1), dim=-1).view(batch, num_events, time_bins)

    # Cause-specific cumulative incidence: sum joint over time up to t
    cif = torch.cumsum(joint, dim=2)  # (batch, num_events, time_bins)

    ll = _log_likelihood(joint, cif, event_times, event_types, time_bins)
    rank = _ranking_loss(cif, event_times, event_types, sigma)

    return (1.0 - alpha) * ll + alpha * rank


def _log_likelihood(
    joint: Tensor,
    cif: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    time_bins: int,
) -> Tensor:
    """Negative log-likelihood term of the DeepHit loss.

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution ``(batch, num_events, time_bins)``.
    cif : Tensor
        Cumulative incidence functions ``(batch, num_events, time_bins)``.
    event_times : Tensor
        Observed times ``(batch,)``.
    event_types : Tensor
        Event indicators ``(batch,)``.
    time_bins : int
        Number of discrete time bins.

    Returns
    -------
    Tensor
        Scalar negative log-likelihood.
    """
    eps = 1e-8
    batch = event_times.shape[0]

    # For uncensored subjects: p(T=t, K=k)
    t_idx = event_times.clamp(0, time_bins - 1)
    k_idx = (event_types - 1).clamp(0)  # shift so event 1 → index 0

    uncensored_mask = event_types > 0
    nll = torch.zeros(batch, device=joint.device)

    if uncensored_mask.any():
        p = joint[uncensored_mask, k_idx[uncensored_mask], t_idx[uncensored_mask]]
        nll[uncensored_mask] = -torch.log(p + eps)

    # For censored subjects: 1 - sum_k CIF_k(t)
    if (~uncensored_mask).any():
        total_cif = cif[~uncensored_mask].sum(dim=1)  # (censored, time_bins)
        t_c = t_idx[~uncensored_mask]
        survival = 1.0 - total_cif[torch.arange(t_c.shape[0]), t_c]
        nll[~uncensored_mask] = -torch.log(survival + eps)

    return nll.mean()


def _ranking_loss(
    cif: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    sigma: float,
) -> Tensor:
    """Cause-specific ranking loss term of DeepHit.

    Parameters
    ----------
    cif : Tensor
        Cumulative incidence functions ``(batch, num_events, time_bins)``.
    event_times : Tensor
        Observed times ``(batch,)``.
    event_types : Tensor
        Event indicators ``(batch,)``.
    sigma : float
        Exponential kernel bandwidth.

    Returns
    -------
    Tensor
        Scalar ranking loss.
    """
    batch, num_events, time_bins = cif.shape
    eps = 1e-8
    loss = torch.tensor(0.0, device=cif.device)
    count = 0

    for k in range(1, num_events + 1):
        mask_k = event_types == k
        if mask_k.sum() < 2:
            continue
        t_k = event_times[mask_k]
        cif_k = cif[mask_k, k - 1, :]  # (n_k, time_bins)

        for i in range(len(t_k)):
            t_i = t_k[i]
            # j's that had event k AFTER subject i
            j_mask = t_k > t_i
            if not j_mask.any():
                continue
            eta_i = cif_k[i, t_i.clamp(0, time_bins - 1)]
            eta_j = cif_k[j_mask, t_i.clamp(0, time_bins - 1)]
            diff = eta_i - eta_j
            loss = loss + torch.exp(-diff / (sigma + eps)).mean()
            count += 1

    return loss / max(count, 1)
