"""Competing-risks loss functions for DeepHit and cause-specific hazard training.

Loss overview
-------------
``deephit_loss``
    Combined DeepHit objective (Lee et al., 2018):
    ``L = (1 − α) · L_likelihood + α · L_ranking``

    * ``L_likelihood`` — negative log-likelihood of the joint distribution
      ``p(T=t, K=k)`` for uncensored subjects plus a survival term for censored.
    * ``L_ranking`` — pairwise concordance penalty that pushes the predicted CIF
      of subjects who experienced an event earlier above those whose events occur
      later (or who are censored).

``cause_specific_loss``
    Negative log-likelihood for the :class:`~capa.model.survival.CauseSpecificHazardHead`.
    Directly uses the discrete sub-hazard formulation without an additional
    ranking term (can be combined with ``_ranking_loss`` if desired).

Internal helpers (public for testing)
--------------------------------------
``_log_likelihood``
    NLL term computed from the joint probability tensor.
``_ranking_loss``
    Vectorised pairwise ranking loss (no Python loops over subjects).
``_survival_from_hazards``
    Cumulative log-survival from cause-specific sub-hazards.

References
----------
Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018).
DeepHit: A deep learning approach to survival analysis with competing risks.
AAAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# DeepHit combined loss
# ---------------------------------------------------------------------------

def deephit_loss(
    logits: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    *,
    alpha: float = 0.5,
    sigma: float = 0.1,
) -> Tensor:
    """Compute the DeepHit combined loss (log-likelihood + ranking).

    Parameters
    ----------
    logits : Tensor
        Raw model output of shape ``(batch, num_events, time_bins)``.
        Will be softmax-normalised over the joint ``(event × time)`` space
        inside this function — do **not** apply softmax beforehand.
    event_times : Tensor
        Observed event or censoring time *indices* of shape ``(batch,)``,
        dtype ``torch.long``.  Must be in ``[0, time_bins − 1]``.
    event_types : Tensor
        Event-type indicator of shape ``(batch,)``, dtype ``torch.long``.
        ``0`` = right-censored; ``1..K`` = cause-specific uncensored event.
    alpha : float
        Weight for the ranking loss term.
        ``0`` → pure log-likelihood; ``1`` → pure ranking loss.
    sigma : float
        Bandwidth of the exponential ranking kernel.  Smaller values impose
        a sharper penalty on concordance violations.

    Returns
    -------
    Tensor
        Scalar combined loss.
    """
    batch, num_events, time_bins = logits.shape

    # Normalise over the joint (event × time) space → joint PMF
    joint = F.softmax(logits.view(batch, -1), dim=-1).view(
        batch, num_events, time_bins
    )
    cif = torch.cumsum(joint, dim=2)  # (batch, num_events, time_bins)

    ll = _log_likelihood(joint, cif, event_times, event_types, time_bins)
    rank = _ranking_loss(cif, event_times, event_types, sigma)

    return (1.0 - alpha) * ll + alpha * rank


# ---------------------------------------------------------------------------
# Log-likelihood term
# ---------------------------------------------------------------------------

def _log_likelihood(
    joint: Tensor,
    cif: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    time_bins: int,
) -> Tensor:
    """Negative log-likelihood term of the DeepHit loss.

    For uncensored subjects (event_type > 0):
        loss_i = −log p(T = t_i, K = k_i)

    For censored subjects (event_type == 0):
        loss_i = −log S(t_i) = −log [1 − Σ_k CIF_k(t_i)]

    Parameters
    ----------
    joint : Tensor
        Joint PMF ``(batch, num_events, time_bins)`` — already softmaxed.
    cif : Tensor
        Cumulative incidence functions ``(batch, num_events, time_bins)``.
    event_times : Tensor
        Observed times ``(batch,)``, dtype long.
    event_types : Tensor
        Event indicators ``(batch,)``, dtype long.  0 = censored.
    time_bins : int
        Number of discrete time bins.

    Returns
    -------
    Tensor
        Scalar mean negative log-likelihood.
    """
    eps = 1e-8
    batch = event_times.shape[0]
    t_idx = event_times.clamp(0, time_bins - 1)

    nll = torch.zeros(batch, device=joint.device, dtype=joint.dtype)

    # --- Uncensored subjects ---
    uncensored = event_types > 0
    if uncensored.any():
        # event_types are 1-indexed; convert to 0-indexed for tensor indexing
        k_idx = (event_types[uncensored] - 1).clamp(0)
        arange = torch.arange(uncensored.sum(), device=joint.device)
        p = joint[uncensored][arange, k_idx, t_idx[uncensored]]
        nll[uncensored] = -torch.log(p + eps)

    # --- Censored subjects ---
    censored = ~uncensored
    if censored.any():
        # Survival at censoring time = 1 − (sum of all CIFs at t_c)
        total_cif = cif[censored].sum(dim=1)  # (n_censored, time_bins)
        t_c = t_idx[censored]
        arange_c = torch.arange(censored.sum(), device=joint.device)
        survival = 1.0 - total_cif[arange_c, t_c]
        nll[censored] = -torch.log(survival.clamp(min=eps))

    return nll.mean()


# ---------------------------------------------------------------------------
# Ranking loss term (vectorised)
# ---------------------------------------------------------------------------

def _ranking_loss(
    cif: Tensor,
    event_times: Tensor,
    event_types: Tensor,
    sigma: float,
) -> Tensor:
    """Cause-specific concordance ranking loss of DeepHit (vectorised).

    For each pair ``(i, j)`` where subject *i* experienced event *k* before
    subject *j* (i.e. ``t_i < t_j`` and ``event_type_i == k``), the ranking
    penalty is::

        φ(CIF_k(t_i | x_i) − CIF_k(t_i | x_j)) = exp(−Δ / σ)

    A well-calibrated model should have ``CIF_k(t_i | x_i) > CIF_k(t_i | x_j)``,
    making ``Δ > 0`` and the penalty small.

    The loss is fully vectorised: no Python loops over individual subjects.

    Parameters
    ----------
    cif : Tensor
        Cumulative incidence functions ``(batch, num_events, time_bins)``.
    event_times : Tensor
        Observed times ``(batch,)``, dtype long.
    event_types : Tensor
        Event indicators ``(batch,)``, dtype long.  0 = censored.
    sigma : float
        Exponential kernel bandwidth (> 0).

    Returns
    -------
    Tensor
        Scalar ranking loss, averaged over valid pairs.
    """
    batch, num_events, time_bins = cif.shape
    sigma = max(sigma, 1e-8)

    total_loss = cif.new_zeros(1).squeeze()
    total_pairs = 0

    for k in range(1, num_events + 1):
        mask_k = event_types == k  # subjects with cause-k event
        n_k = int(mask_k.sum())
        if n_k < 2:
            continue

        t_k = event_times[mask_k].clamp(0, time_bins - 1)  # (n_k,)
        cif_k = cif[mask_k, k - 1, :]                      # (n_k, time_bins)

        # CIF_k(t_i | x_i): scalar per subject i, evaluated at their own event time
        cif_i = cif_k[torch.arange(n_k, device=cif.device), t_k]  # (n_k,)

        # CIF_k(t_i | x_j): for each j, evaluate at each i's event time
        # Result shape: (n_k_i, n_k_j) — rows = i, cols = j
        cif_j_at_ti = cif_k[:, t_k].T  # (n_k, time_bins)[T] → (n_k_i, n_k_j)?
        # cif_k: (n_k, time_bins); t_k: (n_k,)
        # cif_k[:, t_k] → (n_k_j, n_k_i); transpose → (n_k_i, n_k_j)
        cif_j_at_ti = cif_k[:, t_k].permute(1, 0)  # (n_k_i, n_k_j)

        # Valid pairs: (t_j > t_i) — j experienced event AFTER i
        valid = (t_k.unsqueeze(0) > t_k.unsqueeze(1)).float()  # (n_k_i, n_k_j)

        # Ranking penalty: φ(CIF_i(t_i) − CIF_j(t_i))
        delta = cif_i.unsqueeze(1) - cif_j_at_ti  # (n_k_i, n_k_j)
        phi = torch.exp(-delta / sigma)

        pair_losses = valid * phi
        total_loss = total_loss + pair_losses.sum()
        total_pairs += int(valid.sum().item())

    return total_loss / max(total_pairs, 1)


# ---------------------------------------------------------------------------
# Cause-specific hazard NLL
# ---------------------------------------------------------------------------

def cause_specific_loss(
    hazards: Tensor,
    event_times: Tensor,
    event_types: Tensor,
) -> Tensor:
    """Negative log-likelihood for the cause-specific hazard model.

    For uncensored subject *i* (event K=k at time T=t)::

        loss_i = −log h_k(t) − log S(t − 1)

    where ``S(t) = Π_{s≤t} [1 − Σ_k h_k(s)]`` is the overall survival.

    For censored subject *i* (censored at time T=t_c)::

        loss_i = −log S(t_c)

    Parameters
    ----------
    hazards : Tensor
        Sub-hazard tensor of shape ``(batch, num_events, time_bins)`` with
        values in ``(0, 1)`` and ``hazards.sum(dim=1) < 1`` at each time bin.
        Typically the output of
        :meth:`~capa.model.survival.CauseSpecificHazardHead.forward`.
    event_times : Tensor
        Observed event or censoring time indices ``(batch,)``, dtype long.
    event_types : Tensor
        Event-type indicator ``(batch,)``, dtype long.  0 = censored.

    Returns
    -------
    Tensor
        Scalar mean negative log-likelihood.
    """
    batch, num_events, time_bins = hazards.shape
    eps = 1e-8

    t_idx = event_times.clamp(0, time_bins - 1)

    # Overall "no-event" probability at each time step: (batch, time_bins)
    no_event = (1.0 - hazards.sum(dim=1)).clamp(min=eps)

    # Log-survival: S(t) = Π_{s≤t} no_event(s)
    # Cumulative sum of logs gives log S(t): (batch, time_bins)
    log_S = torch.cumsum(torch.log(no_event), dim=1)

    # log S(t − 1): prepend 0 (log 1 = 0), drop last column
    log_S_prev = torch.cat(
        [torch.zeros(batch, 1, device=hazards.device, dtype=hazards.dtype),
         log_S[:, :-1]],
        dim=1,
    )  # (batch, time_bins)

    nll = torch.zeros(batch, device=hazards.device, dtype=hazards.dtype)

    # --- Uncensored subjects ---
    uncensored = event_types > 0
    if uncensored.any():
        k_idx = (event_types[uncensored] - 1).clamp(0)
        arange = torch.arange(uncensored.sum(), device=hazards.device)
        h_at_t = hazards[uncensored][arange, k_idx, t_idx[uncensored]]
        log_s_prev = log_S_prev[uncensored, t_idx[uncensored]]
        nll[uncensored] = -torch.log(h_at_t + eps) - log_s_prev

    # --- Censored subjects ---
    censored = ~uncensored
    if censored.any():
        nll[censored] = -log_S[censored, t_idx[censored]]

    return nll.mean()
