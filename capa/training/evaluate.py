"""Evaluation metrics: C-index, Brier score, calibration."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def concordance_index(
    event_times: npt.NDArray[np.float64],
    predicted_risks: npt.NDArray[np.float64],
    event_observed: npt.NDArray[np.bool_],
) -> float:
    """Compute the Harrell C-index for a single event type.

    Parameters
    ----------
    event_times : npt.NDArray[np.float64]
        Observed event/censoring times, shape ``(n,)``.
    predicted_risks : npt.NDArray[np.float64]
        Predicted risk scores (higher = more risk), shape ``(n,)``.
    event_observed : npt.NDArray[np.bool_]
        Boolean mask: ``True`` for uncensored subjects, shape ``(n,)``.

    Returns
    -------
    float
        C-index in ``[0, 1]``. 0.5 = random, 1.0 = perfect.
    """
    concordant = 0
    discordant = 0
    tied = 0

    n = len(event_times)
    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if event_times[j] <= event_times[i]:
                continue
            diff = predicted_risks[i] - predicted_risks[j]
            if diff > 0:
                concordant += 1
            elif diff < 0:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / total


def brier_score(
    cif: npt.NDArray[np.float64],
    event_times: npt.NDArray[np.float64],
    event_observed: npt.NDArray[np.bool_],
    eval_time: float,
    time_bins: npt.NDArray[np.float64],
) -> float:
    """Compute the Brier score for a single event at a given evaluation time.

    Parameters
    ----------
    cif : npt.NDArray[np.float64]
        Predicted cumulative incidence function, shape ``(n, time_bins)``.
    event_times : npt.NDArray[np.float64]
        Observed event/censoring times, shape ``(n,)``.
    event_observed : npt.NDArray[np.bool_]
        Boolean mask for uncensored subjects, shape ``(n,)``.
    eval_time : float
        Time point at which to evaluate the Brier score.
    time_bins : npt.NDArray[np.float64]
        Array of time bin midpoints, shape ``(time_bins,)``.

    Returns
    -------
    float
        Brier score (lower is better).
    """
    t_idx = int(np.searchsorted(time_bins, eval_time))
    t_idx = min(t_idx, cif.shape[1] - 1)
    predicted = cif[:, t_idx]

    # I(T_i <= t, event observed)
    outcome = (event_times <= eval_time) & event_observed

    # Naive Brier score (no IPCW weighting)
    return float(np.mean((predicted - outcome.astype(float)) ** 2))
