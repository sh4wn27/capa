"""Evaluation metrics for CAPA competing-risks survival models.

Public API
----------
``concordance_index``
    Vectorised Harrell C-index for one event type.

``brier_score``
    Cause-specific Brier score at a single time point (IPCW-weighted).

``integrated_brier_score``
    IBS — trapezoidal integral of the Brier score over a time grid.

``calibration_curve``
    Calibration of predicted CIF against observed cumulative incidence at one
    time point (quantile-binned).

``evaluate_all``
    Master function: runs C-index, Brier scores, IBS, and calibration for
    every event type and returns a nested ``EvaluationResult`` object.

``bootstrap_ci``
    Generic bootstrap wrapper that returns (estimate, lower, upper) over 1000
    re-samples for any scalar metric function.

IPCW
----
All Brier-score variants use inverse-probability-of-censoring weighting
(IPCW, Gerds & Schumacher 2006) to handle right-censored observations
correctly.  The censoring distribution ``G(t)`` is estimated via the
Kaplan-Meier estimator on the censoring indicator (event=0).

References
----------
* Harrell et al. (1982) — C-index.
* Graf et al. (1999) — Brier score for survival data.
* Gerds & Schumacher (2006) — IPCW Brier score.
* Blanche et al. (2013) — time-dependent AUC and Brier for competing risks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Type aliases
F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MetricWithCI:
    """A scalar metric with a bootstrap confidence interval.

    Attributes
    ----------
    value : float
        Point estimate.
    ci_lower : float
        Lower bound of the ``(1 − alpha) × 100 %`` bootstrap CI.
    ci_upper : float
        Upper bound.
    ci_level : float
        Confidence level, e.g. ``0.95``.
    n_bootstrap : int
        Number of bootstrap iterations used.
    """

    value: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 1000

    def __repr__(self) -> str:
        pct = int(self.ci_level * 100)
        return (
            f"{self.value:.4f} "
            f"[{pct}% CI {self.ci_lower:.4f}–{self.ci_upper:.4f}]"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
        }


@dataclass
class CalibrationResult:
    """Calibration of predicted CIF vs observed cumulative incidence.

    Attributes
    ----------
    eval_time : float
        Time point at which calibration was assessed.
    predicted_mean : list[float]
        Mean predicted CIF in each quantile bin.
    observed_mean : list[float]
        Kaplan-Meier–based observed cumulative incidence in each bin.
    n_per_bin : list[int]
        Number of subjects per bin.
    """

    eval_time: float
    predicted_mean: list[float]
    observed_mean: list[float]
    n_per_bin: list[int]

    def to_dict(self) -> dict:
        return {
            "eval_time": self.eval_time,
            "predicted_mean": self.predicted_mean,
            "observed_mean": self.observed_mean,
            "n_per_bin": self.n_per_bin,
        }


@dataclass
class EventMetrics:
    """All metrics for a single competing event.

    Attributes
    ----------
    event_name : str
    cindex : MetricWithCI
    brier_scores : dict[float, MetricWithCI]
        Keyed by evaluation time (float).
    ibs : MetricWithCI
        Integrated Brier Score.
    calibration : list[CalibrationResult]
        One entry per evaluation time.
    """

    event_name: str
    cindex: MetricWithCI
    brier_scores: dict[float, MetricWithCI] = field(default_factory=dict)
    ibs: MetricWithCI | None = None
    calibration: list[CalibrationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "event_name": self.event_name,
            "cindex": self.cindex.to_dict(),
            "brier_scores": {
                str(t): m.to_dict() for t, m in self.brier_scores.items()
            },
            "calibration": [c.to_dict() for c in self.calibration],
        }
        if self.ibs is not None:
            d["ibs"] = self.ibs.to_dict()
        return d


@dataclass
class EvaluationResult:
    """Full evaluation result across all competing events.

    Attributes
    ----------
    events : list[EventMetrics]
    n_subjects : int
    eval_times : list[float]
    """

    events: list[EventMetrics]
    n_subjects: int
    eval_times: list[float]

    def to_dict(self) -> dict:
        return {
            "n_subjects": self.n_subjects,
            "eval_times": self.eval_times,
            "events": {e.event_name: e.to_dict() for e in self.events},
        }


# ---------------------------------------------------------------------------
# Kaplan-Meier helper (used for IPCW and calibration)
# ---------------------------------------------------------------------------

def _kaplan_meier(
    times: F64,
    events: Bool,
) -> tuple[F64, F64]:
    """Estimate the Kaplan-Meier survival function.

    Parameters
    ----------
    times : F64, shape (n,)
        Observed times.
    events : Bool, shape (n,)
        ``True`` if the event of interest occurred.

    Returns
    -------
    unique_times : F64
        Unique event times in ascending order.
    survival : F64
        KM survival estimate at each unique time (same length).
    """
    order = np.argsort(times)
    times_s = times[order]
    events_s = events[order]

    unique_times: list[float] = []
    survival_vals: list[float] = []
    S = 1.0
    n = len(times)

    for i, (t, e) in enumerate(zip(times_s, events_s)):
        if e:
            n_at_risk = n - i
            S *= 1.0 - 1.0 / n_at_risk
            unique_times.append(float(t))
            survival_vals.append(S)

    if not unique_times:
        return np.array([0.0]), np.array([1.0])
    return np.array(unique_times), np.array(survival_vals)


def _km_predict(
    km_times: F64,
    km_survival: F64,
    t: float,
) -> float:
    """Look up S(t) from a pre-computed KM curve (step function, left-continuous)."""
    if t < km_times[0]:
        return 1.0
    idx = np.searchsorted(km_times, t, side="right") - 1
    return float(km_survival[idx])


# ---------------------------------------------------------------------------
# Core metric functions (operate on numpy arrays, no torch)
# ---------------------------------------------------------------------------

def concordance_index(
    event_times: F64,
    predicted_risks: F64,
    event_observed: Bool,
) -> float:
    """Vectorised Harrell C-index for one event type.

    Parameters
    ----------
    event_times : F64, shape (n,)
        Observed event/censoring times.
    predicted_risks : F64, shape (n,)
        Predicted risk scores (higher = more risk).
    event_observed : Bool, shape (n,)
        ``True`` for subjects with the event of interest.

    Returns
    -------
    float
        C-index in ``[0, 1]``.  ``0.5`` = random; ``1.0`` = perfect.
        Returns ``nan`` if no valid comparable pairs exist.
    """
    event_observed = np.asarray(event_observed, dtype=bool)
    if event_observed.sum() == 0:
        return float("nan")

    # Consider only subjects with the event as "index" subjects
    mask = event_observed
    t_i = event_times[mask]        # (n_events,)
    r_i = predicted_risks[mask]    # (n_events,)
    t_all = event_times            # (n,)
    r_all = predicted_risks        # (n,)

    # For each index subject i, compare with all j where t_j > t_i
    # Vectorised: (n_events, n) indicator matrix
    valid = t_all[None, :] > t_i[:, None]               # (n_events, n)
    delta_r = r_i[:, None] - r_all[None, :]             # (n_events, n)

    concordant = float((valid & (delta_r > 0)).sum())
    discordant = float((valid & (delta_r < 0)).sum())
    tied       = float((valid & (delta_r == 0)).sum())
    total = concordant + discordant + tied

    if total == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / total


def _ipcw_weights(
    eval_time: float,
    event_times: F64,
    event_observed: Bool,
) -> F64:
    """Compute IPCW weights for the Brier score at *eval_time*.

    Subjects censored before eval_time receive weight 0.
    Subjects with event or censoring after eval_time receive weight
    ``1 / G(eval_time)``.  Subjects with event at or before eval_time receive
    weight ``1 / G(t_i)``.

    G is estimated by running KM on the censoring indicator (event=0).
    """
    censored = ~event_observed
    km_t, km_g = _kaplan_meier(event_times, censored)
    n = len(event_times)
    weights = np.zeros(n, dtype=np.float64)

    G_tau = _km_predict(km_t, km_g, eval_time)
    if G_tau < 1e-8:
        G_tau = 1e-8  # avoid division by zero

    for i in range(n):
        t_i = float(event_times[i])
        e_i = bool(event_observed[i])
        if e_i and t_i <= eval_time:
            G_ti = _km_predict(km_t, km_g, t_i)
            if G_ti < 1e-8:
                G_ti = 1e-8
            weights[i] = 1.0 / G_ti
        elif t_i > eval_time:
            weights[i] = 1.0 / G_tau
        # else: censored before eval_time → weight stays 0

    return weights


def brier_score(
    cif: F64,
    event_times: F64,
    event_observed: Bool,
    eval_time: float,
    time_bins: F64,
) -> float:
    """IPCW Brier score for one competing event at one time point.

    Parameters
    ----------
    cif : F64, shape (n, T)
        Predicted cumulative incidence function.
    event_times : F64, shape (n,)
        Observed times (bin indices or real time — must match *time_bins*).
    event_observed : Bool, shape (n,)
        ``True`` for subjects with this event.
    eval_time : float
        Time point at which to evaluate.
    time_bins : F64, shape (T,)
        Grid of time bin values (used to find the column of *cif*).

    Returns
    -------
    float
        IPCW Brier score (lower is better, 0 = perfect).
    """
    t_idx = int(np.searchsorted(time_bins, eval_time))
    t_idx = min(t_idx, cif.shape[1] - 1)
    predicted = cif[:, t_idx].astype(np.float64)

    outcome = ((event_times <= eval_time) & event_observed).astype(np.float64)
    weights = _ipcw_weights(eval_time, event_times, event_observed)

    n = len(event_times)
    weighted_sq = weights * (predicted - outcome) ** 2
    return float(weighted_sq.sum() / n)


def integrated_brier_score(
    cif: F64,
    event_times: F64,
    event_observed: Bool,
    eval_times: F64,
    time_bins: F64,
) -> float:
    """Trapezoidal IBS — integral of the Brier score over *eval_times*.

    Parameters
    ----------
    cif : F64, shape (n, T)
    event_times : F64, shape (n,)
    event_observed : Bool, shape (n,)
    eval_times : F64, shape (K,)
        Time points over which to integrate; must be sorted ascending.
    time_bins : F64, shape (T,)

    Returns
    -------
    float
        IBS — lower is better.
    """
    bs_vals = np.array([
        brier_score(cif, event_times, event_observed, float(t), time_bins)
        for t in eval_times
    ])
    time_range = float(eval_times[-1] - eval_times[0])
    if time_range <= 0:
        return float(bs_vals.mean())
    return float(np.trapezoid(bs_vals, eval_times) / time_range)


def calibration_curve(
    cif: F64,
    event_times: F64,
    event_observed: Bool,
    eval_time: float,
    time_bins: F64,
    n_bins: int = 10,
) -> CalibrationResult:
    """Calibration of predicted CIF versus observed cumulative incidence.

    Subjects are grouped into *n_bins* quantile bins of predicted CIF.
    The observed cumulative incidence in each bin is estimated as the
    fraction of subjects with an event by *eval_time* (naive; for small
    datasets censoring is limited).

    Parameters
    ----------
    cif : F64, shape (n, T)
    event_times : F64, shape (n,)
    event_observed : Bool, shape (n,)
    eval_time : float
    time_bins : F64, shape (T,)
    n_bins : int
        Number of quantile groups.

    Returns
    -------
    CalibrationResult
    """
    t_idx = int(np.searchsorted(time_bins, eval_time))
    t_idx = min(t_idx, cif.shape[1] - 1)
    predicted = cif[:, t_idx].astype(np.float64)
    outcome = ((event_times <= eval_time) & event_observed).astype(np.float64)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(predicted, quantiles)
    # Collapse duplicate edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([predicted.min() - 1e-9, predicted.max() + 1e-9])

    pred_means: list[float] = []
    obs_means: list[float] = []
    ns: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted >= lo) & (predicted <= hi)
        if mask.sum() == 0:
            continue
        pred_means.append(float(predicted[mask].mean()))
        obs_means.append(float(outcome[mask].mean()))
        ns.append(int(mask.sum()))

    return CalibrationResult(
        eval_time=eval_time,
        predicted_mean=pred_means,
        observed_mean=obs_means,
        n_per_bin=ns,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable[..., float],
    *args: npt.ArrayLike,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> MetricWithCI:
    """Bootstrap confidence interval for any scalar metric function.

    Computes the point estimate on the full data and then re-samples
    ``n_bootstrap`` times (with replacement, stratified by nothing) to build
    a percentile CI.

    Parameters
    ----------
    metric_fn : callable
        Function ``f(*args) -> float``.  All positional arrays in *args* are
        jointly re-sampled along their first axis.
    *args : array-like
        Arrays passed to *metric_fn*.  Each must have the same first dimension
        (n subjects).
    n_bootstrap : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence level, e.g. ``0.95``.
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    MetricWithCI
        Point estimate plus bootstrap CI.
    """
    arrays = [np.asarray(a) for a in args]
    n = arrays[0].shape[0]

    # Point estimate on full data
    value = metric_fn(*arrays)

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_args = [a[idx] if a.ndim == 1 else a[idx] for a in arrays]
        try:
            v = metric_fn(*boot_args)
        except Exception:
            v = float("nan")
        replicates.append(v)

    reps = np.array(replicates)
    valid = reps[~np.isnan(reps)]
    alpha = 1.0 - ci_level
    if len(valid) < 10:
        lo = hi = float("nan")
    else:
        lo = float(np.percentile(valid, 100 * alpha / 2))
        hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return MetricWithCI(
        value=float(value),
        ci_lower=lo,
        ci_upper=hi,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Master evaluation function
# ---------------------------------------------------------------------------

def evaluate_all(
    cif: F64,
    event_times: F64,
    event_types: npt.NDArray[np.int64],
    event_names: list[str],
    time_bins: F64,
    eval_times: F64 | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    n_calibration_bins: int = 10,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate a competing-risks model across all event types.

    Parameters
    ----------
    cif : F64, shape (n, num_events, T)
        Predicted cumulative incidence functions.
    event_times : F64, shape (n,)
        Observed event/censoring times (bin indices or real time).
    event_types : int array, shape (n,)
        Event-type indicator.  0 = censored; 1..K = event type.
    event_names : list[str]
        Names for events 1..K (length = num_events).
    time_bins : F64, shape (T,)
        Time grid for the CIF columns.
    eval_times : F64 | None
        Time points for Brier score and calibration.  Defaults to
        ``[25th, 50th, 75th]`` percentiles of uncensored event times.
    n_bootstrap : int
        Bootstrap iterations (set to 0 to skip CI computation).
    ci_level : float
        Bootstrap confidence level.
    n_calibration_bins : int
        Quantile bins for calibration curves.
    seed : int
        RNG seed for bootstrap reproducibility.

    Returns
    -------
    EvaluationResult
    """
    n, num_events, T = cif.shape
    assert len(event_names) == num_events, (
        f"len(event_names)={len(event_names)} must equal num_events={num_events}"
    )

    # Default eval times: quartiles of observed (uncensored) times
    if eval_times is None:
        uncensored_times = event_times[event_types > 0]
        if len(uncensored_times) == 0:
            uncensored_times = event_times
        eval_times = np.percentile(uncensored_times, [25, 50, 75]).astype(np.float64)
        eval_times = np.unique(eval_times)

    logger.info(
        "Evaluating %d subjects, %d events, eval_times=%s",
        n, num_events, eval_times.tolist(),
    )

    event_metrics_list: list[EventMetrics] = []

    for k, name in enumerate(event_names):
        k_type = k + 1          # event_types is 1-indexed
        cif_k = cif[:, k, :]   # (n, T)
        observed_k = (event_types == k_type).astype(bool)

        # ── C-index ───────────────────────────────────────────────
        # Risk score = CIF at the median eval time bin
        med_eval = eval_times[len(eval_times) // 2]
        med_bin = min(int(np.searchsorted(time_bins, med_eval)), T - 1)
        risks_k = cif_k[:, med_bin].astype(np.float64)

        if n_bootstrap > 0:
            cindex_metric = bootstrap_ci(
                concordance_index,
                event_times.astype(np.float64),
                risks_k,
                observed_k,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
        else:
            cindex_val = concordance_index(
                event_times.astype(np.float64), risks_k, observed_k
            )
            cindex_metric = MetricWithCI(
                value=cindex_val, ci_lower=float("nan"),
                ci_upper=float("nan"), ci_level=ci_level, n_bootstrap=0,
            )

        # ── Brier scores ──────────────────────────────────────────
        brier_scores_dict: dict[float, MetricWithCI] = {}
        for et in eval_times:
            et_f = float(et)

            def _bs(cif_arr: F64, times: F64, obs: Bool, _et: float = et_f) -> float:
                return brier_score(cif_arr, times, obs, _et, time_bins)

            if n_bootstrap > 0:
                bs_metric = bootstrap_ci(
                    _bs,
                    cif_k.astype(np.float64),
                    event_times.astype(np.float64),
                    observed_k,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=seed,
                )
            else:
                bs_val = _bs(cif_k, event_times.astype(np.float64), observed_k)
                bs_metric = MetricWithCI(
                    value=bs_val, ci_lower=float("nan"),
                    ci_upper=float("nan"), ci_level=ci_level, n_bootstrap=0,
                )
            brier_scores_dict[et_f] = bs_metric

        # ── Integrated Brier Score ────────────────────────────────
        def _ibs(cif_arr: F64, times: F64, obs: Bool) -> float:
            return integrated_brier_score(
                cif_arr, times, obs, eval_times.astype(np.float64), time_bins
            )

        if n_bootstrap > 0:
            ibs_metric = bootstrap_ci(
                _ibs,
                cif_k.astype(np.float64),
                event_times.astype(np.float64),
                observed_k,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed + k,
            )
        else:
            ibs_val = _ibs(cif_k, event_times.astype(np.float64), observed_k)
            ibs_metric = MetricWithCI(
                value=ibs_val, ci_lower=float("nan"),
                ci_upper=float("nan"), ci_level=ci_level, n_bootstrap=0,
            )

        # ── Calibration ───────────────────────────────────────────
        calib_list: list[CalibrationResult] = []
        for et in eval_times:
            calib = calibration_curve(
                cif_k.astype(np.float64),
                event_times.astype(np.float64),
                observed_k,
                float(et),
                time_bins,
                n_bins=n_calibration_bins,
            )
            calib_list.append(calib)

        event_metrics_list.append(EventMetrics(
            event_name=name,
            cindex=cindex_metric,
            brier_scores=brier_scores_dict,
            ibs=ibs_metric,
            calibration=calib_list,
        ))

        logger.info(
            "  %s: C-index=%s  IBS=%s",
            name, cindex_metric, ibs_metric,
        )

    return EvaluationResult(
        events=event_metrics_list,
        n_subjects=n,
        eval_times=eval_times.tolist(),
    )
