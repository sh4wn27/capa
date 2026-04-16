"""Post-hoc isotonic calibration for competing-risks CIF curves.

Motivation
----------
Deep survival models are often poorly calibrated: the predicted CIF at time *t*
does not match the observed event rate at *t* on held-out data.  Isotonic
regression (Platt 1999; Niculescu-Mizil & Caruana 2005) provides a
distribution-free, shape-preserving recalibration that is monotone by
construction — important for cumulative incidence functions which must be
non-decreasing.

Design
------
A separate :class:`IsotonicCalibrator` is fitted *per event × time-point*.
For K events and M time bins, this produces K × M independent isotonic
regression models.  At calibration time the predicted CIFs are evaluated at the
same M bins and passed through the corresponding monotone mapping.

Limitation: with small validation sets (n ≈ 29 for the UCI BMT cohort) the
calibration estimates are noisy.  The calibrator is offered as an optional
post-processing step, not a mandatory component.

Usage
-----
::

    calibrator = IsotonicCalibrator()

    # Fit on held-out validation set
    # cif_val: (n_val, n_events, n_bins)   — model predictions
    # times_val: (n_val,)                  — integer bin indices (event times)
    # types_val: (n_val,)                  — event types (0=censored, 1..K)
    calibrator.fit(cif_val, times_val, types_val)

    # Transform new predictions
    # cif_test: (n_test, n_events, n_bins)
    cif_cal = calibrator.transform(cif_test)

    # Or fit + transform in one call
    cif_cal = calibrator.fit_transform(cif_val, times_val, types_val)

Serialisation
-------------
:class:`IsotonicCalibrator` is a plain Python object and can be saved with
``torch.save`` / ``pickle``.  A convenience ``save`` / ``load`` pair is
provided.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Type alias for CIF arrays
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


class IsotonicCalibrator:
    """Post-hoc isotonic calibration for DeepHit-style CIF outputs.

    One isotonic regressor is fitted per (event, time-bin) pair.  At
    inference, predicted CIFs are passed through the fitted maps and the
    resulting curves are re-sorted to guarantee monotonicity.

    Parameters
    ----------
    num_events : int
        Number of competing events (K).
    time_bins : int
        Number of discrete time bins (M).

    Attributes
    ----------
    is_fitted : bool
        ``True`` after :meth:`fit` has been called.
    """

    def __init__(self, num_events: int = 3, time_bins: int = 100) -> None:
        self._num_events = num_events
        self._time_bins = time_bins
        # _regressors[k][m] → fitted IsotonicRegression or None (not enough data)
        self._regressors: list[list[Any]] = [
            [None] * time_bins for _ in range(num_events)
        ]
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has been called."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        cif: F32 | F64 | npt.NDArray[Any],
        event_times: npt.NDArray[np.intp],
        event_types: npt.NDArray[np.intp],
    ) -> "IsotonicCalibrator":
        """Fit isotonic regressors on held-out validation predictions.

        Parameters
        ----------
        cif : array_like, shape (n, n_events, n_bins)
            Predicted cumulative incidence functions from the model.
        event_times : array_like, shape (n,)
            Observed event times as discrete bin indices in ``[0, n_bins)``.
        event_types : array_like, shape (n,)
            Event type labels: 0 = censored, 1..K = event k.

        Returns
        -------
        self
        """
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

        cif_arr = np.asarray(cif, dtype=np.float64)
        times   = np.asarray(event_times, dtype=np.intp)
        types   = np.asarray(event_types, dtype=np.intp)

        n, k_data, m_data = cif_arr.shape
        if k_data != self._num_events or m_data != self._time_bins:
            raise ValueError(
                f"CIF shape ({n}, {k_data}, {m_data}) does not match "
                f"calibrator ({self._num_events} events, {self._time_bins} bins)."
            )

        n_fitted = 0
        for k in range(self._num_events):
            event_idx = k + 1  # event types are 1-indexed
            observed = (types == event_idx).astype(np.float64)

            for m in range(self._time_bins):
                y_pred = cif_arr[:, k, m]           # predicted P(T ≤ t_m, event=k)
                y_obs  = (times <= m) * observed     # 1 if event k occurred by bin m

                # Need at least 5 observations with the target event
                if observed.sum() < 5 or len(np.unique(y_pred)) < 2:
                    self._regressors[k][m] = None
                    continue

                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(y_pred, y_obs)
                self._regressors[k][m] = ir
                n_fitted += 1

        self._is_fitted = True
        logger.info(
            "IsotonicCalibrator fitted: %d / %d (event × bin) pairs calibrated",
            n_fitted, self._num_events * self._time_bins,
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        cif: F32 | F64 | npt.NDArray[Any],
    ) -> F64:
        """Apply calibration to new CIF predictions.

        Time bins where no calibrator was fitted are passed through unchanged.
        After applying the per-bin maps the curves are sorted along the time
        axis to restore monotonicity (isotonic regression per bin does not
        guarantee the output is non-decreasing across bins).

        Parameters
        ----------
        cif : array_like, shape (n, n_events, n_bins)
            Raw model CIF predictions.

        Returns
        -------
        F64
            Calibrated CIF array of the same shape, values clipped to
            ``[0, 1]``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        cif_arr = np.asarray(cif, dtype=np.float64)
        out = cif_arr.copy()

        for k in range(self._num_events):
            for m in range(self._time_bins):
                ir = self._regressors[k][m]
                if ir is None:
                    continue
                out[:, k, m] = ir.predict(cif_arr[:, k, m])

        # Restore monotonicity along time axis: cummax
        for k in range(self._num_events):
            out[:, k, :] = np.maximum.accumulate(out[:, k, :], axis=1)

        return np.clip(out, 0.0, 1.0)

    def fit_transform(
        self,
        cif: F32 | F64 | npt.NDArray[Any],
        event_times: npt.NDArray[np.intp],
        event_types: npt.NDArray[np.intp],
    ) -> F64:
        """Fit then transform in one call (convenience wrapper).

        Parameters
        ----------
        cif : array_like, shape (n, n_events, n_bins)
            Predictions on the calibration set.
        event_times : array_like, shape (n,)
            Observed event times (bin indices).
        event_types : array_like, shape (n,)
            Event type labels (0=censored, 1..K=event).

        Returns
        -------
        F64
            Calibrated predictions for the same set.
        """
        return self.fit(cif, event_times, event_types).transform(cif)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the calibrator to *path*.

        Parameters
        ----------
        path : str or Path
            Destination file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("IsotonicCalibrator saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        """Load a calibrator previously saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file.

        Returns
        -------
        IsotonicCalibrator
            Restored calibrator.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected IsotonicCalibrator, got {type(obj)}")
        logger.info("IsotonicCalibrator loaded from %s", path)
        return obj
