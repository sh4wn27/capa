"""Tests for IsotonicCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from capa.training.calibrate import IsotonicCalibrator


def _synthetic_data(
    n: int = 80,
    num_events: int = 3,
    time_bins: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # CIF: monotone non-decreasing noise
    cif = rng.random((n, num_events, time_bins)).cumsum(axis=2)
    cif /= cif[:, :, -1:] + 1e-6  # normalise to [0, 1)
    cif = cif.astype(np.float32)
    times = rng.integers(0, time_bins, size=n).astype(np.intp)
    types = rng.integers(0, num_events + 1, size=n).astype(np.intp)  # 0=censored
    return cif, times, types


class TestIsotonicCalibrator:
    def test_fit_returns_self(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        result = cal.fit(cif, times, types)
        assert result is cal

    def test_is_fitted_after_fit(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        assert not cal.is_fitted
        cal.fit(cif, times, types)
        assert cal.is_fitted

    def test_transform_shape_preserved(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        cal.fit(cif, times, types)
        out = cal.transform(cif)
        assert out.shape == cif.shape

    def test_output_clipped_to_unit_interval(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        cal.fit(cif, times, types)
        out = cal.transform(cif)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_monotone_along_time_axis(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        cal.fit(cif, times, types)
        out = cal.transform(cif)
        # Each (sample, event) curve must be non-decreasing
        diffs = np.diff(out, axis=2)
        assert (diffs >= -1e-9).all(), "Calibrated CIF is not monotone"

    def test_transform_before_fit_raises(self) -> None:
        cal = IsotonicCalibrator()
        cif, *_ = _synthetic_data()
        with pytest.raises(RuntimeError, match="fit"):
            cal.transform(cif)

    def test_fit_transform_equivalence(self) -> None:
        cif, times, types = _synthetic_data(seed=1)
        cal1 = IsotonicCalibrator(num_events=3, time_bins=20)
        cal2 = IsotonicCalibrator(num_events=3, time_bins=20)
        out_ft = cal1.fit_transform(cif, times, types)
        cal2.fit(cif, times, types)
        out_sep = cal2.transform(cif)
        np.testing.assert_array_equal(out_ft, out_sep)

    def test_shape_mismatch_raises(self) -> None:
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif_bad = np.zeros((10, 2, 20), dtype=np.float32)  # 2 events, expect 3
        times = np.zeros(10, dtype=np.intp)
        types = np.zeros(10, dtype=np.intp)
        with pytest.raises(ValueError, match="does not match"):
            cal.fit(cif_bad, times, types)

    def test_save_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cal = IsotonicCalibrator(num_events=3, time_bins=20)
        cif, times, types = _synthetic_data()
        cal.fit(cif, times, types)

        path = tmp_path / "cal.pkl"
        cal.save(path)
        cal2 = IsotonicCalibrator.load(path)

        out1 = cal.transform(cif)
        out2 = cal2.transform(cif)
        np.testing.assert_array_equal(out1, out2)
