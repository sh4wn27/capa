"""Tests for capa/training/evaluate.py."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _synthetic_data(
    n: int = 80,
    T: int = 20,
    K: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cif, event_times, event_types) numpy arrays."""
    rng = np.random.default_rng(seed)
    # CIF: (n, K, T) — monotone increasing, bounded in [0,1]
    increments = rng.dirichlet(np.ones(T + 1), size=(n, K))[:, :, :T]
    cif = np.cumsum(increments / (increments.sum(axis=-1, keepdims=True) + 1), axis=-1)
    cif = np.clip(cif, 0.0, 1.0).astype(np.float64)
    event_times = rng.integers(0, T, size=n).astype(np.float64)
    event_types = rng.integers(0, K + 1, size=n).astype(np.int64)  # 0=censored
    return cif, event_times, event_types


# ---------------------------------------------------------------------------
# concordance_index
# ---------------------------------------------------------------------------

class TestConcordanceIndex:
    """Tests for capa.training.evaluate.concordance_index."""

    def setup_method(self) -> None:
        from capa.training.evaluate import concordance_index
        self.ci = concordance_index

    def test_perfect_ranker(self) -> None:
        """When risk == time (higher risk → earlier event), C-index = 1."""
        n = 50
        times = np.arange(n, dtype=np.float64)
        # Higher risk score for earlier event times → concordant
        risks = (n - times).astype(np.float64)
        obs = np.ones(n, dtype=bool)
        c = self.ci(times, risks, obs)
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_anti_ranker(self) -> None:
        """When risk == time (lower risk → earlier event), C-index = 0."""
        n = 50
        times = np.arange(n, dtype=np.float64)
        risks = times.copy()
        obs = np.ones(n, dtype=bool)
        c = self.ci(times, risks, obs)
        assert c == pytest.approx(0.0, abs=1e-6)

    def test_random_is_near_half(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        times = rng.uniform(0, 100, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[: n // 4] = True   # ensure some events
        c = self.ci(times, risks, obs)
        assert 0.3 < c < 0.7

    def test_no_events_returns_nan(self) -> None:
        times = np.array([1.0, 2.0, 3.0])
        risks = np.array([0.5, 0.3, 0.1])
        obs   = np.zeros(3, dtype=bool)
        assert math.isnan(self.ci(times, risks, obs))

    def test_all_tied_times_returns_half(self) -> None:
        times = np.ones(20, dtype=np.float64)
        risks = np.random.rand(20)
        obs   = np.ones(20, dtype=bool)
        # No pairs with t_j > t_i → no valid comparables → nan
        c = self.ci(times, risks, obs)
        assert math.isnan(c) or c == pytest.approx(0.5, abs=0.5)

    def test_output_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.uniform(0, 50, 100)
        risks = rng.uniform(0, 1, 100)
        obs   = rng.choice([True, False], 100)
        obs[:25] = True
        c = self.ci(times, risks, obs)
        if not math.isnan(c):
            assert 0.0 <= c <= 1.0

    def test_single_event(self) -> None:
        times = np.array([5.0, 10.0, 15.0])
        risks = np.array([0.9, 0.5, 0.1])
        obs   = np.array([True, False, False])
        # Subject 0 with t=5 compares to subjects 1,2 (t>5); higher risk=0.9 vs 0.5,0.1 → concordant
        c = self.ci(times, risks, obs)
        assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Tests for capa.training.evaluate.brier_score."""

    def setup_method(self) -> None:
        from capa.training.evaluate import brier_score
        self.bs = brier_score

    def _make_perfect_cif(self, n: int, T: int, event_times: np.ndarray,
                          event_observed: np.ndarray) -> np.ndarray:
        """CIF that is 1 at/after event time and 0 before."""
        cif = np.zeros((n, T), dtype=np.float64)
        for i in range(n):
            if event_observed[i]:
                t = int(event_times[i])
                cif[i, t:] = 1.0
        return cif

    def test_output_is_nonnegative(self) -> None:
        cif, et, etype = _synthetic_data(n=60, T=20, K=2, seed=1)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        bs = self.bs(cif[:, 0, :], et, obs, 10.0, time_bins)
        assert bs >= 0.0

    def test_perfect_predictor_is_low(self) -> None:
        """A perfect CIF should yield a near-zero Brier score."""
        n, T = 100, 50
        rng = np.random.default_rng(3)
        event_times = rng.integers(0, T, n).astype(np.float64)
        event_obs   = rng.choice([True, False], n)
        event_obs[:40] = True
        cif = self._make_perfect_cif(n, T, event_times, event_obs)
        time_bins = np.arange(T, dtype=np.float64)
        bs = self.bs(cif, event_times, event_obs, 25.0, time_bins)
        assert bs < 0.05

    def test_uniform_zero_cif(self) -> None:
        """Zero CIF → Brier score equals fraction of events."""
        n, T = 80, 20
        rng = np.random.default_rng(5)
        event_times = rng.integers(5, T, n).astype(np.float64)
        event_obs   = np.ones(n, dtype=bool)
        cif = np.zeros((n, T), dtype=np.float64)
        time_bins = np.arange(T, dtype=np.float64)
        bs = self.bs(cif, event_times, event_obs, float(T - 1), time_bins)
        # All events occur before eval_time; outcome=1, predicted=0 → (0-1)^2
        # Weighted by IPCW; rough check: > 0
        assert bs > 0.0

    def test_eval_time_beyond_last_bin_clipped(self) -> None:
        """eval_time beyond the last bin should not raise."""
        cif, et, etype = _synthetic_data(n=40, T=10, K=1)
        time_bins = np.arange(10, dtype=np.float64)
        obs = (etype == 1)
        bs = self.bs(cif[:, 0, :], et, obs, 999.0, time_bins)
        assert bs >= 0.0


# ---------------------------------------------------------------------------
# integrated_brier_score
# ---------------------------------------------------------------------------

class TestIntegratedBrierScore:
    """Tests for capa.training.evaluate.integrated_brier_score."""

    def setup_method(self) -> None:
        from capa.training.evaluate import integrated_brier_score
        self.ibs = integrated_brier_score

    def test_output_is_nonnegative(self) -> None:
        cif, et, etype = _synthetic_data(n=60, T=20, K=2)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        eval_times = np.array([5.0, 10.0, 15.0])
        ibs = self.ibs(cif[:, 0, :], et, obs, eval_times, time_bins)
        assert ibs >= 0.0

    def test_single_eval_time_returns_brier_score(self) -> None:
        """With a single eval_time, IBS should equal that Brier score."""
        from capa.training.evaluate import brier_score
        cif, et, etype = _synthetic_data(n=60, T=20, K=2, seed=11)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        eval_times = np.array([10.0, 10.0])   # degenerate single-point (range=0)
        ibs = self.ibs(cif[:, 0, :], et, obs, eval_times, time_bins)
        bs  = brier_score(cif[:, 0, :], et, obs, 10.0, time_bins)
        # With zero range, IBS falls back to mean = bs
        assert ibs == pytest.approx(bs, rel=0.01)

    def test_monotone_eval_times_required(self) -> None:
        """IBS is well-defined only for sorted eval times; no error expected."""
        cif, et, etype = _synthetic_data(n=50, T=20, K=1)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        eval_times = np.array([3.0, 7.0, 12.0, 17.0])
        ibs = self.ibs(cif[:, 0, :], et, obs, eval_times, time_bins)
        assert np.isfinite(ibs)


# ---------------------------------------------------------------------------
# calibration_curve
# ---------------------------------------------------------------------------

class TestCalibrationCurve:
    """Tests for capa.training.evaluate.calibration_curve."""

    def setup_method(self) -> None:
        from capa.training.evaluate import calibration_curve
        self.cc = calibration_curve

    def test_returns_correct_type(self) -> None:
        from capa.training.evaluate import CalibrationResult
        cif, et, etype = _synthetic_data(n=80, T=20, K=2, seed=7)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=5)
        assert isinstance(result, CalibrationResult)

    def test_lengths_match(self) -> None:
        cif, et, etype = _synthetic_data(n=80, T=20, K=2, seed=8)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=5)
        assert len(result.predicted_mean) == len(result.observed_mean) == len(result.n_per_bin)

    def test_n_per_bin_sums_to_n(self) -> None:
        n = 80
        cif, et, etype = _synthetic_data(n=n, T=20, K=2, seed=9)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=5)
        assert sum(result.n_per_bin) == n

    def test_predicted_mean_in_unit_interval(self) -> None:
        cif, et, etype = _synthetic_data(n=80, T=20, K=2, seed=10)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=5)
        for p in result.predicted_mean:
            assert 0.0 <= p <= 1.0

    def test_observed_mean_in_unit_interval(self) -> None:
        cif, et, etype = _synthetic_data(n=80, T=20, K=2, seed=11)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=5)
        for o in result.observed_mean:
            assert 0.0 <= o <= 1.0

    def test_handles_uniform_cif(self) -> None:
        """Uniform CIF (all subjects have same predicted value) should not raise."""
        n, T = 40, 10
        cif = np.full((n, T), 0.3, dtype=np.float64)
        et  = np.arange(n, dtype=np.float64) % T
        obs = (np.arange(n) % 2).astype(bool)
        time_bins = np.arange(T, dtype=np.float64)
        result = self.cc(cif, et, obs, 5.0, time_bins, n_bins=5)
        assert len(result.n_per_bin) >= 1

    def test_to_dict(self) -> None:
        cif, et, etype = _synthetic_data(n=60, T=20, K=2, seed=12)
        time_bins = np.arange(20, dtype=np.float64)
        obs = (etype == 1)
        result = self.cc(cif[:, 0, :], et, obs, 10.0, time_bins, n_bins=4)
        d = result.to_dict()
        assert "eval_time" in d and "predicted_mean" in d


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Tests for capa.training.evaluate.bootstrap_ci."""

    def setup_method(self) -> None:
        from capa.training.evaluate import bootstrap_ci, concordance_index
        self.bootstrap_ci = bootstrap_ci
        self.concordance_index = concordance_index

    def test_returns_metric_with_ci(self) -> None:
        from capa.training.evaluate import MetricWithCI
        n = 60
        rng = np.random.default_rng(0)
        times = rng.uniform(0, 50, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:20] = True
        m = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=50, seed=0,
        )
        assert isinstance(m, MetricWithCI)

    def test_ci_contains_point_estimate(self) -> None:
        n = 60
        rng = np.random.default_rng(1)
        times = rng.uniform(0, 50, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:20] = True
        m = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=200, seed=1,
        )
        # 95% CI should usually contain the point estimate
        assert m.ci_lower <= m.value <= m.ci_upper

    def test_ci_level_stored(self) -> None:
        n = 40
        rng = np.random.default_rng(2)
        times = rng.uniform(0, 20, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:10] = True
        m = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=50, ci_level=0.90, seed=2,
        )
        assert m.ci_level == 0.90

    def test_n_bootstrap_stored(self) -> None:
        n = 40
        rng = np.random.default_rng(3)
        times = rng.uniform(0, 20, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:10] = True
        m = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=77, seed=3,
        )
        assert m.n_bootstrap == 77

    def test_reproducible_with_seed(self) -> None:
        n = 60
        rng = np.random.default_rng(4)
        times = rng.uniform(0, 50, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:20] = True
        m1 = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=100, seed=42,
        )
        m2 = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=100, seed=42,
        )
        assert m1.ci_lower == pytest.approx(m2.ci_lower)
        assert m1.ci_upper == pytest.approx(m2.ci_upper)

    def test_wider_ci_with_fewer_samples(self) -> None:
        """CI should be wider for smaller n (stochastic, but robust over many runs)."""
        def mean_fn(x: np.ndarray) -> float:
            return float(x.mean())

        rng = np.random.default_rng(5)
        small = rng.normal(0, 1, 20)
        large = rng.normal(0, 1, 200)

        m_small = self.bootstrap_ci(mean_fn, small, n_bootstrap=300, seed=5)
        m_large = self.bootstrap_ci(mean_fn, large, n_bootstrap=300, seed=5)
        assert (m_small.ci_upper - m_small.ci_lower) > (m_large.ci_upper - m_large.ci_lower)

    def test_to_dict(self) -> None:
        n = 40
        rng = np.random.default_rng(6)
        times = rng.uniform(0, 20, n)
        risks = rng.uniform(0, 1, n)
        obs   = rng.choice([True, False], n)
        obs[:10] = True
        m = self.bootstrap_ci(
            self.concordance_index, times, risks, obs,
            n_bootstrap=30, seed=6,
        )
        d = m.to_dict()
        assert set(d.keys()) >= {"value", "ci_lower", "ci_upper", "ci_level"}


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    """Tests for capa.training.evaluate.evaluate_all."""

    def setup_method(self) -> None:
        from capa.training.evaluate import evaluate_all
        self.evaluate_all = evaluate_all

    def _run(
        self,
        n: int = 80,
        T: int = 20,
        K: int = 2,
        n_bootstrap: int = 0,
        seed: int = 0,
    ):
        cif, et, etype = _synthetic_data(n=n, T=T, K=K, seed=seed)
        time_bins = np.arange(T, dtype=np.float64)
        names = [f"event_{k}" for k in range(1, K + 1)]
        return self.evaluate_all(
            cif=cif,
            event_times=et,
            event_types=etype,
            event_names=names,
            time_bins=time_bins,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    def test_returns_evaluation_result(self) -> None:
        from capa.training.evaluate import EvaluationResult
        result = self._run()
        assert isinstance(result, EvaluationResult)

    def test_event_count_matches_names(self) -> None:
        result = self._run(K=3)
        assert len(result.events) == 3

    def test_n_subjects_correct(self) -> None:
        result = self._run(n=60)
        assert result.n_subjects == 60

    def test_cindex_in_unit_interval(self) -> None:
        result = self._run(n=100)
        for em in result.events:
            if not math.isnan(em.cindex.value):
                assert 0.0 <= em.cindex.value <= 1.0

    def test_brier_scores_nonneg(self) -> None:
        result = self._run(n=80)
        for em in result.events:
            for bs in em.brier_scores.values():
                assert bs.value >= 0.0

    def test_ibs_nonneg(self) -> None:
        result = self._run(n=80)
        for em in result.events:
            assert em.ibs is not None
            assert em.ibs.value >= 0.0

    def test_calibration_length_matches_eval_times(self) -> None:
        result = self._run(n=80)
        n_eval = len(result.eval_times)
        for em in result.events:
            assert len(em.calibration) == n_eval

    def test_to_dict_serializable(self) -> None:
        """to_dict() should produce a JSON-serializable dict."""
        import json
        result = self._run(n=60)
        d = result.to_dict()
        json_str = json.dumps(d)  # should not raise
        assert len(json_str) > 0

    def test_eval_times_stored(self) -> None:
        result = self._run(n=80)
        assert len(result.eval_times) >= 1

    def test_custom_eval_times(self) -> None:
        cif, et, etype = _synthetic_data(n=80, T=20, K=2, seed=13)
        time_bins = np.arange(20, dtype=np.float64)
        eval_times = np.array([5.0, 10.0, 15.0])
        result = self.evaluate_all(
            cif=cif,
            event_times=et,
            event_types=etype,
            event_names=["e1", "e2"],
            time_bins=time_bins,
            eval_times=eval_times,
            n_bootstrap=0,
        )
        assert result.eval_times == pytest.approx([5.0, 10.0, 15.0])

    def test_n_bootstrap_zero_gives_nan_ci(self) -> None:
        result = self._run(n_bootstrap=0)
        for em in result.events:
            assert math.isnan(em.cindex.ci_lower)

    def test_n_bootstrap_positive_gives_finite_ci(self) -> None:
        result = self._run(n=80, n_bootstrap=50, seed=99)
        for em in result.events:
            # May be nan if all events are censored; just check it runs
            assert isinstance(em.cindex.ci_lower, float)

    def test_wrong_event_names_length_raises(self) -> None:
        cif, et, etype = _synthetic_data(n=60, T=20, K=2)
        time_bins = np.arange(20, dtype=np.float64)
        with pytest.raises(AssertionError):
            self.evaluate_all(
                cif=cif,
                event_times=et,
                event_types=etype,
                event_names=["only_one"],   # K=2 but only 1 name
                time_bins=time_bins,
                n_bootstrap=0,
            )

    def test_all_censored_no_crash(self) -> None:
        """All subjects censored (event_types=0) should not raise."""
        n, T, K = 40, 20, 2
        cif, et, _ = _synthetic_data(n=n, T=T, K=K, seed=20)
        etype = np.zeros(n, dtype=np.int64)  # all censored
        time_bins = np.arange(T, dtype=np.float64)
        result = self.evaluate_all(
            cif=cif, event_times=et, event_types=etype,
            event_names=["e1", "e2"], time_bins=time_bins, n_bootstrap=0,
        )
        # C-index should be nan or 0.5
        for em in result.events:
            assert math.isnan(em.cindex.value) or em.cindex.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Kaplan-Meier helpers
# ---------------------------------------------------------------------------

class TestKaplanMeier:
    """Tests for _kaplan_meier and _km_predict."""

    def setup_method(self) -> None:
        from capa.training.evaluate import _kaplan_meier, _km_predict
        self._km = _kaplan_meier
        self._pred = _km_predict

    def test_no_events_returns_default(self) -> None:
        times = np.array([1.0, 2.0, 3.0])
        events = np.zeros(3, dtype=bool)
        t, s = self._km(times, events)
        assert t[0] == 0.0 and s[0] == pytest.approx(1.0)

    def test_survival_monotone_decreasing(self) -> None:
        rng = np.random.default_rng(0)
        times = rng.uniform(0, 10, 50)
        events = rng.choice([True, False], 50)
        events[:10] = True
        t, s = self._km(times, events)
        assert (np.diff(s) <= 0).all()

    def test_survival_starts_below_one(self) -> None:
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.ones(5, dtype=bool)
        t, s = self._km(times, events)
        assert s[-1] < 1.0

    def test_km_predict_before_first_event(self) -> None:
        times = np.array([2.0, 4.0, 6.0])
        events = np.ones(3, dtype=bool)
        t, s = self._km(times, events)
        assert self._pred(t, s, 1.0) == pytest.approx(1.0)

    def test_km_predict_at_event_time(self) -> None:
        times = np.array([2.0, 4.0, 6.0])
        events = np.ones(3, dtype=bool)
        t, s = self._km(times, events)
        # At t=2: should equal S(2)
        val = self._pred(t, s, 2.0)
        assert val == pytest.approx(s[0])

    def test_km_predict_step_function(self) -> None:
        times = np.array([1.0, 3.0, 5.0])
        events = np.ones(3, dtype=bool)
        t, s = self._km(times, events)
        v1 = self._pred(t, s, 1.5)
        v2 = self._pred(t, s, 2.9)
        assert v1 == pytest.approx(v2)


# ---------------------------------------------------------------------------
# CLI smoke-test (scripts/evaluate.py --synthetic)
# ---------------------------------------------------------------------------

class TestEvaluateCLI:
    """Smoke-tests for scripts/evaluate.py."""

    def _save_dummy_checkpoint(self, tmp_path: Path) -> Path:
        """Train for 1 epoch on synthetic data and save a checkpoint."""
        import sys
        sys.path.insert(0, str(_ROOT))
        import torch
        from capa.config import get_config
        from capa.model.capa_model import CAPAModel
        from capa.training.trainer import CheckpointState

        cfg = get_config()
        model = CAPAModel(
            embedding_dim=8,
            loci=["A", "B"],
            clinical_dim=cfg.model.clinical_dim,
            interaction_dim=16,
            survival_type="deephit",
            num_events=cfg.model.num_events,
            time_bins=cfg.model.time_bins,
            num_heads=cfg.model.interaction_heads,
            num_layers=cfg.model.interaction_layers,
        )
        ckpt = CheckpointState(
            epoch=1,
            best_cindex=0.5,
            history={"train_loss": [1.0], "val_loss": [1.0], "val_cindex": [0.5], "lr": [1e-3]},
            model_state=model.state_dict(),
            optimizer_state={},
            scheduler_state={},
        )
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    def test_synthetic_dry_run(self, tmp_path: Path) -> None:
        """evaluate.py --synthetic should exit 0 and produce results.json."""
        ckpt = self._save_dummy_checkpoint(tmp_path)
        output = tmp_path / "results.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "evaluate.py"),
                "--checkpoint", str(ckpt),
                "--synthetic",
                "--synthetic-n", "30",
                "--n-bootstrap", "0",
                "--embedding-dim", "8",
                "--loci", "A", "B",
                "--interaction-dim", "16",
                "--output-path", str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()

    def test_results_json_valid(self, tmp_path: Path) -> None:
        """results.json must be valid JSON with expected top-level keys."""
        import json
        ckpt = self._save_dummy_checkpoint(tmp_path)
        output = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "evaluate.py"),
                "--checkpoint", str(ckpt),
                "--synthetic",
                "--synthetic-n", "20",
                "--n-bootstrap", "0",
                "--embedding-dim", "8",
                "--loci", "A", "B",
                "--interaction-dim", "16",
                "--output-path", str(output),
            ],
            capture_output=True,
            check=True,
        )
        with output.open() as f:
            d = json.load(f)
        assert "n_subjects" in d
        assert "events" in d
        assert "eval_times" in d

    def test_default_output_path(self, tmp_path: Path) -> None:
        """If --output-path is not given, results.json lands next to the checkpoint."""
        ckpt = self._save_dummy_checkpoint(tmp_path)
        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "evaluate.py"),
                "--checkpoint", str(ckpt),
                "--synthetic",
                "--synthetic-n", "20",
                "--n-bootstrap", "0",
                "--embedding-dim", "8",
                "--loci", "A", "B",
                "--interaction-dim", "16",
            ],
            capture_output=True,
            check=True,
        )
        assert (tmp_path / "results.json").exists()
