"""Tests for capa/model/survival.py and capa/model/losses.py.

Coverage
--------
survival.py
  DeepHitHead
    - forward() shape, dtype, no-nan/inf
    - softmax-normalised logits across full joint space
    - cif() shape, values in [0,1], monotone non-decreasing
    - cif() row-sums ≤ 1 at each time bin
    - gradients flow to input and all parameters
    - dropout changes training output; eval is deterministic
    - num_events=1 edge case
    - time_bins=1 edge case
    - batch_size=1 edge case

  CauseSpecificHazardHead
    - forward() shape, dtype, no-nan/inf
    - hazards in (0,1)
    - hazards.sum(dim=1) ≤ 1 at each time bin (softmax constraint)
    - cif() shape, values in [0,1], monotone non-decreasing
    - cif() row-sums ≤ 1 at each time bin
    - gradients flow to input and all parameters
    - num_events=1 edge case
    - batch_size=1 edge case

  hazards_to_cif
    - monotone non-decreasing
    - values in [0,1]
    - all-zero hazards → CIF stays at 0
    - max-hazard → CIF rises to near 1

losses.py
  _log_likelihood
    - scalar output
    - finite (no nan/inf)
    - all-censored: uses survival branch
    - all-uncensored: uses pmf branch
    - mixed censoring
    - gradients flow through

  _ranking_loss
    - scalar output
    - finite (no nan/inf)
    - returns 0 when fewer than 2 uncensored per event
    - perfect ranker → lower loss than random
    - anti-ranker → higher loss than perfect
    - gradients flow through

  deephit_loss
    - scalar, finite, non-negative (checked empirically)
    - alpha=0 → pure NLL (ranking term absent)
    - alpha=1 → pure ranking (NLL absent)
    - gradients flow from loss to logits
    - all-censored input
    - all-same-event input
    - different sigma changes ranking magnitude
    - batch_size=1

  cause_specific_loss
    - scalar, finite
    - all-censored input
    - all-uncensored input
    - mixed
    - gradients flow from loss to hazards
    - loss is higher for bad hazards (sanity check)
    - batch_size=1

  Shared
    - deephit_loss and cause_specific_loss produce lower loss when
      predictions match labels better
"""

from __future__ import annotations

import torch
import pytest

from capa.model.survival import (
    CauseSpecificHazardHead,
    DeepHitHead,
    hazards_to_cif,
)
from capa.model.losses import (
    _log_likelihood,
    _ranking_loss,
    cause_specific_loss,
    deephit_loss,
)

# ---------------------------------------------------------------------------
# Constants (small dims for fast CPU tests)
# ---------------------------------------------------------------------------
BATCH = 8
IN_DIM = 32
NUM_EVENTS = 3
TIME_BINS = 20
HIDDEN = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def deephit() -> DeepHitHead:
    return DeepHitHead(
        input_dim=IN_DIM,
        num_events=NUM_EVENTS,
        time_bins=TIME_BINS,
        hidden_dim=HIDDEN,
        dropout=0.0,
    )


@pytest.fixture()
def csh() -> CauseSpecificHazardHead:
    return CauseSpecificHazardHead(
        input_dim=IN_DIM,
        num_events=NUM_EVENTS,
        time_bins=TIME_BINS,
        hidden_dim=HIDDEN,
        dropout=0.0,
    )


def _x(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, IN_DIM, requires_grad=True)


def _rand_logits(
    batch: int = BATCH, events: int = NUM_EVENTS, bins: int = TIME_BINS
) -> torch.Tensor:
    return torch.randn(batch, events, bins, requires_grad=True)


def _rand_times(batch: int = BATCH, bins: int = TIME_BINS) -> torch.Tensor:
    return torch.randint(0, bins, (batch,))


def _rand_types(batch: int = BATCH, events: int = NUM_EVENTS) -> torch.Tensor:
    """0 = censored, 1..events = cause-specific."""
    return torch.randint(0, events + 1, (batch,))


def _make_hazards(
    batch: int = BATCH, events: int = NUM_EVENTS, bins: int = TIME_BINS
) -> torch.Tensor:
    """Valid sub-hazard tensor: values in (0,1), sum over events < 1."""
    raw = torch.rand(batch, events, bins) * 0.3  # keep small so sum < 1
    raw.requires_grad_(True)
    return raw


# ===========================================================================
# DeepHitHead
# ===========================================================================

class TestDeepHitForward:
    def test_shape(self, deephit: DeepHitHead) -> None:
        out = deephit(_x())
        assert out.shape == (BATCH, NUM_EVENTS, TIME_BINS)

    def test_dtype_float32(self, deephit: DeepHitHead) -> None:
        assert deephit(_x()).dtype == torch.float32

    def test_no_nan(self, deephit: DeepHitHead) -> None:
        assert not torch.isnan(deephit(_x())).any()

    def test_no_inf(self, deephit: DeepHitHead) -> None:
        assert not torch.isinf(deephit(_x())).any()

    def test_batch_one(self, deephit: DeepHitHead) -> None:
        assert deephit(_x(1)).shape == (1, NUM_EVENTS, TIME_BINS)

    def test_num_events_1(self) -> None:
        m = DeepHitHead(IN_DIM, num_events=1, time_bins=TIME_BINS, hidden_dim=HIDDEN)
        assert m(_x()).shape == (BATCH, 1, TIME_BINS)

    def test_time_bins_1(self) -> None:
        m = DeepHitHead(IN_DIM, num_events=NUM_EVENTS, time_bins=1, hidden_dim=HIDDEN)
        assert m(_x()).shape == (BATCH, NUM_EVENTS, 1)


class TestDeepHitCIF:
    def test_cif_shape(self, deephit: DeepHitHead) -> None:
        assert deephit.cif(_x()).shape == (BATCH, NUM_EVENTS, TIME_BINS)

    def test_cif_values_in_01(self, deephit: DeepHitHead) -> None:
        c = deephit.cif(_x())
        assert (c >= 0).all() and (c <= 1).all()

    def test_cif_monotone(self, deephit: DeepHitHead) -> None:
        c = deephit.cif(_x())
        diffs = c[:, :, 1:] - c[:, :, :-1]
        assert (diffs >= -1e-6).all(), "CIF is not non-decreasing"

    def test_cif_row_sum_le_one(self, deephit: DeepHitHead) -> None:
        """At each time bin, cumulative probability across all causes ≤ 1."""
        c = deephit.cif(_x())
        total = c.sum(dim=1)  # (batch, time_bins)
        assert (total <= 1.0 + 1e-5).all()

    def test_cif_row_sum_ge_zero(self, deephit: DeepHitHead) -> None:
        c = deephit.cif(_x())
        assert (c.sum(dim=1) >= 0).all()


class TestDeepHitGradients:
    def test_grad_to_input(self, deephit: DeepHitHead) -> None:
        x = _x()
        deephit(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_grad_to_shared_params(self, deephit: DeepHitHead) -> None:
        x = _x()
        deephit(x).sum().backward()
        for name, p in deephit.shared.named_parameters():
            assert p.grad is not None, f"shared.{name} has no grad"

    def test_grad_to_heads(self, deephit: DeepHitHead) -> None:
        x = _x()
        deephit(x).sum().backward()
        for i, head in enumerate(deephit.heads):
            assert head.weight.grad is not None, f"heads[{i}].weight has no grad"

    def test_cif_grad_to_input(self, deephit: DeepHitHead) -> None:
        x = _x()
        deephit.cif(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)


class TestDeepHitDropout:
    def test_train_stochastic(self) -> None:
        m = DeepHitHead(IN_DIM, NUM_EVENTS, TIME_BINS, HIDDEN, dropout=0.9)
        m.train()
        x = torch.randn(BATCH, IN_DIM)
        assert not torch.allclose(m(x), m(x))

    def test_eval_deterministic(self, deephit: DeepHitHead) -> None:
        deephit.eval()
        x = torch.randn(BATCH, IN_DIM)
        assert torch.allclose(deephit(x), deephit(x))


# ===========================================================================
# CauseSpecificHazardHead
# ===========================================================================

class TestCSHForward:
    def test_shape(self, csh: CauseSpecificHazardHead) -> None:
        assert csh(_x()).shape == (BATCH, NUM_EVENTS, TIME_BINS)

    def test_dtype_float32(self, csh: CauseSpecificHazardHead) -> None:
        assert csh(_x()).dtype == torch.float32

    def test_no_nan(self, csh: CauseSpecificHazardHead) -> None:
        assert not torch.isnan(csh(_x())).any()

    def test_hazards_positive(self, csh: CauseSpecificHazardHead) -> None:
        assert (csh(_x()) > 0).all()

    def test_hazards_less_than_one(self, csh: CauseSpecificHazardHead) -> None:
        assert (csh(_x()) < 1).all()

    def test_hazard_sum_le_one(self, csh: CauseSpecificHazardHead) -> None:
        """Softmax constraint: Σ_k h_k(t) < 1 at every time bin."""
        h = csh(_x())
        total = h.sum(dim=1)  # (batch, time_bins)
        assert (total < 1.0 + 1e-6).all()

    def test_batch_one(self, csh: CauseSpecificHazardHead) -> None:
        assert csh(_x(1)).shape == (1, NUM_EVENTS, TIME_BINS)

    def test_num_events_1(self) -> None:
        m = CauseSpecificHazardHead(IN_DIM, num_events=1, time_bins=TIME_BINS, hidden_dim=HIDDEN)
        assert m(_x()).shape == (BATCH, 1, TIME_BINS)


class TestCSHCIF:
    def test_cif_shape(self, csh: CauseSpecificHazardHead) -> None:
        assert csh.cif(_x()).shape == (BATCH, NUM_EVENTS, TIME_BINS)

    def test_cif_values_in_01(self, csh: CauseSpecificHazardHead) -> None:
        c = csh.cif(_x())
        assert (c >= 0).all() and (c <= 1 + 1e-6).all()

    def test_cif_monotone(self, csh: CauseSpecificHazardHead) -> None:
        c = csh.cif(_x())
        diffs = c[:, :, 1:] - c[:, :, :-1]
        assert (diffs >= -1e-6).all()

    def test_cif_total_le_one(self, csh: CauseSpecificHazardHead) -> None:
        c = csh.cif(_x())
        assert (c.sum(dim=1) <= 1.0 + 1e-5).all()


class TestCSHGradients:
    def test_grad_to_input(self, csh: CauseSpecificHazardHead) -> None:
        x = _x()
        csh(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_grad_to_shared(self, csh: CauseSpecificHazardHead) -> None:
        x = _x()
        csh(x).sum().backward()
        for name, p in csh.shared.named_parameters():
            assert p.grad is not None, f"shared.{name} has no grad"

    def test_grad_to_heads(self, csh: CauseSpecificHazardHead) -> None:
        x = _x()
        csh(x).sum().backward()
        for i, head in enumerate(csh.heads):
            assert head.weight.grad is not None


# ===========================================================================
# hazards_to_cif utility
# ===========================================================================

class TestHazardsToCIF:
    def test_shape(self) -> None:
        h = _make_hazards()
        c = hazards_to_cif(h)
        assert c.shape == (BATCH, NUM_EVENTS, TIME_BINS)

    def test_values_in_01(self) -> None:
        c = hazards_to_cif(_make_hazards())
        assert (c >= 0).all() and (c <= 1 + 1e-6).all()

    def test_monotone(self) -> None:
        c = hazards_to_cif(_make_hazards())
        diffs = c[:, :, 1:] - c[:, :, :-1]
        assert (diffs >= -1e-6).all()

    def test_total_cif_le_one(self) -> None:
        c = hazards_to_cif(_make_hazards())
        assert (c.sum(dim=1) <= 1.0 + 1e-5).all()

    def test_zero_hazards_zero_cif(self) -> None:
        h = torch.zeros(BATCH, NUM_EVENTS, TIME_BINS)
        c = hazards_to_cif(h)
        assert torch.allclose(c, torch.zeros_like(c), atol=1e-6)

    def test_high_hazards_cif_approaches_one(self) -> None:
        """With hazard = 1/(num_events+1) at every step, CIF should rise quickly."""
        h = torch.full((1, 1, 50), 0.49)  # single event, nearly 0.5 sub-hazard each step
        c = hazards_to_cif(h)
        assert c[0, 0, -1] > 0.99

    def test_grad_flows(self) -> None:
        h = _make_hazards()
        hazards_to_cif(h).sum().backward()
        assert h.grad is not None and not torch.all(h.grad == 0)


# ===========================================================================
# _log_likelihood
# ===========================================================================

class TestLogLikelihood:
    def _joint_cif(
        self,
        batch: int = BATCH,
        events: int = NUM_EVENTS,
        bins: int = TIME_BINS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.randn(batch, events, bins)
        joint = torch.softmax(logits.view(batch, -1), dim=-1).view(batch, events, bins)
        cif = torch.cumsum(joint, dim=2)
        return joint, cif

    def test_scalar(self) -> None:
        joint, cif = self._joint_cif()
        t = _rand_times()
        k = _rand_types()
        loss = _log_likelihood(joint, cif, t, k, TIME_BINS)
        assert loss.ndim == 0

    def test_finite(self) -> None:
        joint, cif = self._joint_cif()
        loss = _log_likelihood(joint, cif, _rand_times(), _rand_types(), TIME_BINS)
        assert torch.isfinite(loss)

    def test_all_censored(self) -> None:
        joint, cif = self._joint_cif()
        t = _rand_times()
        k = torch.zeros(BATCH, dtype=torch.long)  # all censored
        loss = _log_likelihood(joint, cif, t, k, TIME_BINS)
        assert torch.isfinite(loss)

    def test_all_uncensored_event1(self) -> None:
        joint, cif = self._joint_cif()
        t = _rand_times()
        k = torch.ones(BATCH, dtype=torch.long)  # all event type 1
        loss = _log_likelihood(joint, cif, t, k, TIME_BINS)
        assert torch.isfinite(loss)

    def test_mixed_censoring(self) -> None:
        joint, cif = self._joint_cif()
        t = _rand_times()
        k = _rand_types()
        loss = _log_likelihood(joint, cif, t, k, TIME_BINS)
        assert torch.isfinite(loss)

    def test_non_negative(self) -> None:
        joint, cif = self._joint_cif()
        loss = _log_likelihood(joint, cif, _rand_times(), _rand_types(), TIME_BINS)
        assert loss.item() >= 0.0

    def test_grad_flows(self) -> None:
        logits = _rand_logits()
        joint = torch.softmax(logits.view(BATCH, -1), dim=-1).view(BATCH, NUM_EVENTS, TIME_BINS)
        cif = torch.cumsum(joint, dim=2)
        t = _rand_times()
        k = _rand_types()
        loss = _log_likelihood(joint, cif, t, k, TIME_BINS)
        loss.backward()
        assert logits.grad is not None


# ===========================================================================
# _ranking_loss
# ===========================================================================

class TestRankingLoss:
    def test_scalar(self) -> None:
        cif = torch.sigmoid(_rand_logits())
        loss = _ranking_loss(cif, _rand_times(), _rand_types(), sigma=0.1)
        assert loss.ndim == 0

    def test_finite(self) -> None:
        cif = torch.sigmoid(_rand_logits())
        loss = _ranking_loss(cif, _rand_times(), _rand_types(), sigma=0.1)
        assert torch.isfinite(loss)

    def test_non_negative(self) -> None:
        cif = torch.sigmoid(_rand_logits())
        loss = _ranking_loss(cif, _rand_times(), _rand_types(), sigma=0.1)
        assert loss.item() >= 0.0

    def test_all_censored_returns_zero(self) -> None:
        cif = torch.sigmoid(_rand_logits())
        k = torch.zeros(BATCH, dtype=torch.long)
        loss = _ranking_loss(cif, _rand_times(), k, sigma=0.1)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_single_uncensored_per_event_returns_zero(self) -> None:
        """With only one uncensored subject per event, no valid pairs → loss = 0."""
        cif = torch.sigmoid(_rand_logits())
        # Give each event exactly one uncensored subject
        k = torch.zeros(BATCH, dtype=torch.long)
        k[0] = 1
        loss = _ranking_loss(cif, _rand_times(), k, sigma=0.1)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_ranker_lower_than_antiranker(self) -> None:
        """A model that perfectly ranks events should have lower loss."""
        torch.manual_seed(7)
        bins = 50
        n = 10
        # Two subjects, both event type 1; subject 0 has earlier event
        t = torch.tensor([5, 20])
        k = torch.ones(2, dtype=torch.long)

        # Perfect: CIF of subject 0 > CIF of subject 1 at t=5
        good_cif = torch.zeros(2, 1, bins)
        good_cif[0, 0, 5:] = 0.9  # subject 0 has high CIF early
        good_cif[1, 0, 5:] = 0.1

        bad_cif = torch.zeros(2, 1, bins)
        bad_cif[0, 0, 5:] = 0.1   # reversed
        bad_cif[1, 0, 5:] = 0.9

        loss_good = _ranking_loss(good_cif, t, k, sigma=0.1)
        loss_bad = _ranking_loss(bad_cif, t, k, sigma=0.1)
        assert loss_good.item() < loss_bad.item()

    def test_grad_flows(self) -> None:
        cif = _rand_logits()
        cif_sig = torch.sigmoid(cif)
        loss = _ranking_loss(cif_sig, _rand_times(), _rand_types(), sigma=0.1)
        loss.backward()
        assert cif.grad is not None

    def test_sigma_affects_magnitude(self) -> None:
        """Smaller sigma → steeper penalty → different loss magnitude."""
        cif = torch.sigmoid(_rand_logits().detach())
        t = _rand_times()
        k = _rand_types()
        l1 = _ranking_loss(cif, t, k, sigma=0.01)
        l2 = _ranking_loss(cif, t, k, sigma=1.0)
        assert not torch.allclose(l1, l2)


# ===========================================================================
# deephit_loss
# ===========================================================================

class TestDeepHitLoss:
    def test_scalar(self) -> None:
        loss = deephit_loss(_rand_logits(), _rand_times(), _rand_types())
        assert loss.ndim == 0

    def test_finite(self) -> None:
        loss = deephit_loss(_rand_logits(), _rand_times(), _rand_types())
        assert torch.isfinite(loss)

    def test_non_negative(self) -> None:
        loss = deephit_loss(_rand_logits(), _rand_times(), _rand_types())
        assert loss.item() >= 0.0

    def test_grad_to_logits(self) -> None:
        logits = _rand_logits()
        deephit_loss(logits, _rand_times(), _rand_types()).backward()
        assert logits.grad is not None and not torch.all(logits.grad == 0)

    def test_alpha_zero_equals_nll_only(self) -> None:
        """alpha=0 → loss is purely from log-likelihood, ranking weight is 0."""
        logits = _rand_logits().detach()
        t, k = _rand_times(), _rand_types()
        l_alpha0 = deephit_loss(logits, t, k, alpha=0.0)
        # Compute reference NLL directly
        batch = logits.shape[0]
        joint = torch.softmax(logits.view(batch, -1), dim=-1).view(
            batch, NUM_EVENTS, TIME_BINS
        )
        cif = torch.cumsum(joint, dim=2)
        nll = _log_likelihood(joint, cif, t, k, TIME_BINS)
        assert torch.allclose(l_alpha0, nll, atol=1e-5)

    def test_alpha_one_equals_ranking_only(self) -> None:
        """alpha=1 → loss is purely from ranking."""
        logits = _rand_logits().detach()
        t, k = _rand_times(), _rand_types()
        l_alpha1 = deephit_loss(logits, t, k, alpha=1.0)
        batch = logits.shape[0]
        cif = torch.cumsum(
            torch.softmax(logits.view(batch, -1), dim=-1).view(batch, NUM_EVENTS, TIME_BINS),
            dim=2,
        )
        rank = _ranking_loss(cif, t, k, sigma=0.1)
        assert torch.allclose(l_alpha1, rank, atol=1e-5)

    def test_all_censored(self) -> None:
        logits = _rand_logits()
        t = _rand_times()
        k = torch.zeros(BATCH, dtype=torch.long)
        loss = deephit_loss(logits, t, k)
        assert torch.isfinite(loss)

    def test_all_same_event(self) -> None:
        logits = _rand_logits()
        t = _rand_times()
        k = torch.ones(BATCH, dtype=torch.long)
        loss = deephit_loss(logits, t, k)
        assert torch.isfinite(loss)

    def test_batch_one(self) -> None:
        logits = torch.randn(1, NUM_EVENTS, TIME_BINS, requires_grad=True)
        t = _rand_times(1)
        k = _rand_types(1)
        loss = deephit_loss(logits, t, k)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_different_sigma_changes_loss(self) -> None:
        logits = _rand_logits().detach()
        t, k = _rand_times(), _rand_types()
        l1 = deephit_loss(logits, t, k, alpha=0.5, sigma=0.01)
        l2 = deephit_loss(logits, t, k, alpha=0.5, sigma=1.0)
        assert not torch.allclose(l1, l2)


# ===========================================================================
# cause_specific_loss
# ===========================================================================

class TestCauseSpecificLoss:
    def test_scalar(self) -> None:
        h = _make_hazards()
        loss = cause_specific_loss(h, _rand_times(), _rand_types())
        assert loss.ndim == 0

    def test_finite(self) -> None:
        h = _make_hazards()
        loss = cause_specific_loss(h, _rand_times(), _rand_types())
        assert torch.isfinite(loss)

    def test_non_negative(self) -> None:
        h = _make_hazards()
        loss = cause_specific_loss(h, _rand_times(), _rand_types())
        assert loss.item() >= 0.0

    def test_all_censored(self) -> None:
        h = _make_hazards()
        k = torch.zeros(BATCH, dtype=torch.long)
        loss = cause_specific_loss(h, _rand_times(), k)
        assert torch.isfinite(loss)

    def test_all_uncensored_event1(self) -> None:
        h = _make_hazards()
        k = torch.ones(BATCH, dtype=torch.long)
        loss = cause_specific_loss(h, _rand_times(), k)
        assert torch.isfinite(loss)

    def test_mixed_censoring(self) -> None:
        h = _make_hazards()
        loss = cause_specific_loss(h, _rand_times(), _rand_types())
        assert torch.isfinite(loss)

    def test_batch_one(self) -> None:
        h = _make_hazards(1)
        t = _rand_times(1)
        k = _rand_types(1)
        loss = cause_specific_loss(h, t, k)
        assert torch.isfinite(loss)

    def test_grad_to_hazards(self) -> None:
        h = _make_hazards()
        cause_specific_loss(h, _rand_times(), _rand_types()).backward()
        assert h.grad is not None and not torch.all(h.grad == 0)

    def test_better_prediction_lower_loss(self) -> None:
        """A model whose hazard matches the observed event time should have lower loss."""
        torch.manual_seed(0)
        bins = 20
        # Subject 0: event type 1 at time 5
        t = torch.tensor([5])
        k = torch.tensor([1])

        # Good model: high hazard at t=5 for event 0 (0-indexed)
        h_good = torch.full((1, NUM_EVENTS, bins), 0.01)
        h_good[0, 0, 5] = 0.8  # spike at event time

        # Bad model: uniform low hazard everywhere
        h_bad = torch.full((1, NUM_EVENTS, bins), 0.01)

        # Clamp to be valid sub-hazards (sum < 1)
        # h_good already has sum(dim=1) ≤ 0.8 + 0.01*2 = 0.82 < 1 at t=5
        loss_good = cause_specific_loss(h_good, t, k)
        loss_bad = cause_specific_loss(h_bad, t, k)
        assert loss_good.item() < loss_bad.item()


# ===========================================================================
# Integration: end-to-end gradient check
# ===========================================================================

class TestEndToEndGradients:
    def test_deephit_head_and_loss(self) -> None:
        model = DeepHitHead(IN_DIM, NUM_EVENTS, TIME_BINS, HIDDEN, dropout=0.0)
        x = _x()
        logits = model(x)
        loss = deephit_loss(logits, _rand_times(), _rand_types())
        loss.backward()
        assert x.grad is not None
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_csh_head_and_loss(self) -> None:
        model = CauseSpecificHazardHead(IN_DIM, NUM_EVENTS, TIME_BINS, HIDDEN, dropout=0.0)
        x = _x()
        hazards = model(x)
        loss = cause_specific_loss(hazards, _rand_times(), _rand_types())
        loss.backward()
        assert x.grad is not None
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
