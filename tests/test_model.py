"""Tests for CAPA model components."""

from __future__ import annotations

import torch
import pytest

from capa.model.capa_model import CAPAModel
from capa.model.interaction import CrossAttentionInteraction
from capa.model.losses import deephit_loss
from capa.model.survival import DeepHitHead


BATCH = 4
N_LOCI = 5
EMB_DIM = 64   # use small dims for fast CPU tests
INTERACTION_DIM = 32
TIME_BINS = 20
NUM_EVENTS = 3


class TestCrossAttentionInteraction:
    def test_output_shape(self) -> None:
        net = CrossAttentionInteraction(
            embedding_dim=EMB_DIM,
            interaction_dim=INTERACTION_DIM,
            num_heads=4,
            num_layers=1,
        )
        donor = torch.randn(BATCH, N_LOCI, EMB_DIM)
        recip = torch.randn(BATCH, N_LOCI, EMB_DIM)
        out = net(donor, recip)
        assert out.shape == (BATCH, INTERACTION_DIM)

    def test_no_nan(self) -> None:
        net = CrossAttentionInteraction(embedding_dim=EMB_DIM, interaction_dim=INTERACTION_DIM, num_heads=4)
        out = net(torch.randn(2, N_LOCI, EMB_DIM), torch.randn(2, N_LOCI, EMB_DIM))
        assert not torch.isnan(out).any()


class TestDeepHitHead:
    def test_output_shape(self) -> None:
        head = DeepHitHead(input_dim=INTERACTION_DIM, num_events=NUM_EVENTS, time_bins=TIME_BINS)
        x = torch.randn(BATCH, INTERACTION_DIM)
        out = head(x)
        assert out.shape == (BATCH, NUM_EVENTS, TIME_BINS)


class TestCAPAModel:
    def _make_model(self) -> CAPAModel:
        return CAPAModel(
            embedding_dim=EMB_DIM,
            n_loci=N_LOCI,
            clinical_dim=8,
            interaction_dim=INTERACTION_DIM,
            num_events=NUM_EVENTS,
            time_bins=TIME_BINS,
            num_heads=4,
            num_layers=1,
        )

    def test_forward_shape(self) -> None:
        model = self._make_model()
        donor = torch.randn(BATCH, N_LOCI, EMB_DIM)
        recip = torch.randn(BATCH, N_LOCI, EMB_DIM)
        clin = torch.randn(BATCH, 8)
        out = model(donor, recip, clin)
        assert out.shape == (BATCH, NUM_EVENTS, TIME_BINS)


class TestDeepHitLoss:
    def test_loss_scalar(self) -> None:
        logits = torch.randn(BATCH, NUM_EVENTS, TIME_BINS)
        event_times = torch.randint(0, TIME_BINS, (BATCH,))
        event_types = torch.randint(0, NUM_EVENTS + 1, (BATCH,))
        loss = deephit_loss(logits, event_times, event_types)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_loss_positive(self) -> None:
        logits = torch.randn(BATCH, NUM_EVENTS, TIME_BINS)
        event_times = torch.randint(0, TIME_BINS, (BATCH,))
        event_types = torch.ones(BATCH, dtype=torch.long)
        loss = deephit_loss(logits, event_times, event_types)
        assert loss.item() >= 0.0
