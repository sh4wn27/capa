"""Tests for capa.api schemas and predict pipeline."""

from __future__ import annotations

import pytest

from capa.api.schemas import (
    ClinicalCovariates,
    EventRisk,
    HLATyping,
    PredictionRequest,
    PredictionResponse,
)
from capa.api.predict import predict_risk


class TestSchemas:
    def test_hla_typing_defaults(self) -> None:
        t = HLATyping()
        assert t.A is None

    def test_hla_typing_values(self) -> None:
        t = HLATyping(A="A*02:01", B="B*07:02")
        assert t.A == "A*02:01"

    def test_prediction_request(self) -> None:
        req = PredictionRequest(
            donor_hla=HLATyping(A="A*02:01"),
            recipient_hla=HLATyping(A="A*24:02"),
        )
        assert req.clinical.age_recipient is None

    def test_event_risk(self) -> None:
        er = EventRisk(cumulative_incidence=[0.0, 0.1, 0.2], risk_score=0.15)
        assert len(er.cumulative_incidence) == 3

    def test_prediction_response(self) -> None:
        cif = [0.0] * 10
        resp = PredictionResponse(
            gvhd=EventRisk(cumulative_incidence=cif, risk_score=0.0),
            relapse=EventRisk(cumulative_incidence=cif, risk_score=0.0),
            trm=EventRisk(cumulative_incidence=cif, risk_score=0.0),
        )
        assert resp.attention_weights is None


class TestPredictRisk:
    def test_returns_response(self) -> None:
        resp = predict_risk(
            donor_hla={"A": "A*02:01"},
            recipient_hla={"A": "A*24:02"},
        )
        assert isinstance(resp, PredictionResponse)

    def test_cif_length(self) -> None:
        from capa.config import get_config
        cfg = get_config()
        resp = predict_risk(donor_hla={}, recipient_hla={})
        assert len(resp.gvhd.cumulative_incidence) == cfg.model.time_bins
