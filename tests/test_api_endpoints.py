"""FastAPI endpoint smoke tests using Starlette's synchronous TestClient.

These tests run against the full FastAPI application in-process without
starting a real server.  All tests exercise the mock/no-checkpoint path so
they run without a trained model or GPU.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from capa.api.predict import app


@pytest.fixture()
def client() -> TestClient:
    """Synchronous ASGI test client wrapping the CAPA FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_returns_200(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_has_status_ok(self, client) -> None:  # type: ignore[no-untyped-def]
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_ready_false_without_checkpoint(self, client) -> None:  # type: ignore[no-untyped-def]
        data = client.get("/health").json()
        # Model has not been loaded in tests — expect ready=False
        assert data["ready"] is False


class TestPredictEndpoint:
    _VALID_PAYLOAD = {
        "donor_hla":     {"A": "A*02:01", "B": "B*07:02", "DRB1": "DRB1*15:01"},
        "recipient_hla": {"A": "A*03:01", "B": "B*08:01", "DRB1": "DRB1*03:01"},
        "clinical":      {"age_recipient": 8, "disease": "ALL"},
    }

    def test_returns_503_without_model(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.post("/predict", json=self._VALID_PAYLOAD)
        # No checkpoint → 503 Service Unavailable
        assert r.status_code == 503

    def test_422_when_no_loci(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.post("/predict", json={
            "donor_hla": {}, "recipient_hla": {},
        })
        assert r.status_code == 422


class TestCompareEndpoint:
    _VALID_PAYLOAD = {
        "recipient_hla": {"A": "A*03:01", "DRB1": "DRB1*03:01"},
        "donors": [
            {"label": "D1", "donor_hla": {"A": "A*02:01", "DRB1": "DRB1*15:01"}},
            {"label": "D2", "donor_hla": {"A": "A*03:01", "DRB1": "DRB1*03:01"}},
        ],
    }

    def test_returns_200_mock_mode(self, client) -> None:  # type: ignore[no-untyped-def]
        # /compare uses mock_response when model is not loaded
        r = client.post("/compare", json=self._VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_ranked_donors(self, client) -> None:  # type: ignore[no-untyped-def]
        data = client.post("/compare", json=self._VALID_PAYLOAD).json()
        assert "donors" in data
        assert len(data["donors"]) == 2
        assert data["donors"][0]["rank"] == 1
        assert data["donors"][1]["rank"] == 2

    def test_best_donor_label_matches(self, client) -> None:  # type: ignore[no-untyped-def]
        data = client.post("/compare", json=self._VALID_PAYLOAD).json()
        assert data["best_donor_label"] == data["donors"][0]["label"]

    def test_too_few_donors_422(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.post("/compare", json={
            "recipient_hla": {"A": "A*03:01"},
            "donors": [{"donor_hla": {"A": "A*02:01"}}],  # only 1, min is 2
        })
        assert r.status_code == 422
