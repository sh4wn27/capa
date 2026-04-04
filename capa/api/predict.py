"""FastAPI application — CAPA risk prediction backend.

Endpoints
---------
POST /predict
    Accepts a :class:`~capa.api.schemas.PredictionRequest` and returns a
    :class:`~capa.api.schemas.PredictionResponse` with cumulative incidence
    curves, attention weights, and scalar risk scores.

GET /health
    Liveness / readiness probe used by container orchestrators and the
    Next.js API route.

Startup behaviour
-----------------
On startup the app attempts to:

1. Resolve the checkpoint path from the ``CAPA_CHECKPOINT`` environment
   variable (falls back to ``runs/best/model.pt`` relative to the project
   root).
2. Load the embedding cache from the path in ``CAPAConfig``.
3. Instantiate :class:`~capa.model.capa_model.CAPAModel` and restore weights.

If no checkpoint is found the server starts in **mock mode**: it returns
Weibull-sampled synthetic CIF curves so the frontend remains usable without
a trained model.  A ``model_version`` of ``"mock"`` signals this state.

Environment variables
---------------------
CAPA_CHECKPOINT
    Absolute path to the model checkpoint file (``model.pt``).
CAPA_EMBED__CACHE_PATH
    Override the HDF5 embedding cache path (see :class:`~capa.config.EmbeddingConfig`).
CAPA_EMBED__DEVICE
    Compute device for ESM-2 inference (``"cpu"``, ``"cuda"``, ``"mps"``).
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from capa.api.schemas import (
    ClinicalCovariates,
    EventRisk,
    HLATyping,
    PredictionRequest,
    PredictionResponse,
)
from capa.config import get_config
from capa.model.capa_model import CAPAModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons populated during lifespan startup
# ---------------------------------------------------------------------------
_model: CAPAModel | None = None
_model_version: str = "mock"
_startup_ts: float = 0.0
_device: torch.device = torch.device("cpu")

_LOCI: list[str] = ["A", "B", "C", "DRB1", "DQB1"]
_TIME_BINS: int = 100
_MAX_DAYS: float = 730.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weibull_cif(
    shape: float,
    scale: float,
    n_bins: int = _TIME_BINS,
    max_days: float = _MAX_DAYS,
) -> list[float]:
    """Parametric Weibull CIF for mock / fallback responses."""
    ts = np.linspace(0.0, max_days, n_bins)
    cif = 1.0 - np.exp(-((ts / scale) ** shape))
    return cif.tolist()


def _time_points(n_bins: int = _TIME_BINS, max_days: float = _MAX_DAYS) -> list[float]:
    return np.linspace(0.0, max_days, n_bins).tolist()


def _count_mismatches(donor: HLATyping, recip: HLATyping) -> int:
    """Count loci where donor and recipient alleles differ (both must be non-None)."""
    count = 0
    for locus in _LOCI:
        d = getattr(donor, locus)
        r = getattr(recip, locus)
        if d is not None and r is not None and d != r:
            count += 1
    return count


def _mock_response(req: PredictionRequest) -> PredictionResponse:
    """Generate a plausible-looking synthetic response without a real model."""
    mm = _count_mismatches(req.donor_hla, req.recipient_hla)
    # Shift risk upward with more mismatches
    penalty = mm * 0.06

    tp = _time_points()
    gvhd_cif = _weibull_cif(1.4, max(80.0, 280.0 - mm * 30))
    rel_cif   = _weibull_cif(1.1, max(120.0, 490.0 - mm * 20))
    trm_cif   = _weibull_cif(0.85, max(200.0, 650.0 - mm * 15))

    # Clamp risk scores to [0, 1]
    def _score(cif: list[float], extra: float = 0.0) -> float:
        return float(min(1.0, max(0.0, cif[-1] + extra)))

    # Synthetic 5×5 attention — diagonal dominant with mismatch loci highlighted
    rng = np.random.default_rng(mm)
    attn = rng.dirichlet(alpha=[2.0] * 5, size=5).tolist()

    return PredictionResponse(
        gvhd=EventRisk(
            cumulative_incidence=gvhd_cif,
            risk_score=_score(gvhd_cif, penalty),
            time_points=tp,
        ),
        relapse=EventRisk(
            cumulative_incidence=rel_cif,
            risk_score=_score(rel_cif, penalty * 0.5),
            time_points=tp,
        ),
        trm=EventRisk(
            cumulative_incidence=trm_cif,
            risk_score=_score(trm_cif, penalty * 0.3),
            time_points=tp,
        ),
        attention_weights=attn,
        mismatch_count=mm,
        model_version="mock",
    )


def _model_response(req: PredictionRequest) -> PredictionResponse:
    """Run the trained model and package its output as a PredictionResponse."""
    assert _model is not None  # guaranteed by caller

    donor_dict   = req.donor_hla.model_dump(exclude_none=True)
    recip_dict   = req.recipient_hla.model_dump(exclude_none=True)
    clinical_dict: dict[str, Any] = req.clinical.model_dump(exclude_none=True)

    raw = _model.predict(
        donor_hla=donor_dict,
        recipient_hla=recip_dict,
        clinical=clinical_dict,
        device=_device,
    )

    tp = _time_points()
    mm = _count_mismatches(req.donor_hla, req.recipient_hla)

    def _event(name: str) -> EventRisk:
        ev = raw[name]
        return EventRisk(
            cumulative_incidence=ev["cif"],
            risk_score=float(ev["risk_score"]),
            time_points=tp,
        )

    # Attention: take donor_to_recip from the raw dict (already a list[list[float]])
    attn_raw = raw.get("attention_weights")
    if isinstance(attn_raw, dict):
        attn: list[list[float]] | None = attn_raw.get("donor_to_recip")
    else:
        attn = None

    return PredictionResponse(
        gvhd=_event("gvhd"),
        relapse=_event("relapse"),
        trm=_event("trm"),
        attention_weights=attn,
        mismatch_count=mm,
        model_version=_model_version,
    )


# ---------------------------------------------------------------------------
# Lifespan — model loading on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _model_version, _startup_ts, _device

    _startup_ts = time.monotonic()
    cfg = get_config()

    # Determine device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    logger.info("CAPA backend using device: %s", _device)

    # Resolve checkpoint path
    checkpoint_env = os.environ.get("CAPA_CHECKPOINT", "")
    if checkpoint_env:
        ckpt_path = Path(checkpoint_env)
    else:
        ckpt_path = cfg.training.runs_dir / "best" / "model.pt"

    if not ckpt_path.is_file():
        logger.warning(
            "No checkpoint found at %s — running in mock mode. "
            "Set CAPA_CHECKPOINT env var to point at a real model.pt.",
            ckpt_path,
        )
        yield
        return

    try:
        logger.info("Loading CAPA checkpoint from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=_device, weights_only=False)

        mc = cfg.model
        _model = CAPAModel(
            embedding_dim=cfg.embedding.embedding_dim,
            loci=mc.hla_loci,
            clinical_dim=mc.clinical_dim,
            interaction_dim=mc.interaction_dim,
            num_heads=mc.interaction_heads,
            num_layers=mc.interaction_layers,
            dropout=0.0,  # inference — no dropout
            time_bins=mc.time_bins,
            num_events=mc.num_events,
        )

        # Support checkpoints saved as {"model_state_dict": ..., "version": ...}
        # or bare state_dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            _model.load_state_dict(state["model_state_dict"])
            _model_version = str(state.get("version", ckpt_path.stem))
        else:
            _model.load_state_dict(state)
            _model_version = ckpt_path.stem

        _model.to(_device)
        _model.eval()

        # Attach embedding cache if available
        cache_path = cfg.embedding.cache_path
        if cache_path.is_file():
            try:
                from capa.embeddings.cache import EmbeddingCache

                cache = EmbeddingCache(cache_path, mode="r")
                _model.set_inference_components(cache=cache)
                logger.info("Embedding cache loaded from %s", cache_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load embedding cache: %s", exc)

        elapsed = time.monotonic() - _startup_ts
        logger.info(
            "CAPA model %s loaded in %.1f s (%s parameters)",
            _model_version,
            elapsed,
            f"{sum(p.numel() for p in _model.parameters()):,}",
        )
    except Exception as exc:
        logger.error("Failed to load checkpoint — falling back to mock mode: %s", exc)
        _model = None

    yield

    # Shutdown — nothing to clean up (embedding cache is read-only HDF5)
    logger.info("CAPA backend shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CAPA Prediction API",
    description=(
        "Competing-risks survival predictions from HLA typing and clinical "
        "covariates using the CAPA protein-language-model framework."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

# Allow the Next.js frontend (any origin in development; tighten for production
# via the CAPA_CORS_ORIGINS env var).
_cors_origins = os.environ.get("CAPA_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Exception handler — surface upstream errors as structured JSON
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception in %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness + readiness probe")
async def health() -> dict[str, str | bool | float]:
    """Return server status, model version, and uptime.

    A ``200`` response with ``"ready": true`` means the model is loaded.
    ``"ready": false`` indicates mock mode (no checkpoint found).
    """
    return {
        "status": "ok",
        "model_version": _model_version,
        "ready": _model is not None,
        "uptime_seconds": round(time.monotonic() - _startup_ts, 1) if _startup_ts else 0.0,
        "device": str(_device),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict competing-risks outcomes for a donor–recipient pair",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict GvHD, relapse, and TRM cumulative incidence curves.

    Parameters
    ----------
    request : PredictionRequest
        Donor HLA typing, recipient HLA typing, and optional clinical
        covariates.  At least one HLA locus must be provided for each side.

    Returns
    -------
    PredictionResponse
        Cumulative incidence curves (100 time points, 0–730 days) and scalar
        risk scores for GvHD, relapse, and TRM.  Includes the cross-attention
        matrix and mismatch count.

    Raises
    ------
    HTTPException
        422 — if neither donor nor recipient has any HLA locus filled.
        500 — if model inference fails unexpectedly.
    """
    # Basic input guard: at least one locus per side
    donor_loci = [
        v for v in request.donor_hla.model_dump().values() if v is not None
    ]
    recip_loci = [
        v for v in request.recipient_hla.model_dump().values() if v is not None
    ]
    if not donor_loci or not recip_loci:
        raise HTTPException(
            status_code=422,
            detail="At least one HLA locus must be provided for both donor and recipient.",
        )

    if _model is None:
        # Mock mode — still useful for frontend development
        logger.debug("Mock mode: returning synthetic response")
        return _mock_response(request)

    try:
        return _model_response(request)
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
