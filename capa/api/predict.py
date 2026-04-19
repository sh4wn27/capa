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

If no checkpoint is found, or if the checkpoint file is empty / corrupt,
the server starts anyway but ``/predict`` returns **HTTP 503** with
``{"detail": "Model not yet trained. Run scripts/train.py first."}``.
``/health`` always returns HTTP 200 regardless of model state — check the
``"ready"`` and ``"startup_error"`` fields to distinguish the cases.

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
    ComparisonRequest,
    ComparisonResponse,
    DonorRiskSummary,
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
_model_version: str = "untrained"
_startup_ts: float = 0.0
_device: torch.device = torch.device("cpu")

# Set to a human-readable string when the model cannot be loaded.
# None means the model loaded successfully.
# "no_checkpoint" means the file was missing (untrained state).
# Any other string is an unexpected load error.
_startup_error: str | None = "no_checkpoint"

_LOCI: list[str] = ["A", "B", "C", "DRB1", "DQB1", "DPB1"]
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

    # Synthetic n×n attention — diagonal dominant with mismatch loci highlighted
    # Size adapts to the number of loci actually provided
    n_loci = sum(
        1 for loc in _LOCI if getattr(req.donor_hla, loc) is not None
    )
    n_loci = max(n_loci, 1)
    rng = np.random.default_rng(mm)
    attn = rng.dirichlet(alpha=[2.0] * n_loci, size=n_loci).tolist()

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
    global _model, _model_version, _startup_ts, _device, _startup_error

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
            "No checkpoint found at %s. "
            "Run scripts/train.py to train the model, or set CAPA_CHECKPOINT "
            "to point at an existing model.pt. "
            "The server will start but /predict will return 503 until trained.",
            ckpt_path,
        )
        # _startup_error stays "no_checkpoint" (its initial value)
        yield
        return

    # File exists — attempt to load it
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

        _startup_error = None  # model loaded successfully
        elapsed = time.monotonic() - _startup_ts
        logger.info(
            "CAPA model %s loaded in %.1f s (%s parameters)",
            _model_version,
            elapsed,
            f"{sum(p.numel() for p in _model.parameters()):,}",
        )
    except Exception as exc:
        # Checkpoint file exists but is empty, corrupt, or has wrong architecture.
        # Do NOT crash — the server stays up so /health and monitoring still work.
        _model = None
        _startup_error = f"load_failed: {exc}"
        logger.error(
            "Failed to load checkpoint %s — /predict will return 503 until "
            "a valid checkpoint is available. Error: %s",
            ckpt_path,
            exc,
        )

    yield

    # Shutdown — nothing to clean up (embedding cache is read-only HDF5)
    logger.info("CAPA backend shutting down")


# ---------------------------------------------------------------------------
# Public helper — usable without an HTTP server (tests, scripts, notebooks)
# ---------------------------------------------------------------------------


def predict_risk(
    donor_hla: dict[str, str],
    recipient_hla: dict[str, str],
    clinical: dict[str, Any] | None = None,
) -> PredictionResponse:
    """Generate a prediction response without going through the HTTP layer.

    Uses the loaded model when available, otherwise falls back to the mock
    Weibull-based response (same as the server's no-checkpoint behaviour).

    Parameters
    ----------
    donor_hla : dict[str, str]
        Mapping ``locus → allele_name``, e.g. ``{"A": "A*02:01"}``.
    recipient_hla : dict[str, str]
        Same format.
    clinical : dict[str, Any] | None
        Optional clinical covariates dict.

    Returns
    -------
    PredictionResponse
        Competing-risk curves and scalar risk scores.
    """
    req = PredictionRequest(
        donor_hla=HLATyping(**donor_hla),
        recipient_hla=HLATyping(**recipient_hla),
    )
    if clinical:
        from capa.api.schemas import ClinicalCovariates
        req = PredictionRequest(
            donor_hla=HLATyping(**donor_hla),
            recipient_hla=HLATyping(**recipient_hla),
            clinical=ClinicalCovariates(**clinical),
        )
    if _model is not None:
        return _model_response(req)
    return _mock_response(req)


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
async def health() -> dict[str, str | bool | float | None]:
    """Return server status, model version, and uptime.

    Always returns HTTP 200 so container orchestrators and load-balancers
    treat the process as alive.  Callers should inspect ``"ready"`` to
    determine whether predictions are available:

    * ``"ready": true``  — model loaded, ``/predict`` is fully operational.
    * ``"ready": false`` — model not loaded; ``/predict`` will return 503.
      Check ``"startup_error"`` for the reason.
    """
    return {
        "status": "ok",
        "model_version": _model_version,
        "ready": _model is not None,
        "startup_error": _startup_error,
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
        # Model not ready — surface a clear 503 so the caller knows to train first.
        raise HTTPException(
            status_code=503,
            detail="Model not yet trained. Run scripts/train.py first.",
        )

    try:
        return _model_response(request)
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Rank multiple candidate donors for one recipient",
)
async def compare(request: ComparisonRequest) -> ComparisonResponse:
    """Predict outcomes for every candidate donor and return a ranked list.

    Parameters
    ----------
    request : ComparisonRequest
        Recipient HLA typing, list of 2–20 donor entries, and optional
        shared clinical covariates.

    Returns
    -------
    ComparisonResponse
        Ranked donor list (best match first), with full CIF curves per donor.
    """
    summaries: list[DonorRiskSummary] = []

    for i, entry in enumerate(request.donors):
        label = entry.label or f"Donor {i + 1}"
        pred_req = PredictionRequest(
            donor_hla=entry.donor_hla,
            recipient_hla=request.recipient_hla,
            clinical=request.clinical,
        )
        if _model is not None:
            pred = _model_response(pred_req)
        else:
            pred = _mock_response(pred_req)

        summaries.append(
            DonorRiskSummary(
                label=label,
                gvhd_risk=pred.gvhd.risk_score,
                relapse_risk=pred.relapse.risk_score,
                trm_risk=pred.trm.risk_score,
                mismatch_count=pred.mismatch_count,
                rank=0,  # filled in after sorting
                full_prediction=pred,
            )
        )

    # Rank by composite acute-risk score: GvHD + TRM (lower is better)
    summaries.sort(key=lambda s: s.gvhd_risk + s.trm_risk)
    for rank, s in enumerate(summaries, start=1):
        # rank is set after construction; use model_copy for pydantic v2
        summaries[rank - 1] = s.model_copy(update={"rank": rank})

    return ComparisonResponse(
        donors=summaries,
        best_donor_label=summaries[0].label,
        model_version=_model_version,
    )
