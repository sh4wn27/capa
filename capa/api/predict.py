"""Inference pipeline: HLA strings → risk scores."""

from __future__ import annotations

import logging

import numpy as np
import torch

from capa.api.schemas import ClinicalCovariates, HLATyping, PredictionRequest, PredictionResponse, EventRisk
from capa.config import CAPAConfig, get_config

logger = logging.getLogger(__name__)


def predict_risk(
    donor_hla: dict[str, str],
    recipient_hla: dict[str, str],
    clinical_covariates: dict[str, object] | None = None,
    config: CAPAConfig | None = None,
) -> PredictionResponse:
    """High-level inference: HLA typing dicts → competing risk scores.

    Parameters
    ----------
    donor_hla : dict[str, str]
        Locus → allele string for the donor, e.g. ``{"A": "A*02:01"}``.
    recipient_hla : dict[str, str]
        Locus → allele string for the recipient.
    clinical_covariates : dict[str, object] | None
        Optional clinical covariates. Missing values are imputed with zeros.
    config : CAPAConfig | None
        CAPA configuration. Defaults to ``get_config()``.

    Returns
    -------
    PredictionResponse
        Competing risk curves and scalar risk scores.
    """
    cfg = config or get_config()

    request = PredictionRequest(
        donor_hla=HLATyping(**donor_hla),
        recipient_hla=HLATyping(**recipient_hla),
        clinical=ClinicalCovariates(**(clinical_covariates or {})),
    )

    # TODO: Load model, embed HLA alleles, encode clinical features, run forward pass.
    # Placeholder until model loading is implemented.
    logger.warning("predict_risk: model not yet loaded — returning dummy output")

    time_bins = cfg.model.time_bins
    dummy_cif = [float(i / time_bins) for i in range(time_bins)]

    return PredictionResponse(
        gvhd=EventRisk(cumulative_incidence=dummy_cif, risk_score=0.0),
        relapse=EventRisk(cumulative_incidence=dummy_cif, risk_score=0.0),
        trm=EventRisk(cumulative_incidence=dummy_cif, risk_score=0.0),
    )
