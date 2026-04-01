"""Pydantic input/output schemas for the CAPA prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HLATyping(BaseModel):
    """HLA typing for a single subject (donor or recipient).

    Attributes
    ----------
    A : str | None
        HLA-A allele, e.g. ``"A*02:01"``.
    B : str | None
        HLA-B allele.
    C : str | None
        HLA-C allele.
    DRB1 : str | None
        HLA-DRB1 allele.
    DQB1 : str | None
        HLA-DQB1 allele.
    """

    A: str | None = Field(default=None, examples=["A*02:01"])
    B: str | None = Field(default=None, examples=["B*07:02"])
    C: str | None = Field(default=None, examples=["C*07:02"])
    DRB1: str | None = Field(default=None, examples=["DRB1*15:01"])
    DQB1: str | None = Field(default=None, examples=["DQB1*06:02"])


class ClinicalCovariates(BaseModel):
    """Clinical covariates for a transplant pair.

    Attributes
    ----------
    age_recipient : float | None
        Recipient age in years.
    age_donor : float | None
        Donor age in years.
    disease : str | None
        Primary diagnosis (e.g. ``"ALL"``, ``"AML"``).
    conditioning : str | None
        Conditioning regimen (e.g. ``"MAC"``, ``"RIC"``).
    donor_type : str | None
        Donor relationship (e.g. ``"MSD"``, ``"MUD"``).
    """

    age_recipient: float | None = Field(default=None, ge=0.0)
    age_donor: float | None = Field(default=None, ge=0.0)
    disease: str | None = Field(default=None)
    conditioning: str | None = Field(default=None)
    donor_type: str | None = Field(default=None)


class PredictionRequest(BaseModel):
    """Request payload for the CAPA risk prediction endpoint."""

    donor_hla: HLATyping
    recipient_hla: HLATyping
    clinical: ClinicalCovariates = Field(default_factory=ClinicalCovariates)


class EventRisk(BaseModel):
    """Risk curve for a single competing event.

    Attributes
    ----------
    cumulative_incidence : list[float]
        CIF values at each discrete time bin.
    risk_score : float
        Scalar summary risk (CIF at the median follow-up time).
    """

    cumulative_incidence: list[float]
    risk_score: float


class PredictionResponse(BaseModel):
    """Response payload for the CAPA risk prediction endpoint."""

    gvhd: EventRisk
    relapse: EventRisk
    trm: EventRisk
    attention_weights: list[list[float]] | None = Field(
        default=None,
        description="Donor→recipient cross-attention matrix (n_loci × n_loci).",
    )
