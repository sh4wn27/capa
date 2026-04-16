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
    DPB1: str | None = Field(default=None, examples=["DPB1*04:01"],
                              description="HLA-DPB1 allele (optional; enables 6-locus mode).")


class ClinicalCovariates(BaseModel):
    """Clinical covariates for a transplant pair.

    Attributes
    ----------
    age_recipient : float | None
        Recipient age in years.
    age_donor : float | None
        Donor age in years.
    cd34_dose : float | None
        CD34+ cell dose (×10⁶/kg).
    sex_mismatch : bool | None
        True when donor and recipient sex differ.
    disease : str | None
        Primary diagnosis (e.g. ``"ALL"``, ``"AML"``).
    conditioning : str | None
        Conditioning regimen (e.g. ``"MAC"``, ``"RIC"``).
    donor_type : str | None
        Donor relationship (e.g. ``"MSD"``, ``"MUD"``).
    stem_cell_source : str | None
        Graft source (e.g. ``"BM"``, ``"PBSC"``, ``"cord"``).
    """

    age_recipient: float | None = Field(default=None, ge=0.0)
    age_donor: float | None = Field(default=None, ge=0.0)
    cd34_dose: float | None = Field(default=None, ge=0.0)
    sex_mismatch: bool | None = Field(default=None)
    disease: str | None = Field(default=None)
    conditioning: str | None = Field(default=None)
    donor_type: str | None = Field(default=None)
    stem_cell_source: str | None = Field(default=None)


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
        CIF values at each discrete time bin (length = ``time_bins``).
    risk_score : float
        Scalar summary risk — CIF at the final time bin, in ``[0, 1]``.
    time_points : list[float] | None
        Day-axis corresponding to each CIF value (length = ``time_bins``).
        Defaults to uniform 0–730-day grid when omitted by the client.
    """

    cumulative_incidence: list[float]
    risk_score: float = Field(ge=0.0, le=1.0)
    time_points: list[float] | None = Field(default=None)


class PredictionResponse(BaseModel):
    """Response payload for the CAPA risk prediction endpoint."""

    gvhd: EventRisk
    relapse: EventRisk
    trm: EventRisk
    attention_weights: list[list[float]] | None = Field(
        default=None,
        description="Donor→recipient cross-attention matrix (n_loci × n_loci), "
        "last-layer weights averaged over heads.",
    )
    mismatch_count: int | None = Field(
        default=None,
        description="Number of mismatched loci between donor and recipient.",
    )
    model_version: str | None = Field(
        default=None,
        description="Checkpoint identifier, e.g. 'v0.1.0' or a git SHA.",
    )


class DonorEntry(BaseModel):
    """One donor in a multi-donor comparison request.

    Attributes
    ----------
    label : str | None
        Human-readable identifier shown in the comparison table (e.g. donor
        registry ID or a pseudonym).  Defaults to ``"Donor N"`` when omitted.
    donor_hla : HLATyping
        HLA typing for this donor.
    """

    label: str | None = Field(default=None, examples=["Donor A"])
    donor_hla: HLATyping


class ComparisonRequest(BaseModel):
    """Request payload for the multi-donor comparison endpoint.

    Attributes
    ----------
    recipient_hla : HLATyping
        Recipient HLA typing (shared across all donors).
    donors : list[DonorEntry]
        List of candidate donors to compare.  Length must be 2–20.
    clinical : ClinicalCovariates
        Recipient clinical covariates (shared across all donors).
    """

    recipient_hla: HLATyping
    donors: list[DonorEntry] = Field(min_length=2, max_length=20)
    clinical: ClinicalCovariates = Field(default_factory=ClinicalCovariates)


class DonorRiskSummary(BaseModel):
    """Per-donor risk summary within a comparison response.

    Attributes
    ----------
    label : str
        Donor label from the request (or auto-assigned ``"Donor N"``).
    gvhd_risk : float
        GvHD cumulative incidence at the final time bin.
    relapse_risk : float
        Relapse cumulative incidence at the final time bin.
    trm_risk : float
        TRM cumulative incidence at the final time bin.
    mismatch_count : int | None
        Number of mismatched HLA loci.
    rank : int
        1-based rank by ascending composite risk (gvhd + trm, lower is better).
    full_prediction : PredictionResponse
        Full CIF curves for this donor.
    """

    label: str
    gvhd_risk: float = Field(ge=0.0, le=1.0)
    relapse_risk: float = Field(ge=0.0, le=1.0)
    trm_risk: float = Field(ge=0.0, le=1.0)
    mismatch_count: int | None
    rank: int
    full_prediction: PredictionResponse


class ComparisonResponse(BaseModel):
    """Response payload for the multi-donor comparison endpoint.

    Attributes
    ----------
    donors : list[DonorRiskSummary]
        Per-donor results, sorted by ascending rank (best match first).
    best_donor_label : str
        Label of the top-ranked donor.
    model_version : str | None
        Checkpoint version string.
    """

    donors: list[DonorRiskSummary]
    best_donor_label: str
    model_version: str | None = None
