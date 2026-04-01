"""UCI BMT dataset loading and cleaning.

Source: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children
Format: ARFF (Attribute-Relation File Format), 187 patients × 37 attributes.

Column mapping reference
-------------------------
The raw ARFF column names are inconsistently cased and abbreviated.  Every mapping
is documented below next to ``RAW_TO_CLEAN`` and ``COLUMN_NOTES``.

Missing-value strategy
-----------------------
Missing values in the raw file are encoded as ``?``.  For each column the strategy is:

  - ``donor_cmv``, ``recipient_cmv``: 4 rows each — impute with mode (0 = seronegative,
    the most common status; donor CMV+ is a risk factor so conservative imputation matters,
    but the dataset is too small for model-based imputation).
  - ``cmv_serostatus``: same 4 rows — imputed from donor/recipient CMV after their imputation.
  - ``recipient_abo``, ``recipient_rh``, ``abo_match``: 4 rows — impute with mode.
  - ``n_antigen_mismatches``, ``n_allele_mismatches``: 0 rows missing (sentinel −1 = 0 diffs).
  - ``extensive_chronic_gvhd``: 17 rows — patients who died before chronic GvHD could be
    assessed; code as 0 (no extensive cGvHD observed/reported).
  - ``cd3_dose``, ``cd3_cd34_ratio``: 1 row each — impute with median (continuous dosage).
  - ``days_to_anc_recovery``, ``days_to_plt_recovery``, ``days_to_acute_gvhd_iii_iv``:
    sentinel 1_000_000 = event not observed by end of study (censored); replaced with NaN,
    and a companion boolean column ``*_observed`` is added.

Encoding oddities
------------------
  - ``aGvHDIIIIV``  in raw data: 0 = event occurred (Yes), 1 = event did NOT occur (No).
    We recode to the standard convention: 1 = Yes, 0 = No → clean column ``acute_gvhd_iii_iv``.
  - ``extcGvHD`` in raw data: same inversion (0 = Yes, 1 = No).
    Recoded to 1 = Yes, 0 = No → ``extensive_chronic_gvhd``.
  - ``HLAmatch``: lower is better (0 = 10/10 match).
  - ``Antigen`` / ``Alel``: −1 encodes *zero* differences (counter-intuitive);
    0 encodes one difference, 1 encodes two, etc.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping: raw ARFF name → clean snake_case name
# ---------------------------------------------------------------------------
RAW_TO_CLEAN: dict[str, str] = {
    # --- Recipient demographics ---
    "Recipientgender": "recipient_sex",
    # 1 = Male, 0 = Female
    "Recipientage": "recipient_age",
    # Age at transplantation in years (continuous)
    "Recipientage10": "recipient_age_gte10",
    # Binary threshold: 0 = <10 yrs, 1 = ≥10 yrs
    "Recipientageint": "recipient_age_group",
    # Ordinal intervals: 0 = (0–5], 1 = (5–10], 2 = (10–20]
    "Rbodymass": "recipient_body_mass_kg",
    # Body mass at transplantation (kg, continuous)

    # --- Donor demographics ---
    "Donorage": "donor_age",
    # Age at stem-cell apheresis in years (continuous)
    "Donorage35": "donor_age_gte35",
    # Binary threshold: 0 = <35 yrs, 1 = ≥35 yrs

    # --- Stem cell product ---
    "Stemcellsource": "stem_cell_source",
    # 1 = Peripheral blood (mobilised HSCs), 0 = Bone marrow aspirate
    "CD34kgx10d6": "cd34_dose",
    # CD34+ cell dose per kg recipient body weight (×10⁶ cells/kg, continuous)
    # CD34+ marks haematopoietic stem/progenitor cells; higher dose → faster engraftment
    "CD3dkgx10d8": "cd3_dose",
    # CD3+ (T-cell) dose per kg (×10⁸ cells/kg, continuous); higher dose → more GvHD risk
    "CD3dCD34": "cd3_cd34_ratio",
    # CD3+/CD34+ ratio (continuous); captures T-cell contamination relative to stem-cell dose

    # --- Blood group compatibility ---
    "DonorABO": "donor_abo",
    # Donor ABO blood type: 0 = O, 1 = A, −1 = B, 2 = AB
    "RecipientABO": "recipient_abo",
    # Recipient ABO blood type: same encoding as donor_abo
    "RecipientRh": "recipient_rh",
    # Recipient Rh factor: 1 = Rh+, 0 = Rh−
    "ABOmatch": "abo_match",
    # ABO compatibility: 1 = ABO-matched, 0 = ABO-mismatched
    # Note: source ARFF has a typo ("matched−1, mismatched−1"); values are 0 and 1.

    # --- CMV serology ---
    "DonorCMV": "donor_cmv",
    # Cytomegalovirus serostatus of donor: 1 = seropositive, 0 = seronegative
    "RecipientCMV": "recipient_cmv",
    # CMV serostatus of recipient: 1 = seropositive, 0 = seronegative
    "CMVstatus": "cmv_serostatus",
    # Combined donor-recipient CMV compatibility: 0 = D−/R−, 1 = D−/R+,
    # 2 = D+/R−, 3 = D+/R+  (higher = higher CMV-reactivation risk post-transplant)

    # --- Gender match ---
    "Gendermatch": "sex_mismatch_f2m",
    # 1 = Female donor → Male recipient (highest GvHD risk direction), 0 = all other combinations

    # --- Disease ---
    "Disease": "disease",
    # Primary diagnosis: ALL, AML, chronic (CML/MDS), nonmalignant, lymphoma
    "Diseasegroup": "malignant_disease",
    # 1 = malignant disease, 0 = nonmalignant (aplastic anaemia, Fanconi, etc.)
    "Riskgroup": "high_risk",
    # Disease/patient risk at transplant: 1 = high risk, 0 = standard/low risk
    "Txpostrelapse": "retransplant_after_relapse",
    # 0 = first transplant, 1 = second transplant (after relapse of first)

    # --- HLA compatibility ---
    "HLAmatch": "hla_match_score",
    # Overall HLA allele match out of 10 loci (A, B, C, DRB1, DQB1 × 2 alleles each):
    # 0 = 10/10, 1 = 9/10, 2 = 8/10, 3 = 7/10  (lower = better)
    "HLAmismatch": "hla_mismatched",
    # 0 = fully HLA-matched (10/10), 1 = any HLA mismatch
    "Antigen": "n_antigen_mismatches",
    # Number of HLA *antigen*-level differences: −1 = 0, 0 = 1, 1 = 2, 2 = 3
    # (−1 is the sentinel for "no mismatches" — counter-intuitive encoding)
    "Alel": "n_allele_mismatches",
    # Number of HLA *allele*-level differences: −1 = 0, 0 = 1, 1 = 2, 2 = 3, 3 = 4
    "HLAgrI": "hla_mismatch_type",
    # Classification of mismatch type:
    # 0 = matched, 1 = antigen diff only, 2 = allele diff only,
    # 3 = DRB1 diff only, 4 = two diffs (same type), 5 = two diffs (different type),
    # 7 = other/complex  (value 7 appears in data but is undocumented in the ARFF header)

    # --- Outcomes: GvHD ---
    "IIIV": "acute_gvhd_ii_iv",
    # Acute GvHD grade II–IV: 1 = occurred, 0 = did not occur
    "aGvHDIIIIV": "acute_gvhd_iii_iv",
    # Acute GvHD grade III–IV (severe): RAW encoding is INVERTED (0=Yes, 1=No).
    # We recode to standard: 1 = occurred, 0 = did not occur.
    "extcGvHD": "extensive_chronic_gvhd",
    # Extensive chronic GvHD: RAW encoding is INVERTED (0=Yes, 1=No).
    # We recode to standard: 1 = occurred, 0 = did not occur.
    # 17 missing: patients who died before chronic GvHD assessment → coded as 0.

    # --- Outcomes: haematopoietic recovery (time-to-event) ---
    "ANCrecovery": "days_to_anc_recovery",
    # Days from transplant to absolute neutrophil count >0.5×10⁹/L (engraftment)
    # Sentinel 1_000_000 = no recovery observed → replaced with NaN; see days_to_anc_recovery_observed
    "PLTrecovery": "days_to_plt_recovery",
    # Days from transplant to platelet count >50 000/mm³ (platelet engraftment)
    # Sentinel 1_000_000 = no recovery observed → replaced with NaN
    "time_to_aGvHD_III_IV": "days_to_acute_gvhd_iii_iv",
    # Days from transplant to grade III/IV acute GvHD onset
    # Sentinel 1_000_000 = GvHD did not occur (majority of patients) → replaced with NaN

    # --- Outcomes: relapse & survival ---
    "Relapse": "relapse",
    # Disease relapse post-transplant: 1 = relapsed, 0 = no relapse
    "survival_time": "survival_time_days",
    # Overall observation time in days (event or last follow-up)
    "survival_status": "dead",
    # 0 = alive at last follow-up, 1 = deceased (study endpoint)
}

# Sentinel value used in time-to-event columns to indicate the event was NOT observed
_SENTINEL = 1_000_000

# Columns using the 1_000_000 sentinel (must exist in the *cleaned* name space)
_SENTINEL_COLS: tuple[str, ...] = (
    "days_to_anc_recovery",
    "days_to_plt_recovery",
    "days_to_acute_gvhd_iii_iv",
)

# Binary columns whose coding is INVERTED in the raw data (0 = event, 1 = no event).
# We flip them so 1 = event occurred, 0 = event did not occur.
_INVERTED_BINARY_COLS: tuple[str, ...] = (
    "acute_gvhd_iii_iv",
    "extensive_chronic_gvhd",
)

# ---------------------------------------------------------------------------
# Dtype specifications (applied after renaming)
# ---------------------------------------------------------------------------
_CATEGORICAL_COLS: tuple[str, ...] = (
    "disease",
)

_INT8_COLS: tuple[str, ...] = (
    "recipient_sex",
    "donor_age_gte35",
    "stem_cell_source",
    "acute_gvhd_ii_iv",
    "acute_gvhd_iii_iv",
    "sex_mismatch_f2m",
    "donor_abo",
    "recipient_abo",
    "recipient_rh",
    "abo_match",
    "cmv_serostatus",
    "donor_cmv",
    "recipient_cmv",
    "high_risk",
    "retransplant_after_relapse",
    "malignant_disease",
    "hla_match_score",
    "hla_mismatched",
    "n_antigen_mismatches",
    "n_allele_mismatches",
    "hla_mismatch_type",
    "recipient_age_gte10",
    "recipient_age_group",
    "relapse",
    "dead",
    "extensive_chronic_gvhd",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_bmt(path: Path) -> pd.DataFrame:
    """Load, clean, and return the UCI Bone Marrow Transplant dataset.

    Accepts either the raw ARFF file (``bone-marrow.arff``) or a pre-converted
    CSV (``bone-marrow.csv``).

    Parameters
    ----------
    path : Path
        Path to ``bone-marrow.arff`` or ``bone-marrow.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 187 rows and standardised column names.
        Categorical/ordinal columns are :class:`pandas.Int8Dtype` (nullable integer).
        Continuous columns are ``float64``.
        Missing values are ``pd.NA`` / ``NaN`` — **not** imputed here; imputation
        is the responsibility of the training pipeline so that it can be fitted
        on train data only and applied to val/test.

    Notes
    -----
    The returned DataFrame includes three extra boolean columns not present in
    the raw data, derived from sentinel-value handling:

    - ``days_to_anc_recovery_observed``    — ``True`` if neutrophil engraftment occurred
    - ``days_to_plt_recovery_observed``    — ``True`` if platelet engraftment occurred
    - ``days_to_acute_gvhd_iii_iv_observed`` — ``True`` if grade III/IV aGvHD occurred
    """
    logger.info("Loading UCI BMT dataset from %s", path)

    suffix = path.suffix.lower()
    if suffix == ".arff":
        df = _read_arff(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, na_values=["?"])
    else:
        raise ValueError(f"Unsupported file format: {suffix!r} (expected .arff or .csv)")

    logger.info("Raw data: %d rows × %d columns", len(df), len(df.columns))

    df = _rename_columns(df)
    df = _recode_inverted_binary(df)
    df = _handle_sentinels(df)
    df = _fix_extensive_cgvhd_missing(df)
    df = _cast_types(df)

    logger.info("Cleaned data: %d rows × %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_arff(path: Path) -> pd.DataFrame:
    """Parse the ARFF file into a DataFrame.

    The ARFF format is essentially a CSV prefixed by ``@attribute`` metadata
    lines.  We skip everything before ``@data`` and read the rest as CSV,
    treating ``?`` as NaN.

    Parameters
    ----------
    path : Path
        Path to the ``.arff`` file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with ARFF column names.
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    # Collect attribute names in order (preserving ARFF column order)
    attr_names: list[str] = []
    in_data = False
    data_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            # @attribute <name> <type>
            # Name may be quoted; split on whitespace, take token 1
            tokens = stripped.split()
            name = tokens[1].strip("'\"")
            attr_names.append(name)
        elif stripped.lower() == "@data":
            in_data = True
        elif in_data and stripped and not stripped.startswith("%"):
            data_lines.append(stripped)

    csv_text = "\n".join(data_lines)
    df = pd.read_csv(
        io.StringIO(csv_text),
        header=None,
        names=attr_names,
        na_values=["?"],
    )
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the RAW_TO_CLEAN column name mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw ARFF column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with clean snake_case column names.

    Raises
    ------
    KeyError
        If a column present in the DataFrame is not covered by RAW_TO_CLEAN
        (signals that the ARFF schema has changed).
    """
    missing_mappings = set(df.columns) - set(RAW_TO_CLEAN)
    if missing_mappings:
        raise KeyError(
            f"The following raw columns have no clean-name mapping: {sorted(missing_mappings)}. "
            "Update RAW_TO_CLEAN in capa/data/loader.py."
        )
    return df.rename(columns=RAW_TO_CLEAN)


def _recode_inverted_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Flip inverted binary outcome columns to the standard convention.

    In the raw ARFF, ``aGvHDIIIIV`` and ``extcGvHD`` are coded as
    0 = event occurred, 1 = event did NOT occur — the opposite of every other
    binary outcome in this dataset.  We recode them so that 1 = event occurred.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after column renaming.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected binary coding.
    """
    for col in _INVERTED_BINARY_COLS:
        if col in df.columns:
            # 0→1, 1→0, NaN stays NaN
            df[col] = df[col].map({0: 1, 1: 0})
    return df


def _handle_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel 1_000_000 values with NaN and add event-indicator columns.

    For time-to-event columns (ANC recovery, platelet recovery, time to GvHD),
    the raw data encodes "event not observed" as 1_000_000.  We replace these
    with NaN and create companion ``*_observed`` boolean columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after column renaming.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentinel values replaced and new indicator columns.
    """
    for col in _SENTINEL_COLS:
        if col not in df.columns:
            continue
        observed_col = f"{col}_observed"
        df[observed_col] = df[col] != _SENTINEL
        df[col] = df[col].where(df[col] != _SENTINEL, other=float("nan"))
        logger.debug(
            "Column '%s': %d sentinel→NaN, %d events observed",
            col,
            (~df[observed_col]).sum(),
            df[observed_col].sum(),
        )
    return df


def _fix_extensive_cgvhd_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing ``extensive_chronic_gvhd`` as 0 (not observed).

    17 patients have missing chronic GvHD data.  These are almost exclusively
    early deaths (before the ~100-day window required for chronic GvHD assessment).
    Coding them as 0 (extensive cGvHD did not occur) is clinically conservative
    and consistent with competing-risks analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after sentinel handling.

    Returns
    -------
    pd.DataFrame
        DataFrame with extensive_chronic_gvhd NaNs filled with 0.
    """
    col = "extensive_chronic_gvhd"
    if col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(0)
            logger.info(
                "Imputed %d missing '%s' values as 0 (not observed, likely early death)",
                n_missing,
                col,
            )
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate pandas dtypes.

    - Binary / ordinal integer columns → ``pd.Int8Dtype()`` (nullable integer,
      preserves NaN without upcasting to float64).
    - Continuous numeric columns → ``float64`` (default, already correct from CSV parse).
    - Categorical string columns → ``pd.CategoricalDtype``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with default dtypes.

    Returns
    -------
    pd.DataFrame
        DataFrame with explicit dtypes.
    """
    # Nullable integer for binary / ordinal columns
    for col in _INT8_COLS:
        if col in df.columns:
            df[col] = df[col].astype("Int8")

    # Categorical for disease string
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Convenience: column documentation dictionary (for notebooks / inspection)
# ---------------------------------------------------------------------------

COLUMN_DOCS: dict[str, str] = {
    "recipient_sex": "Recipient sex: 1 = Male, 0 = Female",
    "recipient_age": "Recipient age at transplant (years)",
    "recipient_age_gte10": "Recipient age ≥10 yrs: 0 = <10, 1 = ≥10",
    "recipient_age_group": "Recipient age interval: 0 = (0–5], 1 = (5–10], 2 = (10–20]",
    "recipient_body_mass_kg": "Recipient body mass at transplant (kg)",
    "donor_age": "Donor age at stem-cell apheresis (years)",
    "donor_age_gte35": "Donor age ≥35: 0 = <35, 1 = ≥35",
    "stem_cell_source": "Graft source: 1 = Peripheral blood, 0 = Bone marrow",
    "cd34_dose": "CD34+ stem-cell dose (×10⁶/kg recipient weight)",
    "cd3_dose": "CD3+ T-cell dose (×10⁸/kg recipient weight)",
    "cd3_cd34_ratio": "T-cell/stem-cell ratio (CD3+/CD34+)",
    "donor_abo": "Donor ABO: 0 = O, 1 = A, −1 = B, 2 = AB",
    "recipient_abo": "Recipient ABO: 0 = O, 1 = A, −1 = B, 2 = AB",
    "recipient_rh": "Recipient Rh factor: 1 = Rh+, 0 = Rh−",
    "abo_match": "ABO compatibility: 1 = matched, 0 = mismatched",
    "donor_cmv": "Donor CMV serostatus: 1 = positive, 0 = negative",
    "recipient_cmv": "Recipient CMV serostatus: 1 = positive, 0 = negative",
    "cmv_serostatus": (
        "Combined CMV compatibility: 0 = D−/R−, 1 = D−/R+, 2 = D+/R−, 3 = D+/R+ "
        "(higher = greater reactivation risk)"
    ),
    "sex_mismatch_f2m": "Gender mismatch: 1 = Female donor → Male recipient, 0 = other",
    "disease": "Primary diagnosis: ALL, AML, chronic, nonmalignant, lymphoma",
    "malignant_disease": "Disease category: 1 = malignant, 0 = nonmalignant",
    "high_risk": "Risk stratification: 1 = high risk, 0 = standard/low risk",
    "retransplant_after_relapse": "Re-transplant: 1 = second transplant after relapse, 0 = first",
    "hla_match_score": "HLA match out of 10 loci: 0 = 10/10, 1 = 9/10, 2 = 8/10, 3 = 7/10",
    "hla_mismatched": "HLA mismatch: 0 = fully matched (10/10), 1 = any mismatch",
    "n_antigen_mismatches": (
        "Antigen-level mismatches: −1 = zero, 0 = one, 1 = two, 2 = three "
        "(−1 is the raw sentinel for no mismatches)"
    ),
    "n_allele_mismatches": (
        "Allele-level mismatches: −1 = zero, 0 = one, 1 = two, 2 = three, 3 = four"
    ),
    "hla_mismatch_type": (
        "HLA difference classification: 0 = matched, 1 = antigen diff only, "
        "2 = allele diff only, 3 = DRB1 diff only, 4 = two same-type diffs, "
        "5 = two different-type diffs, 7 = other/complex"
    ),
    "acute_gvhd_ii_iv": "Acute GvHD grade II–IV: 1 = occurred, 0 = did not occur",
    "acute_gvhd_iii_iv": (
        "Acute GvHD grade III–IV (severe): 1 = occurred, 0 = did not occur "
        "[recoded from inverted raw encoding]"
    ),
    "extensive_chronic_gvhd": (
        "Extensive chronic GvHD: 1 = occurred, 0 = did not occur "
        "[recoded from inverted raw encoding; 17 early-death patients imputed as 0]"
    ),
    "days_to_anc_recovery": "Days to neutrophil engraftment (ANC >0.5×10⁹/L); NaN = not achieved",
    "days_to_anc_recovery_observed": "True if neutrophil engraftment was observed",
    "days_to_plt_recovery": "Days to platelet engraftment (PLT >50 000/mm³); NaN = not achieved",
    "days_to_plt_recovery_observed": "True if platelet engraftment was observed",
    "days_to_acute_gvhd_iii_iv": "Days to grade III/IV aGvHD onset; NaN = did not occur",
    "days_to_acute_gvhd_iii_iv_observed": "True if grade III/IV aGvHD occurred",
    "relapse": "Disease relapse post-transplant: 1 = relapsed, 0 = no relapse",
    "survival_time_days": "Overall survival / follow-up time (days from transplant)",
    "dead": "Vital status at last follow-up: 1 = deceased, 0 = alive",
}
