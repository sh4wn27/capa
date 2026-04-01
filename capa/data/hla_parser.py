"""Parse HLA typing strings into structured allele representations.

The module handles two distinct input forms:

1. **WHO allele-level notation** (modern, high-resolution):
   ``"A*02:01"``, ``"DRB1*15:01:01:02"``
   Follows the WHO HLA Nomenclature Committee standard:
   ``<gene>*<field1>:<field2>[:<field3>[:<field4>]]``

2. **Serological (antigen-level) notation** (older, lower-resolution):
   ``"A2"``, ``"B7"``, ``"Cw7"``, ``"DR15"``, ``"DQ5"``
   Widely used in older clinical records and the UCI BMT dataset mismatch
   summary columns (Antigen, Alel).  No asterisk; gene prefix directly
   followed by the antigen number.

A special note on the UCI BMT dataset
---------------------------------------
The UCI dataset (``bone-marrow.arff``) does **not** contain per-locus HLA
allele strings for individual donors or recipients.  It records only
aggregate mismatch statistics:

  * ``HLAmatch``  — overall match grade (10/10 to 7/10)
  * ``HLAmismatch``  — binary mismatch flag
  * ``Antigen``  — count of antigen-level differences
  * ``Alel``  — count of allele-level differences
  * ``HLAgrI``  — classification of mismatch type

These are parsed into an :class:`HLAMismatchSummary` via
:func:`parse_uci_hla_columns`.

Per-locus allele strings are expected to come from either:
  * User input to the prediction API (:mod:`capa.api.schemas`)
  * The IPD-IMGT/HLA database (:mod:`capa.embeddings.hla_sequences`)

Gene name normalisation
-----------------------
The module normalises raw gene tokens to canonical WHO gene names:

  ============  ===========
  Raw token     Canonical
  ============  ===========
  A             A
  B             B
  C, Cw         C
  DR, DRB1      DRB1
  DRB3/4/5      DRB3/4/5
  DQ, DQB1      DQB1
  DQA1          DQA1
  DP, DPB1      DPB1
  DPA1          DPA1
  ============  ===========

Representative allele table
----------------------------
When only antigen-level resolution is available (e.g. ``"A2"``), a fallback
two-field allele is returned for sequence lookup.  The table
:data:`ANTIGEN_TO_COMMON_ALLELE` stores the globally most-frequent allele
within each serological group based on published NMDP/WMDA haplotype
frequency data (predominantly Northern-European reference population).  This
is an approximation; actual allele frequency varies by ethnicity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene name normalisation
# ---------------------------------------------------------------------------

#: Maps raw gene-prefix tokens → canonical WHO gene name.
CANONICAL_GENE: dict[str, str] = {
    "A": "A",
    "B": "B",
    "C": "C",
    "Cw": "C",  # old serological prefix for HLA-C
    "DR": "DRB1",
    "DRB1": "DRB1",
    "DRB3": "DRB3",
    "DRB4": "DRB4",
    "DRB5": "DRB5",
    "DQ": "DQB1",
    "DQB1": "DQB1",
    "DQA1": "DQA1",
    "DP": "DPB1",
    "DPB1": "DPB1",
    "DPA1": "DPA1",
}

# Loci used for the standard 10-locus HLA match (A, B, C × 2 alleles + DRB1, DQB1 × 2)
STANDARD_LOCI: tuple[str, ...] = ("A", "B", "C", "DRB1", "DQB1")

# ---------------------------------------------------------------------------
# Antigen → allele-group (field1) mapping
# ---------------------------------------------------------------------------

#: Maps serological antigen name → two-digit first field (zero-padded).
#:
#: A9, B12, etc. are "broad" antigens that have been split into more specific
#: antigens; they are included for backwards compatibility with old records.
ANTIGEN_TO_FIELD1: dict[str, str] = {
    # --- HLA-A ---
    "A1": "01", "A2": "02", "A3": "03",
    "A9": "09",  # broad (→ A23, A24)
    "A10": "10",  # broad (→ A25, A26, A34, A66)
    "A11": "11", "A19": "19",  # broad (→ A29-A33)
    "A23": "23", "A24": "24", "A25": "25", "A26": "26",
    "A28": "28",  # broad (→ A68, A69)
    "A29": "29", "A30": "30", "A31": "31", "A32": "32",
    "A33": "33", "A34": "34", "A36": "36", "A43": "43",
    "A66": "66", "A68": "68", "A69": "69", "A74": "74", "A80": "80",
    # --- HLA-B ---
    "B5": "05",  # broad (→ B51, B52)
    "B7": "07", "B8": "08",
    "B12": "12",  # broad (→ B44, B45)
    "B13": "13",
    "B14": "14",  # broad (→ B64, B65)
    "B15": "15",
    "B16": "16",  # broad (→ B38, B39)
    "B17": "17",  # broad (→ B57, B58)
    "B18": "18",
    "B21": "21",  # broad (→ B49, B50)
    "B22": "22",  # broad (→ B54, B55, B56)
    "B27": "27", "B35": "35", "B37": "37",
    "B38": "38", "B39": "39",
    "B40": "40",  # broad (→ B60, B61)
    "B41": "41", "B42": "42", "B44": "44", "B45": "45",
    "B46": "46", "B47": "47", "B48": "48", "B49": "49",
    "B50": "50", "B51": "51", "B52": "52", "B53": "53",
    "B54": "54", "B55": "55", "B56": "56", "B57": "57",
    "B58": "58", "B59": "59",
    "B60": "40",  # B60 = B*40:01
    "B61": "40",  # B61 = B*40:02
    "B62": "15",  # B62 = B*15:01
    "B63": "15",  # B63 = B*15:03
    "B64": "14",  # B64 = B*14:01
    "B65": "14",  # B65 = B*14:02
    "B67": "67", "B71": "15", "B72": "15",
    "B73": "73", "B75": "15", "B76": "15", "B77": "15",
    "B78": "15",
    # --- HLA-C (serological "Cw" prefix) ---
    "Cw1": "01", "Cw2": "02", "Cw3": "03", "Cw4": "04",
    "Cw5": "05", "Cw6": "06", "Cw7": "07", "Cw8": "08",
    "Cw9": "03",  # Cw9 ≈ C*03:03 in modern nomenclature
    "Cw10": "03",  # Cw10 ≈ C*03:04
    "Cw12": "12", "Cw14": "14", "Cw15": "15",
    "Cw16": "12",  # Cw16 ≈ C*12:02
    "Cw17": "17", "Cw18": "18",
    # --- HLA-DRB1 ("DR" prefix) ---
    "DR1": "01",
    "DR2": "15",   # broad → DR15 (most common split)
    "DR3": "03",   # broad → DR17/DR18; DR3 = DRB1*03 group
    "DR4": "04", "DR5": "11",  # broad → DR11/DR12
    "DR6": "13",   # broad → DR13/DR14
    "DR7": "07", "DR8": "08", "DR9": "09", "DR10": "10",
    "DR11": "11", "DR12": "12", "DR13": "13", "DR14": "14",
    "DR15": "15", "DR16": "16",
    "DR17": "03",  # DR17 is a split of DR3
    "DR18": "03",  # DR18 is a split of DR3
    "DR51": "51", "DR52": "52", "DR53": "53",
    # --- HLA-DQB1 ("DQ" prefix) ---
    "DQ1": "05",   # broad → DQ5/DQ6 (DQ5 most common)
    "DQ2": "02", "DQ3": "07",  # broad → DQ7/DQ8/DQ9
    "DQ4": "04", "DQ5": "05", "DQ6": "06",
    "DQ7": "03", "DQ8": "03", "DQ9": "03",  # DQ7/8/9 = DQB1*03:xx
    # --- HLA-DPB1 ("DP" prefix) ---
    "DP1": "01", "DP2": "02", "DP3": "03", "DP4": "04",
    "DP5": "05", "DP6": "06", "DP8": "08", "DP9": "09",
    "DP10": "10", "DP11": "11", "DP13": "13",
}

# ---------------------------------------------------------------------------
# Antigen → most-common representative allele (for ESM-2 fallback)
# ---------------------------------------------------------------------------

#: Maps serological antigen name → best two-field allele string for sequence
#: lookup when only antigen-level typing is available.
#:
#: Source: NMDP/WMDA haplotype frequency data, predominantly Northern-European.
#: Frequencies vary by ethnicity; treat this as a reasonable fallback only.
ANTIGEN_TO_COMMON_ALLELE: dict[str, str] = {
    # --- HLA-A ---
    "A1": "A*01:01", "A2": "A*02:01", "A3": "A*03:01",
    "A9": "A*23:01",  # broad; A23 is most frequent split globally
    "A10": "A*26:01",  # broad
    "A11": "A*11:01", "A19": "A*29:02",  # broad
    "A23": "A*23:01", "A24": "A*24:02", "A25": "A*25:01",
    "A26": "A*26:01",
    "A28": "A*68:01",  # broad; A68 is most frequent split
    "A29": "A*29:02", "A30": "A*30:01", "A31": "A*31:01",
    "A32": "A*32:01", "A33": "A*33:01", "A34": "A*34:01",
    "A36": "A*36:01", "A43": "A*43:01", "A66": "A*66:01",
    "A68": "A*68:01", "A69": "A*69:01", "A74": "A*74:01",
    "A80": "A*80:01",
    # --- HLA-B ---
    "B5": "B*51:01",  # broad; B51 is most frequent split
    "B7": "B*07:02", "B8": "B*08:01",
    "B12": "B*44:02",  # broad; B44 is most frequent split
    "B13": "B*13:01",
    "B14": "B*14:01",  # broad; B64 = B*14:01
    "B15": "B*15:01",
    "B16": "B*38:01",  # broad; B38 is most frequent split
    "B17": "B*57:01",  # broad; B57 is most frequent split
    "B18": "B*18:01",
    "B21": "B*49:01",  # broad
    "B22": "B*55:01",  # broad
    "B27": "B*27:05",  # B*27:05 most common in Europeans
    "B35": "B*35:01", "B37": "B*37:01",
    "B38": "B*38:01", "B39": "B*39:01",
    "B40": "B*40:01",  # broad; B60 = B*40:01
    "B41": "B*41:01", "B42": "B*42:01",
    "B44": "B*44:02", "B45": "B*45:01",
    "B46": "B*46:01", "B47": "B*47:01", "B48": "B*48:01",
    "B49": "B*49:01", "B50": "B*50:01", "B51": "B*51:01",
    "B52": "B*52:01", "B53": "B*53:01", "B54": "B*54:01",
    "B55": "B*55:01", "B56": "B*56:01", "B57": "B*57:01",
    "B58": "B*58:01", "B59": "B*59:01",
    "B60": "B*40:01", "B61": "B*40:02", "B62": "B*15:01",
    "B63": "B*15:03", "B64": "B*14:01", "B65": "B*14:02",
    "B67": "B*67:01", "B71": "B*15:11", "B72": "B*15:12",
    "B73": "B*73:01", "B75": "B*15:02", "B76": "B*15:07",
    "B77": "B*15:08", "B78": "B*15:09",
    # --- HLA-C ---
    "Cw1": "C*01:02", "Cw2": "C*02:02", "Cw3": "C*03:03",
    "Cw4": "C*04:01", "Cw5": "C*05:01", "Cw6": "C*06:02",
    "Cw7": "C*07:01", "Cw8": "C*08:02",
    "Cw9": "C*03:03",  # Cw9 ≈ C*03:03
    "Cw10": "C*03:04",  # Cw10 ≈ C*03:04
    "Cw12": "C*12:03", "Cw14": "C*14:02", "Cw15": "C*15:02",
    "Cw16": "C*12:02", "Cw17": "C*17:01", "Cw18": "C*18:01",
    # --- HLA-DRB1 ---
    "DR1": "DRB1*01:01",
    "DR2": "DRB1*15:01",   # broad; DR15 (DRB1*15:01) most common split
    "DR3": "DRB1*03:01",   # broad; DR17 = DRB1*03:01
    "DR4": "DRB1*04:01",
    "DR5": "DRB1*11:01",   # broad; DR11 most common split
    "DR6": "DRB1*13:01",   # broad
    "DR7": "DRB1*07:01", "DR8": "DRB1*08:01",
    "DR9": "DRB1*09:01", "DR10": "DRB1*10:01",
    "DR11": "DRB1*11:01", "DR12": "DRB1*12:01",
    "DR13": "DRB1*13:01", "DR14": "DRB1*14:01",
    "DR15": "DRB1*15:01", "DR16": "DRB1*16:01",
    "DR17": "DRB1*03:01",  # split of DR3
    "DR18": "DRB1*03:02",  # split of DR3
    "DR51": "DRB5*01:01",
    "DR52": "DRB3*01:01",
    "DR53": "DRB4*01:03",
    # --- HLA-DQB1 ---
    "DQ1": "DQB1*05:01",  # broad; DQ5 most common split
    "DQ2": "DQB1*02:01",
    "DQ3": "DQB1*03:01",  # broad
    "DQ4": "DQB1*04:02",
    "DQ5": "DQB1*05:01", "DQ6": "DQB1*06:02",
    "DQ7": "DQB1*03:01", "DQ8": "DQB1*03:02", "DQ9": "DQB1*03:03",
    # --- HLA-DPB1 ---
    "DP1": "DPB1*01:01", "DP2": "DPB1*02:01",
    "DP3": "DPB1*03:01",
    "DP4": "DPB1*04:01",  # most common in Europeans (~50% frequency)
    "DP5": "DPB1*05:01", "DP6": "DPB1*06:01",
    "DP8": "DPB1*08:01", "DP9": "DPB1*09:01",
    "DP10": "DPB1*10:01", "DP11": "DPB1*11:01",
    "DP13": "DPB1*13:01",
}

# ---------------------------------------------------------------------------
# HLAgrI mismatch-type classification (UCI dataset column 21)
# ---------------------------------------------------------------------------

#: Maps the integer HLAgrI code used in the UCI dataset to a human-readable label.
MISMATCH_TYPE_LABELS: dict[int, str] = {
    0: "matched",
    1: "antigen_diff_only",    # difference is only at antigen level
    2: "allele_diff_only",     # difference is only at allele level (below antigen)
    3: "DRB1_diff_only",       # the single difference is in the DRB1 locus
    4: "two_diffs_same_type",  # two differences, both antigen or both allele
    5: "two_diffs_mixed_type", # two differences of different types
    7: "complex",              # ≥3 differences or unclassified (undocumented in ARFF)
}

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# WHO allele-level: "A*02:01" or "DRB1*15:01:01:02" or "C*07:01"
_WHO_RE = re.compile(
    r"^(?:HLA-)?(?P<gene>[A-Z][A-Z0-9]*)\*"
    r"(?P<f1>\d{2,3})"
    r"(?::(?P<f2>\d{2,3}))?"
    r"(?::(?P<f3>\d{2,3}))?"
    r"(?::(?P<f4>\d{2,3}))?$"
)

# Serological (antigen-level): "A2", "Cw7", "DR15", "DQ5", "B27"
# Gene prefix = one or more letters (incl. 'w'), antigen number = digits
_SERO_RE = re.compile(
    r"^(?:HLA-)?(?P<prefix>[A-Za-z]+[wW]?)(?P<number>\d+)$"
)


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HLAAllele:
    """Structured representation of a single HLA allele.

    Attributes
    ----------
    gene : str
        Canonical WHO gene name, e.g. ``"A"``, ``"DRB1"``, ``"C"``.
    field1 : str
        First field — allele group (zero-padded two or three digits).
        For serological-level input this is the only field set.
    field2 : str | None
        Second field — specific HLA protein, e.g. ``"01"``.
    field3 : str | None
        Third field — synonymous DNA substitutions (rare).
    field4 : str | None
        Fourth field — non-coding differences (very rare).
    source_notation : str
        The original notation used to produce this allele:
        ``"who"`` for ``"A*02:01"`` or ``"serological"`` for ``"A2"``.
    """

    gene: str
    field1: str
    field2: str | None = None
    field3: str | None = None
    field4: str | None = None
    source_notation: str = "who"

    @property
    def resolution(self) -> str:
        """Return the resolution level: ``"antigen"`` or ``"allele"``."""
        return "antigen" if self.field2 is None else "allele"

    @property
    def two_field(self) -> str:
        """Return two-field string, e.g. ``"A*02:01"``.

        If only first-field resolution is available (serological input),
        returns ``"A*02"`` (no second field).
        """
        if self.field2 is None:
            return f"{self.gene}*{self.field1}"
        return f"{self.gene}*{self.field1}:{self.field2}"

    @property
    def common_allele(self) -> str | None:
        """Return the most-common representative two-field allele string.

        Only meaningful when :attr:`source_notation` is ``"serological"``
        and full allele resolution is unavailable.  Returns ``None`` for
        WHO-notation input (the allele itself *is* the representative).
        """
        if self.source_notation == "serological":
            # Reconstruct the serological antigen key to look up the table
            prefix = _gene_to_serological_prefix(self.gene)
            key = f"{prefix}{int(self.field1)}"
            return ANTIGEN_TO_COMMON_ALLELE.get(key)
        return None

    def __str__(self) -> str:
        parts = [f"{self.gene}*{self.field1}"]
        for f in (self.field2, self.field3, self.field4):
            if f is None:
                break
            parts.append(f)
        return ":".join([parts[0]] + parts[1:])


@dataclass
class HLAProfile:
    """Complete HLA typing for one individual (donor or recipient).

    Holds up to two alleles per locus (heterozygous typing).  Missing loci
    are represented as empty lists.

    Attributes
    ----------
    alleles : dict[str, list[HLAAllele]]
        Maps canonical locus name → list of alleles (1 or 2 elements).
    role : str
        ``"donor"`` or ``"recipient"`` (informational).
    """

    alleles: dict[str, list[HLAAllele]] = field(default_factory=dict)
    role: str = "unknown"

    @property
    def typed_loci(self) -> list[str]:
        """Return list of loci for which at least one allele is known."""
        return [loc for loc, als in self.alleles.items() if als]

    def get_locus(self, locus: str) -> list[HLAAllele]:
        """Return alleles at *locus*, or ``[]`` if not typed.

        Parameters
        ----------
        locus : str
            Canonical locus name, e.g. ``"A"``, ``"DRB1"``.

        Returns
        -------
        list[HLAAllele]
        """
        canon = CANONICAL_GENE.get(locus, locus)
        return self.alleles.get(canon, [])

    def first_allele(self, locus: str) -> HLAAllele | None:
        """Return the first allele at *locus*, or ``None`` if not typed."""
        als = self.get_locus(locus)
        return als[0] if als else None

    def standard_loci_coverage(self) -> dict[str, bool]:
        """Return coverage status for each of the five standard HSCT loci."""
        return {loc: bool(self.get_locus(loc)) for loc in STANDARD_LOCI}


@dataclass(frozen=True)
class HLAMismatchSummary:
    """Structured representation of the aggregate HLA mismatch data from the UCI dataset.

    The UCI BMT dataset does not contain per-locus allele strings.  This
    class captures the aggregate mismatch information encoded in the five
    HLA-related columns of the cleaned DataFrame.

    Attributes
    ----------
    match_grade : str
        Overall match grade: ``"10/10"``, ``"9/10"``, ``"8/10"``, or ``"7/10"``.
    mismatched : bool
        ``True`` if any HLA mismatch is present.
    n_antigen_mismatches : int
        Number of antigen-level mismatches (0–3).
    n_allele_mismatches : int
        Number of allele-level mismatches (0–4).
    total_mismatches : int
        Total number of mismatched loci (0–3), derived from ``match_grade``.
    mismatch_type : str
        Human-readable mismatch classification from :data:`MISMATCH_TYPE_LABELS`.
    mismatch_type_code : int
        Raw UCI ``HLAgrI`` code (0–7).
    """

    match_grade: str
    mismatched: bool
    n_antigen_mismatches: int
    n_allele_mismatches: int
    total_mismatches: int
    mismatch_type: str
    mismatch_type_code: int

    @property
    def is_fully_matched(self) -> bool:
        """``True`` if donor and recipient are 10/10 HLA-matched."""
        return self.match_grade == "10/10"


# ---------------------------------------------------------------------------
# Public parsing functions
# ---------------------------------------------------------------------------


def normalize_gene(raw: str) -> str:
    """Normalise a raw HLA gene token to its canonical WHO name.

    Parameters
    ----------
    raw : str
        Raw gene token, e.g. ``"Cw"``, ``"DR"``, ``"DQ"``.

    Returns
    -------
    str
        Canonical gene name, e.g. ``"C"``, ``"DRB1"``, ``"DQB1"``.

    Raises
    ------
    ValueError
        If the gene token is not recognised.
    """
    canonical = CANONICAL_GENE.get(raw)
    if canonical is None:
        raise ValueError(
            f"Unknown HLA gene token: {raw!r}. "
            f"Known tokens: {sorted(CANONICAL_GENE)}"
        )
    return canonical


def parse_who_allele(raw: str) -> HLAAllele:
    """Parse a WHO allele-level string, e.g. ``"A*02:01"`` or ``"DRB1*15:01:01:02"``.

    Parameters
    ----------
    raw : str
        Raw allele string in WHO format (asterisk notation).
        An optional ``"HLA-"`` prefix is accepted.

    Returns
    -------
    HLAAllele
        Parsed allele with ``source_notation="who"``.

    Raises
    ------
    ValueError
        If the string does not match WHO allele notation.
    """
    m = _WHO_RE.match(raw.strip())
    if m is None:
        raise ValueError(f"Not a valid WHO allele string: {raw!r}")

    raw_gene = m.group("gene")
    gene = CANONICAL_GENE.get(raw_gene, raw_gene)  # keep as-is if not in map

    return HLAAllele(
        gene=gene,
        field1=m.group("f1"),
        field2=m.group("f2"),
        field3=m.group("f3"),
        field4=m.group("f4"),
        source_notation="who",
    )


def parse_serological_allele(raw: str) -> HLAAllele:
    """Parse a serological (antigen-level) HLA string, e.g. ``"A2"`` or ``"DR15"``.

    The antigen number is mapped to a two-digit first field.  The result has
    ``field2=None`` (antigen-level resolution only) and
    ``source_notation="serological"``.

    Parameters
    ----------
    raw : str
        Serological antigen string.  The ``"HLA-"`` prefix is optional.
        Case-insensitive for the gene prefix.

    Returns
    -------
    HLAAllele
        Parsed allele with antigen-level resolution.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a serological antigen.
    """
    raw_stripped = raw.strip()
    m = _SERO_RE.match(raw_stripped)
    if m is None:
        raise ValueError(f"Not a valid serological HLA string: {raw!r}")

    raw_prefix = m.group("prefix")
    antigen_number = m.group("number")

    # Normalise: match case to known keys
    # Try original case first, then title-case, then upper
    prefix_candidates = [
        raw_prefix,
        raw_prefix.upper(),
        raw_prefix[0].upper() + raw_prefix[1:].lower() if len(raw_prefix) > 1 else raw_prefix.upper(),
    ]
    gene: str | None = None
    antigen_key: str | None = None
    for pfx in prefix_candidates:
        candidate = f"{pfx}{antigen_number}"
        if candidate in ANTIGEN_TO_FIELD1:
            gene = CANONICAL_GENE.get(pfx, pfx)
            antigen_key = candidate
            break

    if gene is None or antigen_key is None:
        raise ValueError(
            f"Unrecognised serological antigen: {raw!r}. "
            f"If this is a valid antigen, add it to ANTIGEN_TO_FIELD1."
        )

    field1 = ANTIGEN_TO_FIELD1[antigen_key]
    return HLAAllele(
        gene=gene,
        field1=field1,
        source_notation="serological",
    )


def parse_hla_string(raw: str) -> HLAAllele:
    """Parse any HLA string — WHO allele-level or serological antigen-level.

    Tries WHO notation first (requires an asterisk); falls back to
    serological notation.

    Parameters
    ----------
    raw : str
        HLA string in any supported format.

    Returns
    -------
    HLAAllele

    Raises
    ------
    ValueError
        If the string matches neither format.
    """
    raw = raw.strip()
    if "*" in raw:
        return parse_who_allele(raw)
    try:
        return parse_serological_allele(raw)
    except ValueError:
        raise ValueError(
            f"Cannot parse HLA string {raw!r}: "
            f"not WHO notation (no '*') and not a recognised serological antigen."
        )


def parse_hla_typing(
    typing: dict[str, str | list[str]],
    *,
    role: str = "unknown",
) -> HLAProfile:
    """Build an :class:`HLAProfile` from a locus→allele-string dict.

    Supports:
    * Single allele per locus: ``{"A": "A*02:01"}``
    * Two alleles per locus:   ``{"A": ["A*02:01", "A*24:02"]}``
    * Slash-separated string:  ``{"A": "A*02:01/A*24:02"}``
    * Mixed notations per key.

    Parameters
    ----------
    typing : dict[str, str | list[str]]
        Mapping of locus identifier → allele string(s).
        Locus keys are normalised via :data:`CANONICAL_GENE`.
    role : str
        ``"donor"`` or ``"recipient"`` (stored for reference).

    Returns
    -------
    HLAProfile
    """
    alleles: dict[str, list[HLAAllele]] = {}
    for raw_locus, value in typing.items():
        canon_locus = CANONICAL_GENE.get(raw_locus, raw_locus)

        # Collect raw allele strings
        if isinstance(value, list):
            raw_strings = value
        elif isinstance(value, str) and "/" in value:
            raw_strings = [s.strip() for s in value.split("/")]
        else:
            raw_strings = [str(value)]

        parsed: list[HLAAllele] = []
        for s in raw_strings:
            if not s:
                continue
            allele = parse_hla_string(s)
            # Ensure the gene in the string matches the locus key, if both are present
            if allele.gene != canon_locus and canon_locus in STANDARD_LOCI:
                logger.warning(
                    "Locus key %r (→ %r) does not match allele gene %r in %r; "
                    "using gene from allele string.",
                    raw_locus, canon_locus, allele.gene, s,
                )
            parsed.append(allele)

        if parsed:
            alleles[canon_locus] = parsed

    return HLAProfile(alleles=alleles, role=role)


def parse_uci_hla_columns(row: dict[str, object] | pd.Series) -> HLAMismatchSummary:
    """Build an :class:`HLAMismatchSummary` from a row of the cleaned UCI DataFrame.

    Uses the five HLA-related columns produced by :func:`capa.data.loader.load_bmt`:
    ``hla_match_score``, ``hla_mismatched``, ``n_antigen_mismatches``,
    ``n_allele_mismatches``, ``hla_mismatch_type``.

    Parameters
    ----------
    row : dict[str, object] | pd.Series
        A single row from the cleaned UCI BMT DataFrame (or a plain dict with
        the same keys, e.g. for testing).

    Returns
    -------
    HLAMismatchSummary

    Notes
    -----
    The UCI dataset encodes antigen/allele mismatch counts with a −1 sentinel
    for zero mismatches (see :mod:`capa.data.loader`).  This function
    translates them back to actual counts:
    ``−1 → 0, 0 → 1, 1 → 2, 2 → 3, 3 → 4``.

    The ``HLAgrI`` column value ``7`` is undocumented in the ARFF header but
    appears in the data; it is labelled ``"complex"`` here.
    """
    def _get(key: str) -> object:
        if isinstance(row, dict):
            return row[key]
        return row[key]

    # hla_match_score: 0 = 10/10, 1 = 9/10, 2 = 8/10, 3 = 7/10
    match_score_raw = int(_get("hla_match_score"))
    match_grade_map = {0: "10/10", 1: "9/10", 2: "8/10", 3: "7/10"}
    match_grade = match_grade_map.get(match_score_raw, f"?/{match_score_raw}")
    total_mismatches = match_score_raw  # 0 mismatches → 10/10, etc.

    mismatched = bool(int(_get("hla_mismatched")))

    # UCI sentinel: -1 → 0, 0 → 1, 1 → 2, 2 → 3, 3 → 4
    def _decode_mismatch_count(v: object) -> int:
        val = int(v)  # type: ignore[arg-type]
        return val + 1 if val >= 0 else 0

    n_antigen = _decode_mismatch_count(_get("n_antigen_mismatches"))
    n_allele = _decode_mismatch_count(_get("n_allele_mismatches"))

    mismatch_type_code = int(_get("hla_mismatch_type"))
    mismatch_type = MISMATCH_TYPE_LABELS.get(mismatch_type_code, f"unknown_{mismatch_type_code}")

    return HLAMismatchSummary(
        match_grade=match_grade,
        mismatched=mismatched,
        n_antigen_mismatches=n_antigen,
        n_allele_mismatches=n_allele,
        total_mismatches=total_mismatches,
        mismatch_type=mismatch_type,
        mismatch_type_code=mismatch_type_code,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gene_to_serological_prefix(gene: str) -> str:
    """Return the canonical serological prefix for a gene name.

    Used internally to reconstruct antigen keys for
    :data:`ANTIGEN_TO_COMMON_ALLELE` lookup.

    Parameters
    ----------
    gene : str
        Canonical gene name, e.g. ``"A"``, ``"C"``, ``"DRB1"``.

    Returns
    -------
    str
        Serological prefix, e.g. ``"A"``, ``"Cw"``, ``"DR"``.
    """
    prefix_map = {
        "A": "A", "B": "B", "C": "Cw",
        "DRB1": "DR", "DRB3": "DR", "DRB4": "DR", "DRB5": "DR",
        "DQB1": "DQ", "DPB1": "DP",
    }
    return prefix_map.get(gene, gene)


# Keep the old name as an alias for backwards compatibility with the scaffold
parse_allele = parse_who_allele
