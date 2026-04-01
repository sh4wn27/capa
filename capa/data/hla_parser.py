"""Parse HLA typing strings into structured allele representations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Regex for standard HLA allele notation, e.g. "A*02:01" or "DRB1*15:01:01:02"
_ALLELE_RE = re.compile(
    r"^(?P<gene>[A-Z0-9]+)\*(?P<field1>\d{2,3})(?::(?P<field2>\d{2,3}))?(?::(?P<field3>\d{2,3}))?(?::(?P<field4>\d{2,3}))?$"
)


@dataclass(frozen=True)
class HLAAllele:
    """Structured representation of a single HLA allele.

    Attributes
    ----------
    gene : str
        HLA gene locus, e.g. ``"A"``, ``"DRB1"``.
    field1 : str
        First field (allele group), e.g. ``"02"``.
    field2 : str | None
        Second field (specific HLA protein), e.g. ``"01"``.
    field3 : str | None
        Third field (synonymous DNA substitutions), if present.
    field4 : str | None
        Fourth field (non-coding differences), if present.
    """

    gene: str
    field1: str
    field2: str | None = None
    field3: str | None = None
    field4: str | None = None

    @property
    def two_field(self) -> str:
        """Return two-field resolution string, e.g. ``"A*02:01"``."""
        if self.field2 is None:
            return f"{self.gene}*{self.field1}"
        return f"{self.gene}*{self.field1}:{self.field2}"

    def __str__(self) -> str:
        parts = [self.gene + "*" + self.field1]
        for f in (self.field2, self.field3, self.field4):
            if f is None:
                break
            parts.append(f)
        return ":".join([parts[0]] + parts[1:])


def parse_allele(raw: str) -> HLAAllele:
    """Parse a raw HLA allele string into an :class:`HLAAllele`.

    Parameters
    ----------
    raw : str
        Raw allele string, e.g. ``"A*02:01"`` or ``"DRB1*15:01:01"``.

    Returns
    -------
    HLAAllele
        Parsed allele object.

    Raises
    ------
    ValueError
        If ``raw`` does not match standard HLA allele notation.
    """
    raw = raw.strip()
    m = _ALLELE_RE.match(raw)
    if m is None:
        raise ValueError(f"Cannot parse HLA allele string: {raw!r}")
    return HLAAllele(
        gene=m.group("gene"),
        field1=m.group("field1"),
        field2=m.group("field2"),
        field3=m.group("field3"),
        field4=m.group("field4"),
    )


def parse_hla_typing(typing: dict[str, str]) -> dict[str, HLAAllele]:
    """Parse a donor or recipient HLA typing dict.

    Parameters
    ----------
    typing : dict[str, str]
        Mapping of locus name to raw allele string,
        e.g. ``{"A": "A*02:01", "DRB1": "DRB1*15:01"}``.

    Returns
    -------
    dict[str, HLAAllele]
        Mapping of locus name to parsed :class:`HLAAllele`.
    """
    return {locus: parse_allele(allele) for locus, allele in typing.items()}
