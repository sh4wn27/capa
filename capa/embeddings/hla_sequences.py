"""IPD-IMGT/HLA allele → protein sequence lookup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HLASequenceDB:
    """In-memory lookup from HLA allele name to protein sequence.

    Parameters
    ----------
    sequences_path : Path
        Path to the JSON file mapping allele names to amino acid sequences.
    """

    def __init__(self, sequences_path: Path) -> None:
        self._path = sequences_path
        self._db: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load sequences from JSON into memory."""
        logger.info("Loading HLA sequences from %s", self._path)
        with self._path.open() as fh:
            self._db = json.load(fh)
        logger.info("Loaded %d HLA allele sequences", len(self._db))

    def get(self, allele: str) -> str:
        """Return the protein sequence for *allele*.

        Parameters
        ----------
        allele : str
            Two- or higher-field allele name, e.g. ``"A*02:01"``.

        Returns
        -------
        str
            Amino acid sequence in single-letter code.

        Raises
        ------
        KeyError
            If the allele is not found in the database.
        """
        if allele in self._db:
            return self._db[allele]
        # Try two-field resolution fallback
        two_field = ":".join(allele.split(":")[:2])
        if two_field in self._db:
            logger.debug("Allele %s not found; falling back to %s", allele, two_field)
            return self._db[two_field]
        raise KeyError(f"HLA allele not found in sequence DB: {allele!r}")

    def __len__(self) -> int:
        return len(self._db)
