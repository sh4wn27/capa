"""IPD-IMGT/HLA allele → protein sequence lookup with progressive fallback.

The JSON database is produced by ``scripts/download_hla_seqs.py`` and has the
layout::

    {
      "_meta": { "source_url": "...", "n_alleles": 44630, ... },
      "A*01:01:01:01": "MAVMAPRTL...",
      "A*01:01:01:02N": "MAVMAPRTL...",
      ...
    }

Lookup strategy
---------------
:meth:`HLASequenceDB.get` accepts an :class:`~capa.data.hla_parser.HLAAllele`
and applies three fallback levels:

1. **Exact match** — look up ``str(allele)`` (all fields present in the
   allele object).
2. **Field truncation** — progressively drop the most-specific field, e.g.
   ``A*02:01:01:02`` → ``A*02:01:01`` → ``A*02:01``.  At each step, try an
   exact dictionary hit first, then scan the *prefix index*.
3. **Group fallback** — look up any allele whose name starts with
   ``<gene>*<field1>:`` (for a two-field allele like ``A*02:01``) or
   ``<gene>*<field1>`` (for a single-field serological input like ``A*02``).

At steps 2–3, the *best* match is chosen from all prefix candidates:

* Non-null alleles (no expression-status suffix such as N, L, S, A, Q) are
  preferred over null/aberrant alleles.
* Among equally-ranked candidates, lexicographic order is used so that the
  result is deterministic.

All fallbacks log at ``WARNING`` level so the caller is aware of the
resolution change.

Prefix index
------------
Built at load time in O(n) — for every full allele key, all strict prefix
truncations (e.g., ``A*02:01:01`` for ``A*02:01:01:04``) are added to the
index.  Lookup is then O(1) for the prefix search itself; choosing the best
match is O(k log k) in the size of the candidate list (usually small).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from capa.data.hla_parser import HLAAllele

logger = logging.getLogger(__name__)

# Expression-status suffixes in the IMGT/HLA database:
#   N = null (no expression), L = low expression, S = secreted,
#   A = aberrant expression, Q = questionable expression
_EXPRESSION_SUFFIX_RE = re.compile(r"[NLSAQ]$")

# Characters that separate allele fields in the standard notation
_FIELD_SEP = ":"


def _is_expressed(allele_name: str) -> bool:
    """Return ``True`` if the allele has no expression-status suffix (N/L/S/A/Q).

    Null and aberrant alleles often have partial/unusual sequences; the normal
    (expressed) allele at the same resolution is preferred for ESM-2 input.

    Parameters
    ----------
    allele_name : str
        Allele name as stored in the database, e.g. ``"A*01:01:01:02N"``.

    Returns
    -------
    bool
    """
    # Strip the gene prefix and asterisk, then check for suffix
    after_star = allele_name.split("*")[-1] if "*" in allele_name else allele_name
    return not bool(_EXPRESSION_SUFFIX_RE.search(after_star))


def _candidate_keys(allele: HLAAllele) -> list[str]:
    """Return lookup keys for *allele* from most- to least-specific.

    Parameters
    ----------
    allele : HLAAllele
        Allele to generate keys for.

    Returns
    -------
    list[str]
        E.g. for ``DRB1*15:01:01:02``:
        ``["DRB1*15:01:01:02", "DRB1*15:01:01", "DRB1*15:01", "DRB1*15"]``
    """
    present_fields = [
        f for f in (allele.field1, allele.field2, allele.field3, allele.field4)
        if f is not None
    ]
    keys: list[str] = []
    while present_fields:
        key = f"{allele.gene}*{_FIELD_SEP.join(present_fields)}"
        keys.append(key)
        present_fields.pop()
    return keys


def _best_from_candidates(candidates: list[str]) -> str:
    """Choose the best allele name from a list of candidates.

    Preference order:
    1. Expressed alleles (no N/L/S/A/Q suffix)
    2. Lexicographically first (for determinism)

    Parameters
    ----------
    candidates : list[str]
        Non-empty list of full allele names.

    Returns
    -------
    str
        The best candidate name.
    """
    expressed = [c for c in candidates if _is_expressed(c)]
    pool = expressed if expressed else candidates
    return min(pool)


class HLASequenceDB:
    """In-memory lookup from HLA allele to protein sequence with fallback.

    Parameters
    ----------
    sequences_path : Path
        Path to the JSON file produced by ``scripts/download_hla_seqs.py``.
        The file maps allele names to sequences at the top level, with an
        optional ``"_meta"`` key for provenance information.
    """

    def __init__(self, sequences_path: Path) -> None:
        self._path = sequences_path
        # Primary store: exact allele name → sequence
        self._db: dict[str, str] = {}
        # Prefix index: truncated key → list of exact allele names
        # e.g. "A*02:01:01" → ["A*02:01:01:01", "A*02:01:01:02", ...]
        self._prefix_index: dict[str, list[str]] = {}
        # Metadata from the JSON _meta key (if present)
        self.meta: dict[str, object] = {}

        self._load()
        self._build_prefix_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, allele: HLAAllele) -> str:
        """Return the protein sequence for *allele* with progressive fallback.

        Parameters
        ----------
        allele : HLAAllele
            Parsed allele object from :mod:`capa.data.hla_parser`.

        Returns
        -------
        str
            Amino acid sequence in single-letter code (IUPAC, may include X).

        Raises
        ------
        KeyError
            If no sequence can be found at any resolution for this allele.
        """
        for i, key in enumerate(_candidate_keys(allele)):
            seq = self._lookup_key(key)
            if seq is not None:
                if i > 0:
                    logger.warning(
                        "Allele %r not found; returned sequence for %r (fallback level %d)",
                        str(allele),
                        key,
                        i,
                    )
                return seq

        raise KeyError(
            f"No sequence found for {allele!r} at any resolution "
            f"(tried: {_candidate_keys(allele)})"
        )

    def get_by_name(self, name: str) -> str:
        """Return the protein sequence for a raw allele name string.

        Applies the same progressive fallback as :meth:`get`.

        Parameters
        ----------
        name : str
            Raw allele name, e.g. ``"A*02:01"`` or ``"A*02:01:01:99"``.

        Returns
        -------
        str
            Amino acid sequence.

        Raises
        ------
        KeyError
            If no sequence can be found.
        """
        # Build keys by progressively truncating fields
        if "*" not in name:
            raise KeyError(f"Not a valid allele name (missing '*'): {name!r}")

        gene, rest = name.split("*", 1)
        # Strip any expression-status suffix from the last field for lookup purposes
        rest_clean = _EXPRESSION_SUFFIX_RE.sub("", rest) if _EXPRESSION_SUFFIX_RE.search(rest) else rest

        fields = rest_clean.split(_FIELD_SEP)
        tried: list[str] = []

        # First try the original name as-is (may include suffix)
        original_seq = self._lookup_key(name)
        if original_seq is not None:
            return original_seq

        # Then try progressive truncation on the cleaned name
        while fields:
            key = f"{gene}*{_FIELD_SEP.join(fields)}"
            tried.append(key)
            seq = self._lookup_key(key)
            if seq is not None:
                if key != name:
                    logger.warning(
                        "Allele %r not found; returned sequence for %r",
                        name,
                        key,
                    )
                return seq
            fields.pop()

        raise KeyError(
            f"No sequence found for {name!r} (tried: {tried})"
        )

    def contains(self, name: str) -> bool:
        """Return ``True`` if the database has an exact entry for *name*.

        Parameters
        ----------
        name : str
            Allele name, e.g. ``"A*02:01"``.

        Returns
        -------
        bool
        """
        return name in self._db

    @property
    def allele_names(self) -> list[str]:
        """Return a sorted list of all exact allele names in the database."""
        return sorted(self._db)

    def locus_names(self) -> list[str]:
        """Return sorted list of unique gene/locus names in the database."""
        genes: set[str] = set()
        for name in self._db:
            if "*" in name:
                genes.add(name.split("*")[0])
        return sorted(genes)

    def alleles_for_locus(self, gene: str) -> list[str]:
        """Return all allele names for *gene* (e.g. ``"A"``).

        Parameters
        ----------
        gene : str
            Gene/locus name.

        Returns
        -------
        list[str]
            Sorted list of allele names.
        """
        prefix = f"{gene}*"
        return sorted(k for k in self._db if k.startswith(prefix))

    def __len__(self) -> int:
        """Return the number of alleles in the database."""
        return len(self._db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the JSON file into memory."""
        logger.info("Loading HLA sequence database from %s", self._path)
        with self._path.open(encoding="utf-8") as fh:
            raw: dict[str, object] = json.load(fh)

        if "_meta" in raw:
            self.meta = raw.pop("_meta")  # type: ignore[assignment]
            logger.info(
                "Database metadata: %s alleles, source=%s",
                self.meta.get("n_alleles", "?"),
                self.meta.get("source_url", "?"),
            )

        # Remaining keys are allele name → sequence
        self._db = {k: v for k, v in raw.items() if isinstance(v, str)}
        logger.info("Loaded %d allele sequences", len(self._db))

    def _build_prefix_index(self) -> None:
        """Build the prefix index for truncation-based fallback lookups.

        For every allele ``A*02:01:01:04`` in the database, all strict prefixes
        (``A*02:01:01``, ``A*02:01``, ``A*02``) are added to the index mapping
        to ``"A*02:01:01:04"``.
        """
        index: dict[str, list[str]] = {}
        for full_name in self._db:
            if "*" not in full_name:
                continue
            gene, rest = full_name.split("*", 1)
            # Strip expression suffix for index key construction
            rest_for_index = _EXPRESSION_SUFFIX_RE.sub("", rest)
            parts = rest_for_index.split(_FIELD_SEP)
            # Add all strict prefixes (not the full name itself — that's in _db)
            for n_fields in range(len(parts) - 1, 0, -1):
                prefix_key = f"{gene}*{_FIELD_SEP.join(parts[:n_fields])}"
                index.setdefault(prefix_key, []).append(full_name)

        self._prefix_index = index
        logger.debug(
            "Built prefix index with %d unique prefix keys", len(self._prefix_index)
        )

    def _lookup_key(self, key: str) -> str | None:
        """Try to find a sequence for *key* via exact match or prefix index.

        Returns the sequence string, or ``None`` if not found.

        Parameters
        ----------
        key : str
            Allele key, e.g. ``"A*02:01"`` or ``"A*02:01:01:04"``.

        Returns
        -------
        str | None
        """
        # 1. Exact hit
        if key in self._db:
            return self._db[key]

        # 2. Prefix index — key is a prefix of some stored allele
        if key in self._prefix_index:
            candidates = self._prefix_index[key]
            best = _best_from_candidates(candidates)
            return self._db[best]

        return None
