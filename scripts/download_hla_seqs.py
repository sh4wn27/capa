"""Download the latest IPD-IMGT/HLA protein sequences and save as JSON.

Source
------
GitHub mirror of the IPD-IMGT/HLA database (updated on every IMGT release,
roughly quarterly):
  https://github.com/ANHIG/IMGTHLA/tree/Latest/fasta

File downloaded
---------------
``hla_prot.fasta`` — protein sequences for ALL HLA loci (A, B, C, DRB1,
DQB1, DPB1, DRB3, DRB4, DRB5, DQA1, DPA1, MICA, MICB, …).

As of the 3.57.0 release the file is ~14 MB containing ~44 000 alleles.

FASTA header format
-------------------
``>HLA:HLA00001 A*01:01:01:01 365 bp``

  * Token 0 (``>HLA:HLA00001``) — internal IMGT numeric identifier (ignored)
  * Token 1 (``A*01:01:01:01``) — canonical allele name (our dictionary key)
  * Token 2+ (``365 bp``) — declared sequence length (ignored; we measure directly)

Expression-status suffixes (N, L, S, A, Q) are kept as part of the key
(e.g., ``A*01:01:01:02N``).  Null alleles (N) may have truncated or
unusual sequences; the HLASequenceDB handles them gracefully via prefix
fallback.

Output JSON layout
------------------
All allele entries sit at the top level of the JSON object.  A single
``_meta`` key (underscore prefix distinguishes it from allele names, which
always contain ``*``) stores provenance metadata::

    {
      "_meta": {
        "source_url": "https://...",
        "downloaded_at": "2026-04-01T17:30:00",
        "n_alleles": 44630,
        "loci": ["A", "B", ...]
      },
      "A*01:01:01:01": "MAVMAPRTL...",
      "A*01:01:01:02N": "MAVMAPRTL...",
      ...
    }

Usage
-----
::

    uv run python scripts/download_hla_seqs.py
    uv run python scripts/download_hla_seqs.py --out data/processed/hla_sequences.json
    uv run python scripts/download_hla_seqs.py --loci A B C DRB1 DQB1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

FASTA_URL = (
    "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/hla_prot.fasta"
)

# Regex: extract allele name from a FASTA header line
# Header example: >HLA:HLA00001 A*01:01:01:01 365 bp
_HEADER_RE = re.compile(r"^>HLA:\S+\s+(\S+)")

# Characters that are valid in protein FASTA (IUPAC + X for unknown)
_VALID_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_fasta(url: str, dest: Path) -> None:
    """Stream-download a URL to *dest*, printing a progress dot every 2 MB.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Local file path to write.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    chunk_size = 1024 * 128  # 128 KB
    bytes_written = 0
    dots_printed = 0
    dot_every = 2 * 1024 * 1024  # dot per 2 MB

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        content_length = response.headers.get("Content-Length")
        if content_length:
            logger.info("Expected size: %.1f MB", int(content_length) / 1e6)

        with dest.open("wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                bytes_written += len(chunk)
                while bytes_written >= dot_every * (dots_printed + 1):
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    dots_printed += 1

    if dots_printed:
        sys.stdout.write("\n")
    logger.info("Downloaded %.2f MB", bytes_written / 1e6)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------


def parse_fasta(
    path: Path,
    *,
    loci_filter: set[str] | None = None,
) -> dict[str, str]:
    """Parse a FASTA file into an ``{allele_name: sequence}`` dictionary.

    Parameters
    ----------
    path : Path
        Path to the ``.fasta`` file.
    loci_filter : set[str] | None
        If given, only retain alleles whose gene prefix is in this set
        (e.g. ``{"A", "B", "C", "DRB1", "DQB1"}``).
        ``None`` retains all alleles.

    Returns
    -------
    dict[str, str]
        Mapping from allele name (e.g. ``"A*02:01:01:01"``) to the full
        amino acid sequence as a single uppercase string.

    Notes
    -----
    * Lines that do not start with ``>`` are treated as sequence lines.
    * Stop-codon characters (``*``) are stripped from sequences (rare in
      protein FASTA but present in a handful of partial CDS entries).
    * ``X`` (unknown amino acid) is kept — ESM-2 has a dedicated ``X`` token.
    * Alleles whose sequences are empty after stripping are skipped with a
      warning (this can happen for some null-expression alleles).
    """
    db: dict[str, str] = {}
    current_name: str | None = None
    seq_parts: list[str] = []
    skipped_loci = 0
    skipped_empty = 0

    def _flush() -> None:
        nonlocal skipped_empty
        if current_name is None:
            return
        seq = "".join(seq_parts).upper().replace("*", "")
        seq = _VALID_AA.sub("", seq)  # strip any remaining non-AA characters
        if not seq:
            logger.warning("Skipping %r: empty sequence after cleaning", current_name)
            skipped_empty += 1
            return
        db[current_name] = seq

    with path.open(encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                seq_parts = []
                current_name = None

                m = _HEADER_RE.match(line)
                if m is None:
                    logger.warning("Cannot parse FASTA header: %r", line)
                    continue

                allele_name = m.group(1)

                if loci_filter is not None:
                    # gene = everything before the '*'
                    gene = allele_name.split("*")[0] if "*" in allele_name else ""
                    if gene not in loci_filter:
                        skipped_loci += 1
                        continue

                current_name = allele_name
            elif current_name is not None:
                seq_parts.append(line)

    _flush()  # flush the last record

    logger.info(
        "Parsed %d alleles (skipped %d loci-filtered, %d empty)",
        len(db),
        skipped_loci,
        skipped_empty,
    )
    return db


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def build_output(
    alleles: dict[str, str],
    *,
    source_url: str,
) -> dict[str, object]:
    """Build the final output dict with a ``_meta`` header.

    Parameters
    ----------
    alleles : dict[str, str]
        ``{allele_name: sequence}`` from :func:`parse_fasta`.
    source_url : str
        URL the FASTA was downloaded from (recorded in metadata).

    Returns
    -------
    dict[str, object]
        Combined metadata + allele entries, ready for JSON serialisation.
    """
    locus_counts: Counter[str] = Counter()
    for name in alleles:
        gene = name.split("*")[0] if "*" in name else "unknown"
        locus_counts[gene] += 1

    meta: dict[str, object] = {
        "source_url": source_url,
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_alleles": len(alleles),
        "loci": {gene: count for gene, count in sorted(locus_counts.items())},
    }

    return {"_meta": meta, **alleles}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Download IPD-IMGT/HLA protein sequences (hla_prot.fasta) "
        "and save as a JSON lookup dictionary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "hla_sequences.json",
        help="Destination path for the output JSON.",
    )
    parser.add_argument(
        "--fasta-cache",
        type=Path,
        default=None,
        help="If given, save/load the raw FASTA to/from this path "
        "(avoids re-downloading on re-runs).",
    )
    parser.add_argument(
        "--url",
        default=FASTA_URL,
        help="URL of hla_prot.fasta to download.",
    )
    parser.add_argument(
        "--loci",
        nargs="*",
        default=None,
        metavar="LOCUS",
        help="HLA loci to retain (e.g. A B C DRB1 DQB1). "
        "Omit to keep all loci.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="JSON indentation (None = compact, 2 = pretty-print). "
        "Compact is ~30%% smaller.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``uv run python scripts/download_hla_seqs.py``."""
    args = parse_args(argv)
    loci_filter: set[str] | None = set(args.loci) if args.loci else None

    # Determine where to cache/read the raw FASTA
    fasta_path: Path = args.fasta_cache or (
        PROJECT_ROOT / "data" / "raw" / "hla_prot.fasta"
    )

    if fasta_path.exists():
        logger.info("Using cached FASTA at %s", fasta_path)
    else:
        download_fasta(args.url, fasta_path)

    alleles = parse_fasta(fasta_path, loci_filter=loci_filter)

    if not alleles:
        logger.error("No alleles parsed — the FASTA may be empty or the loci filter too strict.")
        raise SystemExit(1)

    output = build_output(alleles, source_url=args.url)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=args.indent, ensure_ascii=False)

    logger.info(
        "Saved %d alleles → %s (%.1f MB)",
        len(alleles),
        args.out,
        args.out.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
