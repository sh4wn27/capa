"""Download HLA allele protein sequences from the IPD-IMGT/HLA database."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Project root (scripts/ is one level below root)
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download IPD-IMGT/HLA protein sequences")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "hla_sequences.json",
        help="Output path for the allele → sequence JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading HLA sequences → %s", args.out)
    # TODO: Implement download from IPD-IMGT/HLA FASTA files.
    # Reference: https://www.ebi.ac.uk/ipd/imgt/hla/
    # Typical approach:
    #   1. Fetch hla_prot.fasta from the IMGT/HLA FTP release
    #   2. Parse FASTA headers to extract allele names
    #   3. Write {allele_name: sequence} JSON

    placeholder: dict[str, str] = {}
    args.out.write_text(json.dumps(placeholder, indent=2))
    logger.info(
        "Placeholder written (%d alleles). Implement full download to populate.", len(placeholder)
    )


if __name__ == "__main__":
    main()
