"""Compute and cache ESM-2 embeddings for all HLA alleles relevant to the UCI BMT dataset.

Background
----------
The UCI Bone Marrow Transplant dataset records HLA typing at **aggregate** resolution
only — it stores mismatch counts (HLAmatch, Antigen, Alel, HLAgrI) rather than
individual allele names (e.g. "A*02:01") per patient.

To embed the HLA alleles that are clinically relevant to this cohort, this script
derives the allele set from :data:`capa.data.hla_parser.ANTIGEN_TO_COMMON_ALLELE` —
the table that maps each serological antigen (as it would appear in clinical HLA
typing reports for pediatric HSCT) to the most common corresponding WHO allele.
After deduplication, this gives ~90–100 unique two-field alleles across the five
standard loci (A, B, C, DRB1, DQB1).

Workflow
--------
1. Load the UCI BMT dataset to verify the data pipeline is intact.
2. Extract unique alleles for the requested loci from the antigen→allele table.
3. Load the IPD-IMGT/HLA protein sequence database (``hla_sequences.json``).
4. Look up the protein sequence for every allele (with progressive fallback).
5. Check the embedding cache — skip alleles already embedded.
6. Run ESM-2 on the remaining sequences.
7. Write new embeddings to the HDF5 cache.
8. Print a structured report: total alleles, sequence failures, cache hits,
   newly computed, elapsed time.

Usage
-----
::

    # Quick run — embeds ~100 alleles, device auto-detected (CPU on most laptops)
    uv run python scripts/compute_embeddings.py

    # Specific loci only
    uv run python scripts/compute_embeddings.py --loci A B DRB1

    # Force CPU, smaller batch for low-RAM machines
    uv run python scripts/compute_embeddings.py --device cpu --batch-size 4

    # Preview what would be computed (no GPU/model load)
    uv run python scripts/compute_embeddings.py --dry-run

    # Custom paths
    uv run python scripts/compute_embeddings.py \\
        --bmt-path data/raw/bone-marrow.arff \\
        --sequences-path data/processed/hla_sequences.json \\
        --cache-path data/processed/hla_embeddings.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is importable when the script is run directly via
#   uv run python scripts/compute_embeddings.py
# Python replaces sys.path[0] with the script's own directory (scripts/),
# so 'capa' is not findable unless we insert the project root explicitly.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compute_embeddings")

# Project root — two levels up from this file (scripts/ → project root)
_PROJECT_ROOT = Path(__file__).parent.parent

_DEFAULT_BMT = _PROJECT_ROOT / "data" / "raw" / "bone-marrow.arff"
_DEFAULT_SEQS = _PROJECT_ROOT / "data" / "processed" / "hla_sequences.json"
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "processed" / "hla_embeddings.h5"
_DEFAULT_LOCI = ("A", "B", "C", "DRB1", "DQB1")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bmt-path",
        type=Path,
        default=_DEFAULT_BMT,
        help="Path to bone-marrow.arff (UCI BMT dataset).",
    )
    parser.add_argument(
        "--sequences-path",
        type=Path,
        default=_DEFAULT_SEQS,
        help="Path to hla_sequences.json. Run scripts/download_hla_seqs.py first.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Path to HDF5 embedding cache file (created if absent).",
    )
    parser.add_argument(
        "--loci",
        nargs="+",
        default=list(_DEFAULT_LOCI),
        metavar="LOCUS",
        help="HLA loci to embed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: 'cpu', 'cuda', 'mps'. None = auto-detect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sequences per ESM-2 forward pass. Reduce if OOM.",
    )
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace ESM-2 model identifier.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be computed without loading ESM-2 or writing to cache.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Allele extraction
# ---------------------------------------------------------------------------

def extract_alleles_from_bmt(loci: list[str]) -> list[str]:
    """Return the unique HLA alleles relevant to the UCI BMT cohort.

    The UCI BMT dataset stores only aggregate HLA match scores, not per-patient
    allele names.  The clinically typed alleles for this paediatric HSCT cohort
    are represented by :data:`~capa.data.hla_parser.ANTIGEN_TO_COMMON_ALLELE` —
    the most common WHO allele within each serological antigen group.

    Parameters
    ----------
    loci : list[str]
        HLA gene names to include, e.g. ``["A", "B", "C", "DRB1", "DQB1"]``.

    Returns
    -------
    list[str]
        Deduplicated, sorted list of allele names in WHO notation
        (e.g. ``"A*02:01"``).
    """
    from capa.data.hla_parser import ANTIGEN_TO_COMMON_ALLELE

    loci_set = set(loci)
    alleles: set[str] = set()

    for allele in ANTIGEN_TO_COMMON_ALLELE.values():
        # Gene is everything before the '*'
        if "*" in allele:
            gene = allele.split("*")[0]
            if gene in loci_set:
                alleles.add(allele)

    logger.info(
        "Extracted %d unique alleles for loci %s from the antigen→allele mapping "
        "(UCI BMT dataset has aggregate HLA columns only — allele set is "
        "derived from ANTIGEN_TO_COMMON_ALLELE).",
        len(alleles),
        ", ".join(loci),
    )
    return sorted(alleles)


# ---------------------------------------------------------------------------
# Sequence lookup
# ---------------------------------------------------------------------------

def lookup_sequences(
    alleles: list[str],
    sequences_path: Path,
) -> tuple[dict[str, str], list[str]]:
    """Look up protein sequences for every allele in *alleles*.

    Parameters
    ----------
    alleles : list[str]
        Allele names in WHO notation (e.g. ``"A*02:01"``).
    sequences_path : Path
        Path to ``hla_sequences.json``.

    Returns
    -------
    tuple[dict[str, str], list[str]]
        ``(allele_sequences, failures)`` where *allele_sequences* maps each
        successfully resolved allele to its amino acid sequence, and *failures*
        is the list of allele names for which no sequence could be found at any
        resolution level.
    """
    from capa.embeddings.hla_sequences import HLASequenceDB

    logger.info("Loading HLA sequence database from %s", sequences_path)
    seq_db = HLASequenceDB(sequences_path)

    allele_sequences: dict[str, str] = {}
    failures: list[str] = []

    for allele in alleles:
        try:
            seq = seq_db.get_by_name(allele)
            allele_sequences[allele] = seq
        except KeyError:
            logger.warning("No sequence found for allele %r — skipping.", allele)
            failures.append(allele)

    logger.info(
        "Sequence lookup complete: %d found, %d not found.",
        len(allele_sequences),
        len(failures),
    )
    return allele_sequences, failures


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def compute_and_cache(
    allele_sequences: dict[str, str],
    cache_path: Path,
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> tuple[int, int, list[str]]:
    """Embed *allele_sequences* using the cache to skip previously computed alleles.

    Parameters
    ----------
    allele_sequences : dict[str, str]
        Mapping of allele name → amino acid sequence.
    cache_path : Path
        HDF5 cache file path.
    model_name : str
        HuggingFace ESM-2 model identifier.
    device : str | None
        Torch device string, or ``None`` for auto-detection.
    batch_size : int
        Sequences per forward pass.

    Returns
    -------
    tuple[int, int, list[str]]
        ``(n_cache_hits, n_newly_computed, compute_failures)``
    """
    from capa.embeddings.cache import EmbeddingCache
    from capa.embeddings.esm_embedder import ESMEmbedder

    cache = EmbeddingCache(cache_path)

    # Tally pre-existing cache hits before touching the embedder
    cache_hits = [a for a in allele_sequences if cache.contains(a)]
    to_compute = [a for a in allele_sequences if not cache.contains(a)]

    logger.info(
        "Cache status: %d hits, %d to compute.",
        len(cache_hits),
        len(to_compute),
    )

    if not to_compute:
        logger.info("All embeddings already cached — nothing to compute.")
        return len(cache_hits), 0, []

    embedder = ESMEmbedder(model_name=model_name, device=device, batch_size=batch_size)

    compute_failures: list[str] = []
    newly_computed = 0

    try:
        missing_seqs = {a: allele_sequences[a] for a in to_compute}
        cache.embed_alleles(missing_seqs, embedder, show_progress=True)
        newly_computed = len(to_compute)
    except Exception as exc:
        # Compute in smaller chunks so partial progress is not lost
        logger.warning(
            "Batch embedding failed (%s); falling back to one-by-one.", exc
        )
        for allele in to_compute:
            try:
                vec = embedder.embed_one(allele_sequences[allele])
                cache.put(allele, vec)
                newly_computed += 1
            except Exception as inner_exc:
                logger.error("Failed to embed %r: %s", allele, inner_exc)
                compute_failures.append(allele)

    return len(cache_hits), newly_computed, compute_failures


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_SEP = "─" * 55


def print_report(
    *,
    bmt_path: Path,
    n_patients: int,
    loci: list[str],
    alleles: list[str],
    n_sequence_found: int,
    failures_lookup: list[str],
    n_cache_hits: int,
    n_newly_computed: int,
    failures_compute: list[str],
    cache_path: Path,
    elapsed: float,
    dry_run: bool,
) -> None:
    """Print a structured summary to stdout."""
    n_total = len(alleles)
    n_embedded = n_cache_hits + n_newly_computed

    lines = [
        "",
        "═" * 55,
        "  ESM-2 Embedding Report — CAPA / UCI BMT",
        "═" * 55,
        f"  Dataset      : {bmt_path.name}  ({n_patients} patients)",
        f"  HLA loci     : {', '.join(loci)}",
        _SEP,
        f"  Alleles (unique)     : {n_total:>6}",
        f"  Sequences found      : {n_sequence_found:>6}  "
        f"({100 * n_sequence_found / max(n_total, 1):.1f}%)",
        f"  Sequences not found  : {len(failures_lookup):>6}  "
        f"({100 * len(failures_lookup) / max(n_total, 1):.1f}%)",
        _SEP,
    ]

    if dry_run:
        lines += [
            "  [DRY RUN — ESM-2 was NOT loaded, no embeddings were computed]",
            f"  Would compute        : {n_sequence_found:>6}",
        ]
    else:
        lines += [
            f"  Cache hits           : {n_cache_hits:>6}  "
            f"({100 * n_cache_hits / max(n_sequence_found, 1):.1f}%)",
            f"  Newly computed       : {n_newly_computed:>6}  "
            f"({100 * n_newly_computed / max(n_sequence_found, 1):.1f}%)",
            f"  Compute failures     : {len(failures_compute):>6}",
            _SEP,
            f"  Embeddings in cache  : {n_embedded:>6}",
            f"  Cache file           : {cache_path}",
            f"  Elapsed              : {elapsed:.1f}s",
        ]

    if failures_lookup:
        lines += [_SEP, "  Sequence lookup failures:"]
        for a in failures_lookup:
            lines.append(f"    • {a}")

    if failures_compute:
        lines += [_SEP, "  Compute failures:"]
        for a in failures_compute:
            lines.append(f"    • {a}")

    lines.append("═" * 55)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns an exit code (0 = success, 1 = partial failure)."""
    args = parse_args(argv)
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load UCI BMT dataset (validates data pipeline, reports patient count)
    # ------------------------------------------------------------------
    if not args.bmt_path.exists():
        logger.error(
            "UCI BMT dataset not found at %s. "
            "Download it from https://archive.ics.uci.edu/dataset/565/"
            "bone+marrow+transplant+children",
            args.bmt_path,
        )
        return 1

    from capa.data.loader import load_bmt

    logger.info("Loading UCI BMT dataset from %s", args.bmt_path)
    bmt_df = load_bmt(args.bmt_path)
    n_patients = len(bmt_df)
    logger.info("Dataset loaded: %d patients × %d columns.", n_patients, len(bmt_df.columns))

    # ------------------------------------------------------------------
    # 2. Extract unique alleles for the requested loci
    # ------------------------------------------------------------------
    alleles = extract_alleles_from_bmt(args.loci)
    if not alleles:
        logger.error("No alleles found for loci %s. Aborting.", args.loci)
        return 1

    # ------------------------------------------------------------------
    # 3. Dry-run with no sequence DB: preview allele count + cache state
    # ------------------------------------------------------------------
    if args.dry_run and not args.sequences_path.exists():
        logger.warning(
            "HLA sequence database not found at %s — "
            "run 'uv run python scripts/download_hla_seqs.py' to download it.\n"
            "Reporting allele count only.",
            args.sequences_path,
        )
        from capa.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(args.cache_path)
        n_cache_hits = sum(1 for a in alleles if cache.contains(a))
        print_report(
            bmt_path=args.bmt_path,
            n_patients=n_patients,
            loci=args.loci,
            alleles=alleles,
            n_sequence_found=0,
            failures_lookup=[],
            n_cache_hits=n_cache_hits,
            n_newly_computed=0,
            failures_compute=[],
            cache_path=args.cache_path,
            elapsed=time.perf_counter() - t_start,
            dry_run=True,
        )
        return 0

    # ------------------------------------------------------------------
    # 4. Check that the HLA sequence database exists (full run)
    # ------------------------------------------------------------------
    if not args.sequences_path.exists():
        logger.error(
            "HLA sequence database not found at %s.\n"
            "Run first:  uv run python scripts/download_hla_seqs.py",
            args.sequences_path,
        )
        return 1

    # ------------------------------------------------------------------
    # 5. Look up protein sequences
    # ------------------------------------------------------------------
    allele_sequences, failures_lookup = lookup_sequences(alleles, args.sequences_path)
    n_sequence_found = len(allele_sequences)

    if not allele_sequences:
        logger.error(
            "No sequences found for any allele. "
            "Check that the HLA sequence database covers loci %s.",
            args.loci,
        )
        return 1

    # ------------------------------------------------------------------
    # 6. Dry-run with sequence DB: full preview (no ESM-2 loaded)
    # ------------------------------------------------------------------
    if args.dry_run:
        from capa.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(args.cache_path)
        n_cache_hits = sum(1 for a in allele_sequences if cache.contains(a))
        print_report(
            bmt_path=args.bmt_path,
            n_patients=n_patients,
            loci=args.loci,
            alleles=alleles,
            n_sequence_found=n_sequence_found,
            failures_lookup=failures_lookup,
            n_cache_hits=n_cache_hits,
            n_newly_computed=0,
            failures_compute=[],
            cache_path=args.cache_path,
            elapsed=time.perf_counter() - t_start,
            dry_run=True,
        )
        return 0

    # ------------------------------------------------------------------
    # 7. Compute embeddings (uses cache to skip existing)
    # ------------------------------------------------------------------
    n_cache_hits, n_newly_computed, failures_compute = compute_and_cache(
        allele_sequences,
        args.cache_path,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    print_report(
        bmt_path=args.bmt_path,
        n_patients=n_patients,
        loci=args.loci,
        alleles=alleles,
        n_sequence_found=n_sequence_found,
        failures_lookup=failures_lookup,
        n_cache_hits=n_cache_hits,
        n_newly_computed=n_newly_computed,
        failures_compute=failures_compute,
        cache_path=args.cache_path,
        elapsed=elapsed,
        dry_run=False,
    )

    return 1 if failures_compute else 0


if __name__ == "__main__":
    sys.exit(main())
