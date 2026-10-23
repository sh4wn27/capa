#!/usr/bin/env python3
"""Ground ESM-2 HLA embedding distances against EpRegistry eplet loads.

Addresses the reviewer's highest-leverage request: show that the continuous
ESM-2 representation tracks an *established* immunogenicity metric, rather than
asserting it does. The external reference is PELC (Python Eplet Load
Calculator, Lhotte et al. 2022, doi:10.5281/zenodo.7526198), which computes
eplet loads from the HLA Eplet Registry (epregistry.com.br). PELC is an
independent package -- it shares no code or data with CAPA, so the correlation
is not circular.

Three analyses are reported per locus and pooled:

1. **Grounding.** Spearman rho between the symmetric ESM-2 L2 distance and the
   total EpRegistry eplet load over all within-locus allele pairs.

2. **Internal proxy validation.** Spearman rho between CAPA's own
   ``compute_eplet_features`` position-mismatch proxy and the PELC reference,
   reported both with and without the leader-peptide offset correction, to
   document the magnitude of that fix.

3. **Irreducible asymmetry.** EpRegistry eplet load is *directional*: the
   eplets present in donor-but-not-recipient differ in number from the
   reverse. Any symmetric pair function f(a,b) = f(b,a) necessarily predicts
   the same value for both orderings, so its best possible prediction is the
   per-pair mean. The residual is an error floor that no symmetric metric can
   cross, however sophisticated. This quantifies that floor, which is direct
   external support for CAPA's signed-difference design.

Usage
-----
    uv run python scripts/run_eplet_correlation.py
    uv run python scripts/run_eplet_correlation.py --loci A DRB1 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.embeddings.hla_sequences import HLASequenceDB  # noqa: E402
from capa.model.baselines import _EPLET_POSITIONS, _LEADER_LENGTH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LOCI = ["A", "B", "C", "DRB1", "DQB1"]
EMB_PATH = PROJECT_ROOT / "data" / "processed" / "hla_embeddings.h5"
SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "hla_sequences.json"
OUT_PATH = PROJECT_ROOT / "data" / "results" / "eplet_correlation.json"


def load_embeddings(path: Path, loci: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load the cached ESM-2 allele embeddings, grouped by locus.

    Parameters
    ----------
    path : Path
        HDF5 embedding cache, one dataset per allele name.
    loci : list[str]
        Loci to retain.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        ``{locus: {allele_name: embedding}}``.
    """
    out: dict[str, dict[str, np.ndarray]] = {loc: {} for loc in loci}
    with h5py.File(path, "r") as fh:
        for name in fh:
            locus = name.split("*")[0]
            if locus in out:
                out[locus][name] = np.asarray(fh[name][:], dtype=np.float64)
    for loc in loci:
        logger.info("locus %-5s %3d cached alleles", loc, len(out[loc]))
    return out


def pelc_loads(allele_a: str, allele_b: str) -> tuple[int, int] | None:
    """Return the directional EpRegistry eplet loads for one allele pair.

    Parameters
    ----------
    allele_a, allele_b : str
        Two-field allele names, e.g. ``"A*01:01"``.

    Returns
    -------
    tuple[int, int] | None
        ``(load_a_not_b, load_b_not_a)``, or None if PELC cannot resolve
        either allele.
    """
    from pelc.simple_comparison import simple_comparison

    try:
        df = simple_comparison(allele_a, allele_b, None, interlocus2=False)
    except Exception as exc:  # PELC raises bare errors on unknown alleles
        logger.debug("PELC failed for %s vs %s: %s", allele_a, allele_b, exc)
        return None
    if df is None or "Eplet Load" not in df:
        return None
    vals = [int(v) for v in df["Eplet Load"]]
    if len(vals) != 2:
        return None
    return vals[0], vals[1]


def proxy_mismatch(d_seq: str, r_seq: str, locus: str, *, corrected: bool) -> int:
    """Count amino-acid mismatches at CAPA's known eplet positions.

    Parameters
    ----------
    d_seq, r_seq : str
        Full IPD-IMGT protein sequences, including the leader peptide.
    locus : str
        Locus name, used to look up eplet positions and leader length.
    corrected : bool
        If True, offset positions by the leader length (eplet nomenclature
        numbers from the mature protein). If False, reproduce the original
        uncorrected indexing.

    Returns
    -------
    int
        Number of mismatched eplet positions.
    """
    offset = _LEADER_LENGTH.get(locus, 0) if corrected else 0
    positions = _EPLET_POSITIONS.get(locus, [])
    return sum(
        1
        for pos in positions
        if (idx := pos - 1 + offset) < len(d_seq)
        and idx < len(r_seq)
        and d_seq[idx] != r_seq[idx]
    )


def _spearman(x: list[float], y: list[float]) -> dict[str, float | None]:
    """Spearman correlation with n and p-value, tolerant of degenerate input."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p": None, "n": len(x)}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": len(x)}


def asymmetry_floor(fwd: list[int], rev: list[int]) -> dict[str, float]:
    """Fraction of directional eplet-load variance no symmetric metric can explain.

    For each unordered pair the two orderings carry loads ``f`` and ``r``. A
    symmetric predictor must emit one value for both, and the variance-minimising
    choice is their mean ``s = (f + r) / 2``. The residual sum of squares is then
    ``sum[(f - s)^2 + (r - s)^2] = 2 * sum(a^2)`` with ``a = (f - r) / 2``, while
    the total sum of squares is taken about the grand mean over ordered pairs.
    Their ratio is an exact lower bound on the error of any symmetric metric.

    Parameters
    ----------
    fwd, rev : list[int]
        Directional loads, aligned per unordered pair.

    Returns
    -------
    dict[str, float]
        Floor fraction plus descriptive asymmetry statistics.
    """
    f = np.asarray(fwd, dtype=np.float64)
    r = np.asarray(rev, dtype=np.float64)
    ordered = np.concatenate([f, r])
    total_ss = float(((ordered - ordered.mean()) ** 2).sum())
    asym = (f - r) / 2.0
    resid_ss = float(2.0 * (asym**2).sum())
    diff = np.abs(f - r)
    return {
        "unexplainable_fraction": (resid_ss / total_ss) if total_ss > 0 else 0.0,
        "pct_pairs_asymmetric": float(100.0 * (diff > 0).mean()),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "max_abs_diff": float(diff.max()),
        "mean_total_load": float((f + r).mean()),
    }


def analyse_locus(
    locus: str,
    embeddings: dict[str, np.ndarray],
    seq_db: HLASequenceDB,
) -> dict[str, Any] | None:
    """Run all three analyses for one locus over every within-locus allele pair."""
    names = sorted(embeddings)
    if len(names) < 3:
        logger.warning("locus %s: only %d alleles, skipping", locus, len(names))
        return None

    esm_l2: list[float] = []
    esm_cos: list[float] = []
    pelc_total: list[float] = []
    pelc_fwd: list[int] = []
    pelc_rev: list[int] = []
    # Proxy rows carry their own aligned targets: sequences are missing for some
    # alleles, so these lists are shorter than the PELC lists and must not be
    # index-matched against them.
    proxy_fixed: list[float] = []
    proxy_buggy: list[float] = []
    proxy_target_pelc: list[float] = []
    proxy_target_esm: list[float] = []
    n_pairs = 0
    n_unresolved = 0

    seqs: dict[str, str] = {}
    for name in names:
        try:
            seqs[name] = seq_db.get_by_name(name)
        except Exception as exc:
            logger.debug("no sequence for %s: %s", name, exc)

    for a, b in itertools.combinations(names, 2):
        n_pairs += 1
        loads = pelc_loads(a, b)
        if loads is None:
            n_unresolved += 1
            continue
        fwd, rev = loads

        va, vb = embeddings[a], embeddings[b]
        esm_l2.append(float(np.linalg.norm(va - vb)))
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        esm_cos.append(1.0 - float(va @ vb) / denom if denom > 0 else 0.0)

        pelc_fwd.append(fwd)
        pelc_rev.append(rev)
        pelc_total.append(float(fwd + rev))

        if a in seqs and b in seqs:
            proxy_fixed.append(float(proxy_mismatch(seqs[a], seqs[b], locus, corrected=True)))
            proxy_buggy.append(float(proxy_mismatch(seqs[a], seqs[b], locus, corrected=False)))
            proxy_target_pelc.append(float(fwd + rev))
            proxy_target_esm.append(esm_l2[-1])

    if not pelc_total:
        logger.warning("locus %s: no pairs resolved by PELC", locus)
        return None

    n_ok = len(pelc_total)
    logger.info(
        "locus %-5s %5d pairs, %5d resolved (%4.1f%%)",
        locus, n_pairs, n_ok, 100.0 * n_ok / n_pairs,
    )

    result: dict[str, Any] = {
        "n_alleles": len(names),
        "n_pairs": n_pairs,
        "n_resolved": n_ok,
        "n_unresolved": n_unresolved,
        "esm_l2_vs_pelc_total": _spearman(esm_l2, pelc_total),
        "esm_cosine_vs_pelc_total": _spearman(esm_cos, pelc_total),
        "asymmetry": asymmetry_floor(pelc_fwd, pelc_rev),
    }
    if proxy_fixed:
        result["n_proxy_pairs"] = len(proxy_fixed)
        result["proxy_corrected_vs_pelc_total"] = _spearman(
            proxy_fixed, proxy_target_pelc
        )
        result["proxy_uncorrected_vs_pelc_total"] = _spearman(
            proxy_buggy, proxy_target_pelc
        )
        result["proxy_corrected_vs_esm_l2"] = _spearman(
            proxy_fixed, proxy_target_esm
        )
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loci", nargs="+", default=LOCI, help="Loci to analyse")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Output JSON path")
    parser.add_argument("--embeddings", type=Path, default=EMB_PATH)
    parser.add_argument("--sequences", type=Path, default=SEQ_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the eplet-correlation study and write results to JSON."""
    args = parse_args()
    emb = load_embeddings(args.embeddings, args.loci)
    seq_db = HLASequenceDB(args.sequences)

    results: dict[str, Any] = {
        "_meta": {
            "reference": "PELC 0.5.4.5 (EpRegistry) doi:10.5281/zenodo.7526198",
            "embeddings": str(args.embeddings.relative_to(PROJECT_ROOT)),
            "interlocus2": False,
            "verified_only": False,
        },
        "per_locus": {},
    }
    for locus in args.loci:
        res = analyse_locus(locus, emb[locus], seq_db)
        if res is not None:
            results["per_locus"][locus] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)

    logger.info(
        "%-6s %6s %12s %12s %14s",
        "locus", "pairs", "rho(L2,load)", "rho(proxy)", "sym-err-floor",
    )
    for locus, res in results["per_locus"].items():
        rho = res["esm_l2_vs_pelc_total"]["rho"]
        prox = res.get("proxy_corrected_vs_pelc_total", {}).get("rho")
        floor = res["asymmetry"]["unexplainable_fraction"]
        logger.info(
            "%-6s %6d %12s %12s %13.1f%%",
            locus,
            res["n_resolved"],
            f"{rho:.3f}" if rho is not None else "n/a",
            f"{prox:.3f}" if prox is not None else "n/a",
            100 * floor,
        )


if __name__ == "__main__":
    main()
