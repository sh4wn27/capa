#!/usr/bin/env python3
"""Impute allele-level HLA strings into the UCI BMT dataset.

The UCI BMT ARFF has only aggregate mismatch counts (HLAmatch, Antigen, Alel).
This script assigns plausible donor-recipient allele pairs for the five standard
HSCT loci (A, B, C, DRB1, DQB1) using European population allele frequencies,
constrained so each patient's imputed mismatch count matches their recorded score.

OUTPUT LIMITATION
-----------------
Imputed allele assignments have NO relationship to actual patient HLA types.
ESM-2 embeddings of imputed alleles carry zero outcome-specific biological signal.
CAPA evaluated on this dataset performs at the clinical-only baseline — this
experiment validates end-to-end pipeline functionality, not predictive gain.

All alleles are restricted to those already present in data/processed/hla_embeddings.h5.
Frequencies are approximate values from NMDP/EFI European registry data.

Output
------
data/processed/bmt_imputed_hla.csv

Usage
-----
    uv run python scripts/impute_hla_alleles.py [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.data.loader import load_bmt  # noqa: E402

SEED = 42
LOCI = ["A", "B", "C", "DRB1", "DQB1"]

# European allele frequencies restricted to alleles in hla_embeddings.h5.
# Source: approximate values derived from published NMDP/EFI European data.
EURO_FREQS: dict[str, dict[str, float]] = {
    "A": {
        "A*02:01": 0.283, "A*01:01": 0.163, "A*03:01": 0.140,
        "A*24:02": 0.089, "A*11:01": 0.056, "A*29:02": 0.037,
        "A*26:01": 0.028, "A*68:01": 0.025, "A*30:01": 0.020,
        "A*31:01": 0.018, "A*32:01": 0.015, "A*25:01": 0.015,
        "A*23:01": 0.014, "A*33:01": 0.010, "A*34:01": 0.005,
        "A*69:01": 0.005, "A*36:01": 0.005, "A*66:01": 0.005,
        "A*43:01": 0.003, "A*74:01": 0.003, "A*80:01": 0.002,
    },
    "B": {
        "B*07:02": 0.122, "B*44:02": 0.115, "B*08:01": 0.099,
        "B*35:01": 0.070, "B*51:01": 0.055, "B*57:01": 0.050,
        "B*15:01": 0.045, "B*40:01": 0.043, "B*18:01": 0.040,
        "B*38:01": 0.025, "B*52:01": 0.020, "B*13:01": 0.018,
        "B*27:05": 0.017, "B*49:01": 0.015, "B*55:01": 0.013,
        "B*50:01": 0.012, "B*40:02": 0.012, "B*14:02": 0.011,
        "B*39:01": 0.010, "B*37:01": 0.009, "B*14:01": 0.008,
        "B*41:01": 0.007, "B*56:01": 0.006, "B*47:01": 0.005,
        "B*53:01": 0.005, "B*58:01": 0.005, "B*45:01": 0.004,
        "B*48:01": 0.004, "B*46:01": 0.004, "B*42:01": 0.003,
        "B*59:01": 0.003, "B*54:01": 0.003, "B*73:01": 0.002,
        "B*67:01": 0.002, "B*15:02": 0.002, "B*15:03": 0.002,
        "B*15:07": 0.001, "B*15:08": 0.001, "B*15:09": 0.001,
        "B*15:11": 0.001, "B*15:12": 0.001,
    },
    "C": {
        "C*07:01": 0.195, "C*04:01": 0.125, "C*06:02": 0.095,
        "C*05:01": 0.090, "C*03:04": 0.080, "C*03:03": 0.065,
        "C*12:03": 0.055, "C*08:02": 0.040, "C*02:02": 0.035,
        "C*12:02": 0.025, "C*01:02": 0.020, "C*15:02": 0.015,
        "C*14:02": 0.015, "C*17:01": 0.010, "C*18:01": 0.005,
    },
    "DRB1": {
        "DRB1*15:01": 0.143, "DRB1*07:01": 0.129, "DRB1*03:01": 0.127,
        "DRB1*04:01": 0.118, "DRB1*11:01": 0.085, "DRB1*13:01": 0.080,
        "DRB1*01:01": 0.072, "DRB1*16:01": 0.033, "DRB1*14:01": 0.030,
        "DRB1*08:01": 0.025, "DRB1*12:01": 0.020, "DRB1*10:01": 0.015,
        "DRB1*09:01": 0.010, "DRB1*03:02": 0.008,
    },
    "DQB1": {
        "DQB1*02:01": 0.218, "DQB1*03:01": 0.165, "DQB1*06:02": 0.140,
        "DQB1*05:01": 0.125, "DQB1*03:02": 0.120, "DQB1*03:03": 0.050,
        "DQB1*04:02": 0.025,
    },
}


def _normalize(d: dict[str, float]) -> tuple[list[str], np.ndarray]:
    alleles = list(d.keys())
    probs = np.array(list(d.values()), dtype=float)
    probs /= probs.sum()
    return alleles, probs


_POOLS: dict[str, tuple[list[str], np.ndarray]] = {
    loc: _normalize(EURO_FREQS[loc]) for loc in LOCI
}


def _sample_genotype(rng: np.random.Generator) -> dict[str, list[str]]:
    """Sample 2 alleles per locus from European frequency distribution."""
    return {
        loc: list(rng.choice(alleles, size=2, replace=True, p=probs))
        for loc, (alleles, probs) in _POOLS.items()
    }


def _add_mismatches(
    recipient: dict[str, list[str]],
    n_mm: int,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Return a donor genotype that differs from recipient at exactly n_mm positions.

    Mismatch positions are sampled without replacement from all 10 allele slots
    (2 per locus × 5 loci). Each mismatched slot gets a different allele drawn
    from the same-locus frequency distribution.
    """
    donor = copy.deepcopy(recipient)
    if n_mm == 0:
        return donor

    all_slots = [(loc, i) for loc in LOCI for i in range(2)]
    chosen = [all_slots[j] for j in rng.choice(len(all_slots), size=n_mm, replace=False)]

    for loc, idx in chosen:
        alleles, probs = _POOLS[loc]
        current = recipient[loc][idx]
        others = [a for a in alleles if a != current]
        if not others:
            continue
        other_probs = np.array([probs[alleles.index(a)] for a in others])
        other_probs /= other_probs.sum()
        donor[loc][idx] = rng.choice(others, p=other_probs)

    return donor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/bmt_imputed_hla.csv",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = load_bmt(PROJECT_ROOT / "data/raw/bone-marrow.arff")
    print(f"Loaded {len(df)} patients from UCI BMT")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        n_mm = int(row["hla_match_score"])  # 0→10/10, 1→9/10, 2→8/10, 3→7/10
        recip = _sample_genotype(rng)
        donor = _add_mismatches(recip, n_mm, rng)

        rec: dict = {"patient_idx": i}
        for loc in LOCI:
            rec[f"recipient_{loc}_1"] = recip[loc][0]
            rec[f"recipient_{loc}_2"] = recip[loc][1]
            rec[f"donor_{loc}_1"] = donor[loc][0]
            rec[f"donor_{loc}_2"] = donor[loc][1]
        records.append(rec)

    allele_df = pd.DataFrame(records).drop(columns=["patient_idx"])
    result = pd.concat([df.reset_index(drop=True), allele_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(result)} rows, {len(result.columns)} columns)")

    # Verify mismatch consistency
    n_verified = 0
    for _, row in result.iterrows():
        expected = int(row["hla_match_score"])
        actual = sum(
            1
            for loc in LOCI
            for idx in (1, 2)
            if row[f"donor_{loc}_{idx}"] != row[f"recipient_{loc}_{idx}"]
        )
        assert actual == expected, f"Mismatch count error: expected {expected}, got {actual}"
        n_verified += 1
    print(f"Verified mismatch counts for all {n_verified} patients")

    print("\nSample (first 3 patients):")
    sample_cols = ["hla_match_score"] + [
        f"{side}_{loc}_1"
        for side in ("recipient", "donor")
        for loc in ("A", "DRB1")
    ]
    print(result[sample_cols].head(3).to_string())


if __name__ == "__main__":
    main()
