#!/usr/bin/env python3
"""Test whether DPB1 residue-level similarity predicts outcomes within a TCE category.

CAPA's founding claim is that amino-acid-level structure carries information
that a categorical mismatch label discards.  This script tests that claim on
real outcomes at HLA-DPB1, where the T-cell-epitope (TCE) algorithm already
groups mismatches into permissive and non-permissive categories.

Data
----
CIBMTR publicly available datasets IB19-01a (P-5390) and IB19-01b (P-5597):
the same 5,108 unrelated-donor HCT recipients, linked one-to-one on
``pseudoid`` (covariates and outcomes are identical across the two files).
IB19-01a supplies classical-locus match grade and the DPB1 allele mismatch
count; IB19-01b supplies two structural refinements, defined only for TCE3
3/3 permissive mismatches (both alleles in TCE group 3, n=2,205):

``v84_87``   donor and recipient share the DEAV/GGPM motif at DPB1 codons
             84--87 (1) or not (0)
``coretce``  the mismatch lies within the structurally similar "core" TCE3
             allele group (1) or not (0); every core mismatch is motif-matched

The two releases disagree on TCE status for 108 patients (105 permissive in
IB19-01a are non-permissive in IB19-01b; 3 permissive become match).  None of
them is in the TCE3 3/3 permissive subcohort, which is permissive in both.
The full-cohort model uses IB19-01b's coding throughout.

The datasets are used under the CIBMTR publicly-available-dataset terms: they
are read from ``data/raw/cibmtr/`` (gitignored) and only aggregate statistics
are written out.

Design
------
1. Within the TCE3 3/3 permissive subcohort, cause-specific Cox models adjust
   for clinical covariates, HLA-A/-B/-C/-DRB1/-DQB1 match grade and DPB1
   allele mismatch count (1 vs 2).  Nested likelihood-ratio tests (df=1 each):
   ``motif`` vs ``clinical`` (does codon 84--87 motif match matter?) and
   ``core`` vs ``motif`` (does core-group membership add to motif match?).
2. On the full cohort, centre-clustered hazard ratios for each DPB1 subgroup
   vs DPB1 match, so the structural effects can be read on the same scale as
   non-permissive mismatch.

Usage
-----
    uv run python scripts/run_cibmtr_dpb1_structure.py
    uv run python scripts/run_cibmtr_dpb1_structure.py --repeats 2 --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from run_cibmtr_directional import cv_cindex, fit_cox, lrt
from run_cibmtr_dpb1_tce import (
    HLA_MM,
    OUTCOMES,
    clinical_matrix,
    hr_rows,
    outcome_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW = PROJECT_ROOT / "data" / "raw" / "cibmtr"
DATA_A = RAW / "IB19-01a" / "P-5390" / "ib1901a.sas7bdat"
DATA_B = RAW / "IB19-01b" / "ib1901b.sas7bdat"
OUT_PATH = PROJECT_ROOT / "data" / "results" / "cibmtr_dpb1_structure.json"

# IB19-01a columns carried into the merged frame (the rest are duplicated in 01b).
FROM_A = ("pseudoid", "tce", *HLA_MM, "DPB1_MM")

STRUCT_COLUMNS = {
    "clinical": [],
    "motif": ["motif_match"],
    "core": ["motif_match", "core_tce3"],
}
FEATURE_SETS = tuple(STRUCT_COLUMNS)

# Full-cohort DPB1 subgroups vs match (IB19-01b TCE coding).
SUBGROUPS = (
    "nonpermissive", "permissive_other", "tce3_motif_mismatch",
    "tce3_motif_match_noncore", "tce3_core",
)


def merge_releases(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Link IB19-01a and IB19-01b on ``pseudoid`` and derive DPB1 subgroups.

    Parameters
    ----------
    a, b : pd.DataFrame
        Raw IB19-01a and IB19-01b frames.

    Returns
    -------
    pd.DataFrame
        One row per patient with ``dpb1_group``, ``in_tce3`` and 0/1
        ``motif_match`` / ``core_tce3`` / subgroup indicator columns.

    Raises
    ------
    ValueError
        If the two releases do not link one-to-one.
    """
    m = b.merge(a[list(FROM_A)], on="pseudoid", how="inner", validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"IB19-01a/b linked {len(m)} of {len(a)}/{len(b)} patients")

    m["in_tce3"] = m["TCE3"] == 1
    m["motif_match"] = (m["v84_87"] == 1).astype(float)
    m["core_tce3"] = (m["coretce"] == 1).astype(float)

    group = pd.Series("match", index=m.index)
    group[m["TCE"] == 3] = "nonpermissive"
    group[m["TCE"] == 2] = "permissive_other"
    group[m["in_tce3"] & (m["v84_87"] == 0)] = "tce3_motif_mismatch"
    group[m["in_tce3"] & (m["v84_87"] == 1) & (m["coretce"] == 0)] = "tce3_motif_match_noncore"
    group[m["in_tce3"] & (m["coretce"] == 1)] = "tce3_core"
    m["dpb1_group"] = group
    for g in SUBGROUPS:
        m[g] = (group == g).astype(float)
    # The two rarest prophylaxis regimens join "other", as in run_cibmtr_dpb1_tce.
    m["gvhdgp"] = m["gvhdgp"].replace({10.0: 20.0, 11.0: 20.0})
    return m


def subcohort_design(
    sub: pd.DataFrame, clin: pd.DataFrame, feature_set: str,
) -> pd.DataFrame:
    """Design matrix for the TCE3 subcohort: clinical + DPB1 MM count + features."""
    dpb1_two = (sub["DPB1_MM"] == 2).astype(float).rename("dpb1_two_mm")
    x = pd.concat(
        [clin.loc[sub.index], dpb1_two, sub[STRUCT_COLUMNS[feature_set]]], axis=1,
    )
    x = x.loc[:, x.std() > 0]
    x["T"] = sub["T"]
    x["E"] = sub["E"]
    return x


def subgroup_hazard_ratios(
    sub: pd.DataFrame, clin: pd.DataFrame, cluster: pd.Series,
) -> dict[str, dict[str, float]]:
    """Full-cohort centre-clustered HRs for each DPB1 subgroup vs match."""
    x = pd.concat([clin.loc[sub.index], sub[list(SUBGROUPS)]], axis=1)
    x = x.loc[:, x.std() > 0]
    x[["T", "E"]] = sub[["T", "E"]]
    x["_centre"] = cluster.loc[sub.index].to_numpy()
    return hr_rows(fit_cox(x, cluster_col="_centre"), list(SUBGROUPS))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the DPB1 structural analysis and write aggregate results to JSON."""
    args = parse_args()
    for path in (DATA_A, DATA_B):
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Download IB19-01a (P-5390) and IB19-01b (P-5597) "
                "from the CIBMTR publicly available datasets page and unzip them "
                "into data/raw/cibmtr/."
            )
    df = merge_releases(
        pd.read_sas(DATA_A, encoding="latin-1"), pd.read_sas(DATA_B, encoding="latin-1"),
    )
    clin = clinical_matrix(df)
    tce3 = df[df["in_tce3"]]
    logger.info(
        "cohort n=%d, TCE3 3/3 permissive n=%d (motif-matched %d, core %d)",
        len(df), len(tce3), int(tce3["motif_match"].sum()), int(tce3["core_tce3"].sum()),
    )

    results: dict[str, Any] = {
        "_meta": {
            "source": "CIBMTR IB19-01a (P-5390) + IB19-01b (P-5597), linked on pseudoid",
            "n_patients": len(df),
            "n_tce3_permissive": len(tce3),
            "dpb1_group_counts": {
                k: int(v) for k, v in df["dpb1_group"].value_counts().items()
            },
            "tce_release_discordant": int(
                ((df["tce"] == 3) & (df["TCE"] != 2)).sum()
            ),
            "folds": args.folds,
            "repeats": args.repeats,
            "seed": args.seed,
            "time_unit": "months",
        },
        "outcomes": {},
    }

    for outcome in OUTCOMES:
        full = outcome_frame(df, outcome)
        sub = full[full["in_tce3"]]
        entry: dict[str, Any] = {
            "n_tce3": len(sub), "events_tce3": int(sub["E"].sum()),
            "n_full": len(full), "events_full": int(full["E"].sum()),
        }
        designs = {fs: subcohort_design(sub, clin, fs) for fs in FEATURE_SETS}
        entry["cv"] = {
            fs: cv_cindex(x, args.folds, args.repeats, args.seed)
            for fs, x in designs.items()
        }
        fits = {fs: fit_cox(x) for fs, x in designs.items()}
        entry["aic"] = {fs: float(f.AIC_partial_) for fs, f in fits.items()}
        entry["lrt"] = {
            "motif_vs_clinical": lrt(fits["motif"], fits["clinical"], 1),
            "core_vs_motif": lrt(fits["core"], fits["motif"], 1),
        }
        xc = designs["core"].copy()
        xc["_centre"] = df.loc[sub.index, "pseudoccn"].to_numpy()
        entry["hr_tce3"] = hr_rows(
            fit_cox(xc, cluster_col="_centre"), ["motif_match", "core_tce3"],
        )
        entry["hr_vs_match"] = subgroup_hazard_ratios(full, clin, df["pseudoccn"])
        results["outcomes"][outcome] = entry

        hm = entry["hr_tce3"]["motif_match"]
        logger.info(
            "%-18s TCE3 n=%4d ev=%4d | C clin %.3f motif %.3f core %.3f | "
            "HR motif %.2f (%.2f-%.2f) | LRT motif p=%.3g core|motif p=%.3g",
            outcome, entry["n_tce3"], entry["events_tce3"],
            *(entry["cv"][fs]["cindex_mean"] for fs in FEATURE_SETS),
            hm["hr"], hm["ci_lo"], hm["ci_hi"],
            entry["lrt"]["motif_vs_clinical"]["p"], entry["lrt"]["core_vs_motif"]["p"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
