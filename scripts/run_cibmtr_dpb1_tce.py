#!/usr/bin/env python3
"""Test whether DPB1 T-cell-epitope mismatch *direction* predicts outcomes.

Companion to ``run_cibmtr_directional.py`` (IB20-02, eplet loads).  That
analysis asked whether directional eplet load adds to symmetric load in
haploidentical HCT.  This one asks the same question with a different
directional measure, a different locus and a ~4x larger cohort.

Data
----
CIBMTR publicly available dataset IB19-01a: 5,108 unrelated-donor HCT
recipients (AML/ALL/MDS, 2008--2017) with DPB1 T-cell-epitope (TCE) match
status coded by direction:

``tce``  0 = GvH non-permissive, 1 = HvG non-permissive, 2 = match,
         3 = permissive

A GvH non-permissive mismatch is one where the recipient's TCE group is the
more immunogenic, so it is a target for donor T cells; HvG non-permissive is
the reverse.  IB19-01b covers the same patients but codes TCE without
direction, so it cannot answer this question and is not used.

The dataset is used under the CIBMTR publicly-available-dataset terms: it is
read from ``data/raw/cibmtr/`` (gitignored) and only aggregate statistics are
written out.

Design
------
For each outcome, cause-specific Cox models (competing events censored) are
fitted with clinical covariates, HLA-A/-B/-C/-DRB1/-DQB1 match grade, and one
DPB1 feature set:

``clinical``     no DPB1 features
``symmetric``    permissive, non-permissive (direction pooled)
``directional``  symmetric + GvH non-permissive indicator

The extra ``directional`` coefficient is the log hazard ratio of GvH vs HvG
non-permissive, so the df=1 likelihood-ratio test of directional vs symmetric
asks: *given a non-permissive DPB1 mismatch, does its direction matter?*
Hazard ratios use centre-clustered robust standard errors.

Usage
-----
    uv run python scripts/run_cibmtr_dpb1_tce.py
    uv run python scripts/run_cibmtr_dpb1_tce.py --repeats 2 --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from lifelines import CoxPHFitter

from run_cibmtr_directional import cv_cindex, fit_cox, lrt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "cibmtr" / "IB19-01a" / "P-5390" / "ib1901a.sas7bdat"
)
OUT_PATH = PROJECT_ROOT / "data" / "results" / "cibmtr_dpb1_tce.json"

MISSING_CODE = 99.0
MIN_LEVEL_N = 20

TCE_LABELS = {0: "gvh_nonpermissive", 1: "hvg_nonpermissive", 2: "match", 3: "permissive"}

# outcome name -> (event indicator column, time column); times in months
OUTCOMES: dict[str, tuple[str, str]] = {
    "aGvHD II-IV": ("agvhd24", "intxagvhd24"),
    "aGvHD III-IV": ("agvhd34", "intxagvhd34"),
    "cGvHD": ("cgvhd", "intxcgvhd"),
    "Relapse": ("rel", "intxrel"),
    "TRM": ("trm", "intxsurv"),
    "Overall mortality": ("dead", "intxsurv"),
}

CONTINUOUS = ("age", "dnrage", "yeartx")
# Missing (99) is kept as its own level, CIBMTR's usual convention.
CATEGORICAL = (
    "kps_ge90", "hctcigp", "status", "condint", "gvhdgp", "graftype", "drcmvpr",
    "drsex", "atg_campath", "disease",
)
# Classical-locus match grade (0/1/2 mismatches); DPB1_MM is excluded because
# it is determined by the TCE category under test.
HLA_MM = ("A_MM", "B_MM", "C_MM", "DRB1_MM", "DQB1_MM")

DPB1_COLUMNS = {
    "clinical": [],
    "symmetric": ["tce_permissive", "tce_nonpermissive"],
    "directional": ["tce_permissive", "tce_nonpermissive", "tce_gvh_nonpermissive"],
}
FEATURE_SETS = tuple(DPB1_COLUMNS)


def load_cohort(path: Path) -> pd.DataFrame:
    """Load IB19-01a and add DPB1 TCE indicator columns.

    Parameters
    ----------
    path : Path
        Path to ``ib1901a.sas7bdat``.

    Returns
    -------
    pd.DataFrame
        One row per patient, with ``tce_group`` and 0/1 TCE indicators.
    """
    df = add_tce_indicators(pd.read_sas(path, encoding="latin-1"))
    # The two rarest prophylaxis regimens (n=18, 19) join "other".
    df["gvhdgp"] = df["gvhdgp"].replace({10.0: 20.0, 11.0: 20.0})
    return df


def add_tce_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Decode ``tce`` into ``tce_group`` and 0/1 indicator columns.

    Raises
    ------
    ValueError
        If ``tce`` holds a code outside :data:`TCE_LABELS`.
    """
    df = df.copy()
    df["tce_group"] = df["tce"].astype(int).map(TCE_LABELS)
    if df["tce_group"].isna().any():
        raise ValueError("unexpected tce code in IB19-01a")
    df["tce_permissive"] = (df["tce_group"] == "permissive").astype(float)
    df["tce_nonpermissive"] = df["tce_group"].str.endswith("nonpermissive").astype(float)
    df["tce_gvh_nonpermissive"] = (df["tce_group"] == "gvh_nonpermissive").astype(float)
    df["tce_hvg_nonpermissive"] = (df["tce_group"] == "hvg_nonpermissive").astype(float)
    return df


def clinical_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous covariates, HLA match grade, one-hot categoricals.

    Dummy levels with fewer than ``MIN_LEVEL_N`` patients (e.g. HCT-CI or
    donor/recipient sex missing, n=2) are folded into the reference level;
    otherwise a CV training fold can contain none of them and the Cox fit is
    singular.
    """
    cat = pd.get_dummies(
        df[list(CATEGORICAL)].astype(int).astype(str), drop_first=True, dtype=float,
    )
    cat = cat.loc[:, cat.sum() >= MIN_LEVEL_N]
    return pd.concat(
        [df[list(CONTINUOUS)].astype(float), df[list(HLA_MM)].astype(float), cat], axis=1,
    )


def outcome_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Rows usable for one outcome, with ``T`` and ``E`` columns.

    Drops patients with a missing event code or a non-positive event time, and
    the handful missing donor age.
    """
    ev_col, t_col = OUTCOMES[outcome]
    keep = (
        (df[ev_col] != MISSING_CODE) & df[t_col].notna() & (df[t_col] > 0)
        & df["dnrage"].notna()
    )
    out = df.loc[keep].copy()
    out["E"] = out[ev_col].astype(int)
    out["T"] = out[t_col].astype(float)
    return out


def design(df: pd.DataFrame, clin: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Design matrix: clinical covariates + DPB1 features + T/E."""
    x = pd.concat([clin.loc[df.index], df[DPB1_COLUMNS[feature_set]]], axis=1)
    x = x.loc[:, x.std() > 0]  # drop levels absent from this subset
    x["T"] = df["T"]
    x["E"] = df["E"]
    return x


def hr_rows(cph: CoxPHFitter, cols: list[str]) -> dict[str, dict[str, float]]:
    """Hazard ratio, 95% CI and p for selected coefficients."""
    s = cph.summary
    return {
        col: {
            "hr": float(s.loc[col, "exp(coef)"]),
            "ci_lo": float(s.loc[col, "exp(coef) lower 95%"]),
            "ci_hi": float(s.loc[col, "exp(coef) upper 95%"]),
            "p": float(s.loc[col, "p"]),
        }
        for col in cols
    }


def tce_hazard_ratios(
    sub: pd.DataFrame, clin: pd.DataFrame, cluster: pd.Series,
) -> dict[str, Any]:
    """Centre-clustered HRs for each TCE group vs match, and GvH vs HvG.

    Two parameterisations of the same model: indicators vs match for the
    clinically familiar HRs, and the ``directional`` set whose
    ``tce_gvh_nonpermissive`` coefficient is the GvH-vs-HvG contrast.
    """
    vs_match = ["tce_permissive", "tce_gvh_nonpermissive", "tce_hvg_nonpermissive"]
    x = pd.concat([clin.loc[sub.index], sub[vs_match]], axis=1)
    x = x.loc[:, x.std() > 0]
    x[["T", "E"]] = sub[["T", "E"]]
    x["_centre"] = cluster.loc[sub.index].to_numpy()
    out = hr_rows(fit_cox(x, cluster_col="_centre"), vs_match)

    xd = design(sub, clin, "directional")
    xd["_centre"] = cluster.loc[sub.index].to_numpy()
    contrast = hr_rows(fit_cox(xd, cluster_col="_centre"), ["tce_gvh_nonpermissive"])
    out["gvh_vs_hvg_nonpermissive"] = contrast["tce_gvh_nonpermissive"]
    return out


def crude_rates(sub: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Crude event proportion per TCE group (not a cumulative incidence)."""
    g = sub.groupby("tce_group")["E"]
    return {
        k: {"n": int(n), "events": int(e), "proportion": float(e / n)}
        for k, n, e in zip(g.size().index, g.size(), g.sum(), strict=True)
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the DPB1 TCE direction analysis and write aggregate results to JSON."""
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found. Download IB19-01a (P-5390.zip) from the CIBMTR "
            "publicly available datasets page under the CIBMTR terms and unzip it "
            "into data/raw/cibmtr/IB19-01a/."
        )
    df = load_cohort(args.data)
    clin = clinical_matrix(df)
    logger.info("cohort n=%d, clinical features=%d", len(df), clin.shape[1])

    results: dict[str, Any] = {
        "_meta": {
            "source": "CIBMTR IB19-01a (publicly available dataset P-5390)",
            "n_patients": len(df),
            "n_centres": int(df["pseudoccn"].nunique()),
            "tce_counts": {k: int(v) for k, v in df["tce_group"].value_counts().items()},
            "hla_adjustment": list(HLA_MM),
            "folds": args.folds,
            "repeats": args.repeats,
            "seed": args.seed,
            "time_unit": "months",
        },
        "outcomes": {},
    }

    for outcome in OUTCOMES:
        sub = outcome_frame(df, outcome)
        entry: dict[str, Any] = {"n": len(sub), "events": int(sub["E"].sum())}
        designs = {fs: design(sub, clin, fs) for fs in FEATURE_SETS}

        entry["crude"] = crude_rates(sub)
        entry["cv"] = {
            fs: cv_cindex(x, args.folds, args.repeats, args.seed)
            for fs, x in designs.items()
        }
        fits = {fs: fit_cox(x) for fs, x in designs.items()}
        entry["aic"] = {fs: float(f.AIC_partial_) for fs, f in fits.items()}
        entry["lrt"] = {
            "symmetric_vs_clinical": lrt(fits["symmetric"], fits["clinical"], 2),
            "directional_vs_symmetric": lrt(fits["directional"], fits["symmetric"], 1),
        }
        entry["hr"] = tce_hazard_ratios(sub, clin, df["pseudoccn"])
        results["outcomes"][outcome] = entry

        cv, hr = entry["cv"], entry["hr"]["gvh_vs_hvg_nonpermissive"]
        logger.info(
            "%-18s n=%4d ev=%4d | C clin %.3f sym %.3f dir %.3f | "
            "HR GvH/HvG %.2f (%.2f-%.2f) | LRT sym>clin p=%.3g dir>sym p=%.3g",
            outcome, entry["n"], entry["events"],
            *(cv[fs]["cindex_mean"] for fs in FEATURE_SETS),
            hr["hr"], hr["ci_lo"], hr["ci_hi"],
            entry["lrt"]["symmetric_vs_clinical"]["p"],
            entry["lrt"]["directional_vs_symmetric"]["p"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
