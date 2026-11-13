#!/usr/bin/env python3
"""Test whether mismatch *direction* carries outcome information in registry data.

CAPA's central claim is that donor--recipient HLA incompatibility is
directional: a donor allele foreign to the recipient (graft-versus-host, GvH)
is not the same event as a recipient allele foreign to the donor
(host-versus-graft, HvG), so any symmetric pair function discards signal.
Sections ``sec:eplet`` and ``sec:directional`` support this with EpRegistry
eplet asymmetry and simulation.  This script tests it on real patients.

Data
----
CIBMTR publicly available dataset IB20-02 (Zou et al., *Transplant Cell
Ther* 2022, PMID 34774819): 1,287 haploidentical HCT recipients with
post-transplant cyclophosphamide, with HLAMatchmaker eplet mismatch counts
scored separately in the GvH and HvG directions at HLA-A, -B, -C, -DRB1 and
-DQB1, plus clinical covariates and outcomes with event times (months).

The dataset is used under the CIBMTR publicly-available-dataset terms: it is
read from ``data/raw/cibmtr/`` (gitignored) and only aggregate statistics are
written out.

Design
------
For each outcome, cause-specific Cox models (competing events censored) are
fitted with clinical covariates plus one eplet feature set:

``clinical``     no eplet features
``symmetric``    per-locus total load, GvH + HvG (direction-blind)
``gvh``          per-locus GvH load only
``hvg``          per-locus HvG load only
``directional``  per-locus GvH and HvG as separate features

``directional`` nests ``symmetric`` (it is a reparameterisation of total load
plus the GvH - HvG difference), so a likelihood-ratio test of directional vs
symmetric on the full cohort asks the precise question: *given total eplet
load, does knowing its direction improve the fit?*  Discrimination is
reported as repeated stratified k-fold cross-validated Harrell's C.

Usage
-----
    uv run python scripts/run_cibmtr_directional.py
    uv run python scripts/run_cibmtr_directional.py --repeats 2 --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cibmtr" / "IB20-02" / "P-5545" / "ib2002.sas7bdat"
OUT_PATH = PROJECT_ROOT / "data" / "results" / "cibmtr_directional.json"

LOCI = ("a", "b", "c", "drb1", "dq")
MISSING_CODE = 99.0
PENALIZER = 0.01

# outcome name -> (event indicator column, time column)
OUTCOMES: dict[str, tuple[str, str]] = {
    "aGvHD II-IV": ("agvhd", "intxagvhd"),
    "aGvHD III-IV": ("agvhd34", "intxagvhd34"),
    "cGvHD": ("cgvhd", "intxcgvhd"),
    "Relapse": ("rel", "intxrel"),
    "TRM": ("trm", "intxsurv"),
    "Overall mortality": ("dead", "intxsurv"),
}

CONTINUOUS = ("age", "dnrage", "yeartx")
# Missing (99) is kept as its own level, CIBMTR's usual convention.
CATEGORICAL = (
    "kps", "hctcigp", "drigp", "graftype", "condmye", "drcmvpr",
    "drsex", "relative_new", "disease",
)

FEATURE_SETS = ("clinical", "symmetric", "gvh", "hvg", "directional")


def load_cohort(path: Path) -> pd.DataFrame:
    """Load IB20-02 and add per-locus symmetric eplet totals.

    Parameters
    ----------
    path : Path
        Path to ``ib2002.sas7bdat``.

    Returns
    -------
    pd.DataFrame
        One row per patient.
    """
    df = pd.read_sas(path, encoding="latin-1")
    for locus in LOCI:
        df[f"{locus}_sym_count"] = df[f"{locus}_gvh_count"] + df[f"{locus}_hvg_count"]
    return df


def eplet_columns(feature_set: str) -> list[str]:
    """Eplet feature columns for one feature set."""
    if feature_set == "clinical":
        return []
    if feature_set == "symmetric":
        return [f"{loc}_sym_count" for loc in LOCI]
    if feature_set == "gvh":
        return [f"{loc}_gvh_count" for loc in LOCI]
    if feature_set == "hvg":
        return [f"{loc}_hvg_count" for loc in LOCI]
    if feature_set == "directional":
        return [f"{loc}_{d}_count" for loc in LOCI for d in ("gvh", "hvg")]
    raise ValueError(f"unknown feature set: {feature_set!r}")


def clinical_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot clinical covariates (reference level dropped)."""
    cat = pd.get_dummies(
        df[list(CATEGORICAL)].astype(int).astype(str), drop_first=True, dtype=float,
    )
    return pd.concat([df[list(CONTINUOUS)].astype(float), cat], axis=1)


def outcome_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Rows usable for one outcome, with ``T`` and ``E`` columns.

    Drops patients with a missing event code or a non-positive event time.
    """
    ev_col, t_col = OUTCOMES[outcome]
    keep = (df[ev_col] != MISSING_CODE) & df[t_col].notna() & (df[t_col] > 0)
    out = df.loc[keep].copy()
    out["E"] = out[ev_col].astype(int)
    out["T"] = out[t_col].astype(float)
    return out


def design(df: pd.DataFrame, clin: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Design matrix: clinical covariates + eplet features + T/E."""
    x = pd.concat([clin.loc[df.index], df[eplet_columns(feature_set)]], axis=1)
    x = x.loc[:, x.std() > 0]  # drop levels absent from this subset
    x["T"] = df["T"]
    x["E"] = df["E"]
    return x


def fit_cox(x: pd.DataFrame, cluster_col: str | None = None) -> CoxPHFitter:
    """Fit a lightly penalised Cox model, silencing convergence chatter.

    With ``cluster_col``, standard errors are robust and clustered on it.
    """
    cph = CoxPHFitter(penalizer=PENALIZER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(
            x, duration_col="T", event_col="E",
            cluster_col=cluster_col, robust=cluster_col is not None,
        )
    return cph


def cv_cindex(
    x: pd.DataFrame, folds: int, repeats: int, seed: int,
) -> dict[str, float]:
    """Repeated stratified k-fold Harrell's C for one design matrix.

    Features are standardised on the training fold only.
    """
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    feats = [c for c in x.columns if c not in ("T", "E")]
    scores: list[float] = []
    for tr, te in rskf.split(x, x["E"]):
        xtr, xte = x.iloc[tr].copy(), x.iloc[te].copy()
        mu, sd = xtr[feats].mean(), xtr[feats].std().replace(0, 1.0)
        xtr[feats] = (xtr[feats] - mu) / sd
        xte[feats] = (xte[feats] - mu) / sd
        if xte["E"].sum() < 2:
            continue
        cph = fit_cox(xtr)
        risk = cph.predict_partial_hazard(xte[feats]).to_numpy()
        scores.append(concordance_index(xte["T"], -risk, xte["E"]))
    arr = np.asarray(scores)
    return {
        "cindex_mean": float(arr.mean()),
        "cindex_std": float(arr.std(ddof=1)),
        "n_folds": int(arr.size),
    }


def lrt(full: CoxPHFitter, reduced: CoxPHFitter, df_diff: int) -> dict[str, float]:
    """Likelihood-ratio test of nested Cox models (penalised log-likelihoods)."""
    stat = 2.0 * (full.log_likelihood_ - reduced.log_likelihood_)
    return {"chi2": float(stat), "df": df_diff, "p": float(chi2.sf(max(stat, 0.0), df_diff))}


def directional_hazard_ratios(x: pd.DataFrame, cluster: pd.Series) -> dict[str, Any]:
    """Per-locus GvH and HvG hazard ratios per 10 eplets, centre-clustered SEs."""
    xs = x.copy()
    ep = eplet_columns("directional")
    xs[ep] = xs[ep] / 10.0
    xs["_centre"] = cluster.loc[xs.index].to_numpy()
    cph = fit_cox(xs, cluster_col="_centre")
    s = cph.summary
    return {
        col: {
            "hr_per10": float(s.loc[col, "exp(coef)"]),
            "ci_lo": float(s.loc[col, "exp(coef) lower 95%"]),
            "ci_hi": float(s.loc[col, "exp(coef) upper 95%"]),
            "p": float(s.loc[col, "p"]),
        }
        for col in ep
    }


def asymmetry_summary(df: pd.DataFrame) -> dict[str, Any]:
    """How often, and how strongly, GvH and HvG load disagree per locus."""
    out: dict[str, Any] = {}
    for loc in (*LOCI, "classI", "classII"):
        g, h = df[f"{loc}_gvh_count"], df[f"{loc}_hvg_count"]
        out[loc] = {
            "frac_asymmetric": float((g != h).mean()),
            "pearson_r": float(g.corr(h)),
            "gvh_median": float(g.median()),
            "hvg_median": float(h.median()),
        }
    return out


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
    """Run the directional analysis and write aggregate results to JSON."""
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found. Download IB20-02 (P-5545.zip) from "
            "https://cibmtr.org/Manuscript/a028a00001HHGJxAAP/P-5545 under the "
            "CIBMTR terms and unzip it into data/raw/cibmtr/IB20-02/."
        )
    df = load_cohort(args.data)
    clin = clinical_matrix(df)
    logger.info("cohort n=%d, clinical features=%d", len(df), clin.shape[1])

    results: dict[str, Any] = {
        "_meta": {
            "source": "CIBMTR IB20-02 (Zou et al. 2021, PMID 34774819)",
            "n_patients": len(df),
            "n_centres": int(df["pseudoccn"].nunique()),
            "loci": list(LOCI),
            "folds": args.folds,
            "repeats": args.repeats,
            "seed": args.seed,
            "penalizer": PENALIZER,
            "time_unit": "months",
        },
        "asymmetry": asymmetry_summary(df),
        "outcomes": {},
    }

    n_loci = len(LOCI)
    for outcome in OUTCOMES:
        sub = outcome_frame(df, outcome)
        entry: dict[str, Any] = {"n": len(sub), "events": int(sub["E"].sum())}
        designs = {fs: design(sub, clin, fs) for fs in FEATURE_SETS}

        entry["cv"] = {
            fs: cv_cindex(x, args.folds, args.repeats, args.seed)
            for fs, x in designs.items()
        }
        fits = {fs: fit_cox(x) for fs, x in designs.items()}
        entry["aic"] = {fs: float(f.AIC_partial_) for fs, f in fits.items()}
        entry["lrt"] = {
            "symmetric_vs_clinical": lrt(fits["symmetric"], fits["clinical"], n_loci),
            "directional_vs_symmetric": lrt(fits["directional"], fits["symmetric"], n_loci),
            "gvh_vs_clinical": lrt(fits["gvh"], fits["clinical"], n_loci),
            "hvg_vs_clinical": lrt(fits["hvg"], fits["clinical"], n_loci),
        }
        entry["directional_hr"] = directional_hazard_ratios(
            designs["directional"], df["pseudoccn"],
        )
        results["outcomes"][outcome] = entry

        cv = entry["cv"]
        logger.info(
            "%-18s n=%4d ev=%4d | C clin %.3f sym %.3f gvh %.3f hvg %.3f dir %.3f "
            "| LRT dir>sym p=%.3g",
            outcome, entry["n"], entry["events"],
            *(cv[fs]["cindex_mean"] for fs in FEATURE_SETS),
            entry["lrt"]["directional_vs_symmetric"]["p"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
