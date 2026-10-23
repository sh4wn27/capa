#!/usr/bin/env python3
"""Sensitivity of competing-risks results to the event-priority hierarchy.

CAPA assigns each patient exactly one competing event via a priority
hierarchy in which relapse outranks transplant-related mortality, which in
turn outranks severe acute GvHD (Section ``sec:data`` of the manuscript).
Because a patient may experience more than one of these, the ordering is a
modelling choice rather than a property of the data, and it is load-bearing:
ranking GvHD lowest means any patient who developed GvHD *and* later relapsed
or died is counted as a relapse or TRM event, never as a GvHD event.

This script re-derives the labels under three hierarchies and re-runs the
tabular baselines under each, so the dependence can be reported rather than
assumed away.

Hierarchies
-----------
``outcome``
    The manuscript's hierarchy: relapse > TRM > GvHD.  Reads as "what
    ultimately happened to this patient".  Event time is always
    ``survival_time_days``.
``gvhd_first``
    Reverses the ordering: GvHD > relapse > TRM.  Isolates how much of the
    GvHD signal the ``outcome`` hierarchy suppresses, holding times fixed.
``earliest``
    Time-to-first-event, the standard competing-risks construction: whichever
    event occurred first is the event, and the event *time* is that event's
    own time.  GvHD onset is taken from ``days_to_acute_gvhd_iii_iv``;
    relapse and death use ``survival_time_days``.  This is the only one of the
    three that is internally consistent with a DeepHit-style
    time-to-first-event likelihood.

Usage
-----
    uv run python scripts/run_hierarchy_sensitivity.py
    uv run python scripts/run_hierarchy_sensitivity.py --repeats 2 --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_real_baselines import (  # noqa: E402
    EVENT_NAMES,
    MAX_DAYS,
    build_feature_matrix,
    run_kfold_cv,
)

from capa.data.loader import load_bmt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BMT_PATH = PROJECT_ROOT / "data" / "raw" / "bone-marrow.arff"
OUT_PATH = PROJECT_ROOT / "data" / "results" / "hierarchy_sensitivity.json"

HIERARCHIES = ("outcome", "gvhd_first", "earliest")


def make_labels(df: pd.DataFrame, hierarchy: str) -> tuple[np.ndarray, np.ndarray]:
    """Derive competing-risk event codes and times under one hierarchy.

    Event codes follow the manuscript convention: 0 censored, 1 GvHD,
    2 relapse, 3 TRM.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned UCI BMT frame.
    hierarchy : str
        One of ``"outcome"``, ``"gvhd_first"``, ``"earliest"``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(event_type, event_time)``.
    """
    n = len(df)
    etype = np.zeros(n, dtype=np.int64)
    t_surv = df["survival_time_days"].astype(float).to_numpy().copy()
    etime = t_surv.copy()

    relapse = df["relapse"].fillna(0).astype(int).to_numpy()
    dead = df["dead"].fillna(0).astype(int).to_numpy()
    gvhd = df["acute_gvhd_iii_iv"].fillna(0).astype(int).to_numpy()

    if hierarchy == "outcome":
        etype[(gvhd == 1) & (relapse == 0) & (dead == 0)] = 1
        etype[(dead == 1) & (relapse == 0)] = 3
        etype[relapse == 1] = 2

    elif hierarchy == "gvhd_first":
        etype[(dead == 1) & (relapse == 0)] = 3
        etype[relapse == 1] = 2
        etype[gvhd == 1] = 1  # GvHD now overrides everything

    elif hierarchy == "earliest":
        # Baseline assignment on the survival endpoint.
        etype[(dead == 1) & (relapse == 0)] = 3
        etype[relapse == 1] = 2
        # GvHD wins wherever its onset strictly precedes the survival endpoint,
        # and the event time becomes the GvHD onset time.
        t_gvhd = df["days_to_acute_gvhd_iii_iv"].astype(float).to_numpy()
        gvhd_first = (gvhd == 1) & np.isfinite(t_gvhd) & (t_gvhd < t_surv)
        etype[gvhd_first] = 1
        etime[gvhd_first] = t_gvhd[gvhd_first]
        n_tied = int(((gvhd == 1) & ~gvhd_first).sum())
        if n_tied:
            logger.info(
                "earliest: %d GvHD case(s) not strictly before the survival "
                "endpoint, left as their competing event", n_tied,
            )
    else:
        raise ValueError(f"unknown hierarchy: {hierarchy!r}")

    etime = np.clip(etime, 0.0, MAX_DAYS)
    return etype, etime


def summarise_counts(etype: np.ndarray) -> dict[str, int]:
    """Count patients per event code."""
    return {
        "censored": int((etype == 0).sum()),
        "GvHD": int((etype == 1).sum()),
        "Relapse": int((etype == 2).sum()),
        "TRM": int((etype == 3).sum()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--data", type=Path, default=BMT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the hierarchy sensitivity analysis and write results to JSON."""
    args = parse_args()
    df = load_bmt(args.data)
    x_df = build_feature_matrix(df)
    logger.info("cohort n=%d, features=%d", len(df), x_df.shape[1])

    results: dict[str, Any] = {
        "_meta": {
            "n_patients": len(df),
            "folds": args.folds,
            "repeats": args.repeats,
            "event_codes": {"0": "censored", "1": "GvHD", "2": "Relapse", "3": "TRM"},
        },
        "hierarchies": {},
    }

    for hierarchy in HIERARCHIES:
        etype, etime = make_labels(df, hierarchy)
        counts = summarise_counts(etype)
        logger.info("hierarchy %-11s counts=%s", hierarchy, counts)

        entry: dict[str, Any] = {"counts": counts}
        # Events with too few cases cannot support stratified CV; record and skip.
        if counts["GvHD"] < args.folds:
            logger.warning(
                "hierarchy %s: only %d GvHD events, below %d folds",
                hierarchy, counts["GvHD"], args.folds,
            )
        cv = run_kfold_cv(
            x_df, etype, etime,
            n_folds=args.folds, n_repeats=args.repeats,
            feature_label=f"hierarchy={hierarchy}",
        )
        entry["cv"] = cv
        results["hierarchies"][hierarchy] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    logger.info("wrote %s", args.out)

    # Summary table
    logger.info("")
    logger.info(
        "%-12s %9s %9s %9s %9s", "hierarchy", "GvHD n", "Relapse n", "TRM n", "cens n",
    )
    for h, entry in results["hierarchies"].items():
        c = entry["counts"]
        logger.info(
            "%-12s %9d %9d %9d %9d",
            h, c["GvHD"], c["Relapse"], c["TRM"], c["censored"],
        )
    logger.info("")
    header = "  ".join(f"{e:>14s}" for e in EVENT_NAMES)
    logger.info("%-12s %-10s %s", "hierarchy", "model", header)
    for h, entry in results["hierarchies"].items():
        for model, per_event in entry["cv"].items():
            cells = []
            for e in EVENT_NAMES:
                r = per_event.get(e, {})
                m, s = r.get("cindex_mean"), r.get("cindex_std")
                cells.append(f"{m:.3f}+-{s:.3f}" if m is not None else "n/a")
            logger.info("%-12s %-10s %s", h, model, "  ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
