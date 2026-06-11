#!/usr/bin/env python3
"""Validate CAPA's core hypothesis on the IHWG HCT dataset.

The IHWG Hematopoietic Cell Transplantation dataset (NCBI FTP Final Archive)
contains 1,347 unrelated-donor HSCT patients with:
  - 4-digit allele-level HLA typing (A, B, C, DRB1, DQB1, DPB1)
  - Overall survival (Day + Died indicator)
  - Clinical covariates (age, sex, diagnosis, risk)

This script demonstrates the core CAPA hypothesis on real data:
  ESM-2 alloreactivity distances predict OS better than binary
  mismatch indicators, especially among patients with identical
  binary mismatch profiles.

Usage
-----
    uv run python scripts/run_ihwg_validation.py
    uv run python scripts/run_ihwg_validation.py --seed 42
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LOCI = ["A", "B", "C", "DRB1", "DQB1"]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ihwg_hct.tab"
EMB_PATH = PROJECT_ROOT / "data" / "processed" / "hla_embeddings.h5"
SEED = 42


# ---------------------------------------------------------------------------
# Allele normalization
# ---------------------------------------------------------------------------

def old_to_new(allele: str) -> str | None:
    """Convert old-format allele (A*0201, Cw*0202) to new format (A*02:01, C*02:02)."""
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele.startswith("Cw*"):
        allele = "C*" + allele[3:]
    # Handle ambiguous codes like DQB1*02AB → DQB1*02:01
    if re.search(r"[A-Z]{2}$", allele):
        allele = re.sub(r"[A-Z]{2}$", "01", allele)
    m = re.match(r"^([A-Z0-9]+\*\d{2})(\d{2})", allele)
    if m:
        return m.group(1) + ":" + m.group(2)
    return allele


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ihwg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    logger.info("Loaded IHWG HCT: %d patients × %d columns", len(df), len(df.columns))
    return df


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    embs: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    logger.info("Loaded %d embeddings (dim=%d)", len(embs), next(iter(embs.values())).shape[0])
    return embs


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def resolve_allele(raw: str | None, embs: dict[str, np.ndarray]) -> str | None:
    """Convert old-format allele and check cache; return key or None."""
    if raw is None or pd.isna(raw):
        return None
    new = old_to_new(str(raw))
    if new is None:
        return None
    if new in embs:
        return new
    # Supertype fallback: X*ab:cd → X*ab:01, X*ab:02, ...
    prefix = new.rsplit(":", 1)[0]
    for suffix in ["01", "02", "03", "04", "05"]:
        candidate = f"{prefix}:{suffix}"
        if candidate in embs:
            return candidate
    return None


def build_features(df: pd.DataFrame, embs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build per-patient feature matrix: binary mismatch + ESM-2 distances + covariates."""
    rows = []
    for _, row in df.iterrows():
        rec: dict = {}

        # Survival outcome
        rec["surv_days"] = float(row["Day"])
        rec["died"] = 1 if row["Died"] == "X" else 0

        # Clinical covariates (fill NaN with safe defaults)
        age = row.get("Age at Tx")
        rec["age_norm"] = float(age) / 100.0 if pd.notna(age) else 0.30
        donor_age = row.get("Donor Age")
        rec["donor_age_norm"] = float(donor_age) / 100.0 if pd.notna(donor_age) else 0.35
        sx_tx = row.get("Sex Tx", "M")
        sx_dn = row.get("Sex Dn", "M")
        if pd.isna(sx_tx): sx_tx = "M"
        if pd.isna(sx_dn): sx_dn = "M"
        rec["sex_mm"] = int(str(sx_tx) != str(sx_dn))

        # Diagnosis one-hot (simplified)
        diag = str(row.get("Diagnosis", "other")) if pd.notna(row.get("Diagnosis")) else "other"
        rec["diag_cml"] = int(diag == "CML")
        rec["diag_all"] = int(diag == "ALL")
        rec["diag_aml"] = int(diag == "AML")

        # Per-locus binary mismatch + ESM-2 L2 distance
        n_mm = 0
        loci_ok = 0
        for loc in LOCI:
            pt1_raw = row.get(f"Pt HLA-{loc} 1")
            pt2_raw = row.get(f"Pt HLA-{loc} 2")
            dn1_raw = row.get(f"Dn HLA-{loc} 1")
            dn2_raw = row.get(f"Dn HLA-{loc} 2")

            pt1_key = resolve_allele(pt1_raw, embs)
            pt2_key = resolve_allele(pt2_raw, embs)
            dn1_key = resolve_allele(dn1_raw, embs)
            dn2_key = resolve_allele(dn2_raw, embs)

            # Binary mismatch: any allele differs
            raw_alleles = [pt1_raw, pt2_raw, dn1_raw, dn2_raw]
            if any(pd.isna(v) for v in raw_alleles):
                bin_mm = 0
                dist = 0.0
            else:
                pt1_new = old_to_new(str(pt1_raw)) or ""
                pt2_new = old_to_new(str(pt2_raw)) or ""
                dn1_new = old_to_new(str(dn1_raw)) or ""
                dn2_new = old_to_new(str(dn2_raw)) or ""
                bin_mm = int(
                    {pt1_new, pt2_new} != {dn1_new, dn2_new}
                )
                n_mm += bin_mm

                # ESM-2 distance: mean L2 over allele pairs
                if pt1_key and dn1_key and pt2_key and dn2_key:
                    d1 = float(np.linalg.norm(embs[pt1_key] - embs[dn1_key]))
                    d2 = float(np.linalg.norm(embs[pt2_key] - embs[dn2_key]))
                    dist = (d1 + d2) / 2.0
                    loci_ok += 1
                elif pt1_key and dn1_key:
                    dist = float(np.linalg.norm(embs[pt1_key] - embs[dn1_key]))
                    loci_ok += 1
                else:
                    dist = 0.0

            rec[f"bin_mm_{loc}"] = bin_mm
            rec[f"dist_{loc}"] = dist

        rec["n_mm"] = n_mm
        rec["n_loci_ok"] = loci_ok
        rows.append(rec)

    feat_df = pd.DataFrame(rows)
    logger.info("Built features: %d patients × %d columns", len(feat_df), len(feat_df.columns))
    return feat_df


# ---------------------------------------------------------------------------
# Normalise distances (per-locus to [0,1])
# ---------------------------------------------------------------------------

def normalise_dists(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tr = df_tr.copy()
    df_te = df_te.copy()
    for loc in LOCI:
        col = f"dist_{loc}"
        max_val = df_tr[col].max()
        if max_val > 0:
            df_tr[col] = df_tr[col] / max_val
            df_te[col] = df_te[col] / max_val
    return df_tr, df_te


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------

def run_cox(df_tr: pd.DataFrame, df_te: pd.DataFrame, feat_cols: list[str], label: str) -> dict:
    """Cause-specific Cox PH for OS; returns C-index."""
    from lifelines import CoxPHFitter

    # Drop zero-variance columns to avoid singularity
    active = [c for c in feat_cols if df_tr[c].std() > 1e-8]
    dropped = set(feat_cols) - set(active)
    if dropped:
        logger.info("%s: dropping constant cols %s", label, sorted(dropped))

    train_df = df_tr[active + ["surv_days", "died"]].copy()
    test_df = df_te[active + ["surv_days", "died"]].copy()

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="surv_days", event_col="died")
        c = cph.concordance_index_
    except Exception as exc:
        logger.warning("%s Cox fit failed: %s", label, exc)
        c = 0.5

    # Test set C-index via lifelines
    try:
        risk = cph.predict_partial_hazard(test_df[active])
        from lifelines.utils import concordance_index as li_ci
        c_test = li_ci(test_df["surv_days"], -risk, test_df["died"])
    except Exception as exc:
        logger.warning("%s test C-index failed: %s", label, exc)
        c_test = 0.5

    logger.info("%s  train C=%.3f  test C=%.3f", label, c, c_test)
    return {"train_c": c, "test_c": c_test}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--min-loci", type=int, default=4,
                        help="Minimum number of loci with embeddings to include patient.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = load_ihwg(DATA_PATH)
    embs = load_embeddings(EMB_PATH)

    feats = build_features(df, embs)

    # Filter patients with sufficient loci coverage
    feats_ok = feats[feats["n_loci_ok"] >= args.min_loci].copy()
    logger.info(
        "Patients with ≥%d loci covered: %d/%d",
        args.min_loci, len(feats_ok), len(feats)
    )

    # Train/test split
    idx = rng.permutation(len(feats_ok))
    n_te = max(1, int(len(feats_ok) * args.test_frac))
    te_idx = idx[:n_te]
    tr_idx = idx[n_te:]
    df_tr = feats_ok.iloc[tr_idx].copy()
    df_te = feats_ok.iloc[te_idx].copy()

    # Event stats
    logger.info(
        "Train: %d (died=%d, %.1f%%)  Test: %d (died=%d, %.1f%%)",
        len(df_tr), df_tr["died"].sum(), 100 * df_tr["died"].mean(),
        len(df_te), df_te["died"].sum(), 100 * df_te["died"].mean(),
    )

    df_tr, df_te = normalise_dists(df_tr, df_te)

    # Feature sets (drop risk_group: all-NaN in IHWG data)
    clinical = ["age_norm", "donor_age_norm", "sex_mm",
                "diag_cml", "diag_all", "diag_aml"]
    binary_feats = clinical + ["n_mm"] + [f"bin_mm_{loc}" for loc in LOCI]
    dist_feats = clinical + [f"dist_{loc}" for loc in LOCI]

    # ── Cox models ───────────────────────────────────────────────────────────
    res_binary = run_cox(df_tr, df_te, binary_feats, "Cox-binary")
    res_dists  = run_cox(df_tr, df_te, dist_feats,  "Cox-distances")
    res_clin   = run_cox(df_tr, df_te, clinical,    "Cox-clinical")

    print("\n" + "=" * 70)
    print("IHWG HCT Validation — ESM-2 Distances vs. Binary Mismatch")
    print("=" * 70)
    print(f"{'Model':<20} {'Train C':>8} {'Test C':>8}")
    print("-" * 70)
    print(f"{'Cox-clinical':<20} {res_clin['train_c']:>8.3f} {res_clin['test_c']:>8.3f}")
    print(f"{'Cox-binary':<20} {res_binary['train_c']:>8.3f} {res_binary['test_c']:>8.3f}")
    print(f"{'Cox-ESM2-dist':<20} {res_dists['train_c']:>8.3f} {res_dists['test_c']:>8.3f}")
    print("-" * 70)
    delta = res_dists["test_c"] - res_binary["test_c"]
    print(f"  Δ C (ESM-2 dist − binary): {delta:+.3f}")
    print("=" * 70)

    mismatch_counts = feats_ok["n_mm"].value_counts().sort_index()
    print("\nMismatch distribution:")
    for mm, cnt in mismatch_counts.items():
        print(f"  {mm} mismatches: {cnt} patients")

    # ── Subset analysis: 1-mismatch patients ────────────────────────────────
    feats_1mm = feats_ok[feats_ok["n_mm"] == 1].copy()
    if len(feats_1mm) >= 50:
        logger.info("Running 1-mismatch subset analysis (n=%d)", len(feats_1mm))
        n_te_1 = max(1, int(len(feats_1mm) * args.test_frac))
        idx_1 = rng.permutation(len(feats_1mm))
        tr_1 = feats_1mm.iloc[idx_1[n_te_1:]].copy()
        te_1 = feats_1mm.iloc[idx_1[:n_te_1]].copy()
        tr_1, te_1 = normalise_dists(tr_1, te_1)
        r_bin_1 = run_cox(tr_1, te_1, binary_feats, "1mm-binary")
        r_dst_1 = run_cox(tr_1, te_1, dist_feats, "1mm-distances")
        print(f"\n1-mismatch subset (n={len(feats_1mm)}):")
        print(f"  Cox-binary     test C = {r_bin_1['test_c']:.3f}")
        print(f"  Cox-ESM2-dist  test C = {r_dst_1['test_c']:.3f}")
        print(f"  Δ C = {r_dst_1['test_c'] - r_bin_1['test_c']:+.3f}")
    else:
        logger.warning("Too few 1-mismatch patients (%d) for subset analysis", len(feats_1mm))


if __name__ == "__main__":
    main()
