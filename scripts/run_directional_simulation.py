#!/usr/bin/env python3
"""Directional alloreactivity simulation — where CAPA beats scalar distances.

SCIENTIFIC QUESTION
-------------------
A scalar mismatch distance  d_l = ||e^D_l - e^R_l||  is SYMMETRIC and collapses
the 1280-dim difference vector into a single magnitude.  It therefore discards:

  1. DIRECTIONALITY — graft-versus-host alloreactivity depends on antigens the
     RECIPIENT carries that the DONOR lacks (donor T-cells see them as foreign).
     This is the *signed* difference  (e^R - e^D), not its magnitude.
     ||e^D - e^R|| == ||e^R - e^D||, so a distance cannot represent the sign.
  2. POSITION — two allele pairs with identical L2 distance can differ at an
     immunodominant groove residue vs. a benign surface residue.

A LEARNED model operating on the signed difference embeddings can recover the
immunodominant direction; a hand-computed scalar distance provably cannot.

DESIGN
------
GvHD hazard is driven by the rectified projection of the signed per-locus
difference onto a fixed (immunodominant) direction w_l:

    log h_GvHD = log λ₀ + 2.5·relu(z_DRB1) + 1.5·relu(z_DQB1) + 0.3·age
    where z_l = standardised( w_l · (e^R_l - e^D_l) )

relu() encodes directionality: only the GvH direction (recipient antigen the
donor lacks) raises hazard; the reverse does not.  TRM uses ordinary scalar
distances (a control where the distance representation IS sufficient).

BASELINES
---------
  • Cox (binary)            — all mismatched → chance on GvHD.
  • Cox (scalar distances)  — magnitude only; blind to direction → degraded.
  • Cox (oracle direction)  — given the true relu(z_l) features → ceiling.
  • CAPA (signed diff)      — self-attends over SIGNED (e^D - e^R) embeddings;
                              can learn w_l end-to-end → should beat distances.

Usage
-----
    uv run python scripts/run_directional_simulation.py --device mps
    uv run python scripts/run_directional_simulation.py --no-capa   # Cox only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.model.capa_model import CAPAModel
from capa.training.evaluate import concordance_index, bootstrap_ci
from scripts.run_haplo_simulation import (
    EMB_PATH, LOCI, MAX_DAYS, N_TIME_BINS, T365_BIN, N_BOOTSTRAP, SEED,
    _POOLS, _compute_max_l2,
    build_clinical_tensors, build_embedding_tensors, eval_cindex,
    run_cox, train_capa,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cohort generation with directional (signed-projection) alloreactivity
# ---------------------------------------------------------------------------

def generate_directional_cohort(
    n: int,
    embs: dict[str, np.ndarray],
    rng: np.random.Generator,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """All-loci-mismatched cohort; GvHD driven by directional projection.

    Returns the patient DataFrame and the per-locus immunodominant directions
    ``w_l`` (for diagnostics / reproducibility only — not used by any model).
    """
    emb_dim = next(iter(embs.values())).shape[0]
    max_l2 = _compute_max_l2(embs)

    # Fixed immunodominant direction per locus (unit vector). Seeded so the
    # simulation is reproducible; never revealed to any model.
    w_rng = np.random.default_rng(seed + 7)
    w: dict[str, np.ndarray] = {}
    for loc in LOCI:
        v = w_rng.standard_normal(emb_dim).astype(np.float64)
        w[loc] = v / np.linalg.norm(v)

    # --- draw alleles, compute scalar distance + signed projection ---
    rows: list[dict] = []
    raw_proj: dict[str, list[float]] = {loc: [] for loc in LOCI}
    for i in range(n):
        rec: dict = {"patient_idx": i}
        for loc in LOCI:
            alleles_l, probs_l = _POOLS[loc]
            recip_a = rng.choice(alleles_l, p=probs_l)
            others = [a for a in alleles_l if a != recip_a]
            op = np.array([probs_l[alleles_l.index(a)] for a in others])
            op /= op.sum()
            donor_a = rng.choice(others, p=op)

            e_d = embs[donor_a].astype(np.float64)
            e_r = embs[recip_a].astype(np.float64)
            rec[f"dist_{loc}"] = float(np.linalg.norm(e_d - e_r)) / max_l2[loc]
            raw_proj[loc].append(float(w[loc] @ (e_r - e_d)))   # GvH direction

            rec[f"bin_mm_{loc}"] = 1
            rec[f"donor_{loc}_1"] = donor_a
            rec[f"donor_{loc}_2"] = donor_a
            rec[f"recip_{loc}_1"] = recip_a
            rec[f"recip_{loc}_2"] = recip_a

        age = float(rng.uniform(20, 70))
        rec["age"] = age
        rec["age_norm"] = (age - 45.0) / 20.0
        rec["disease_risk"] = int(rng.random() < 0.35)
        rec["n_mismatches"] = 5
        rows.append(rec)

    df = pd.DataFrame(rows)

    # standardise projections per locus, then rectify (directional)
    for loc in LOCI:
        p = np.asarray(raw_proj[loc])
        z = (p - p.mean()) / (p.std() + 1e-8)
        df[f"zproj_{loc}"] = z
        df[f"relu_zproj_{loc}"] = np.maximum(z, 0.0)   # oracle directional feature

    # --- competing-risks outcomes ---
    times_t, etypes = [], []
    for _, r in df.iterrows():
        log_h_gvhd = (
            np.log(1 / 60000)
            + 2.5 * r["relu_zproj_DRB1"]
            + 1.5 * r["relu_zproj_DQB1"]
            + 0.3 * r["age_norm"]
        )
        # TRM: ordinary scalar distances (distance representation IS sufficient)
        log_h_trm = (
            np.log(1 / 150000)
            + 2.0 * r["dist_DRB1"] + 2.0 * r["dist_A"] + 1.0 * r["dist_C"]
            + 0.5 * r["age_norm"]
        )
        log_h_rel = (
            np.log(1 / 4000) - 0.8 * r["dist_B"] + 2.5 * r["disease_risk"]
        )
        rates = [np.exp(np.clip(h, -12, 4)) for h in (log_h_gvhd, log_h_rel, log_h_trm)]
        ev_times = [float(rng.exponential(1.0 / rt)) if rt > 0 else 1e9 for rt in rates]
        t_cens = float(rng.uniform(300, 2000))
        all_t = ev_times + [t_cens]
        winner = int(np.argmin(all_t))
        etypes.append(winner + 1 if winner < 3 else 0)
        times_t.append(min(all_t[winner], MAX_DAYS))

    df["survival_time_days"] = times_t
    df["event_type"] = etypes
    return df, w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-capa", action="store_true")
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Master seed for cohort, w-directions, split, init, "
                             "and minibatch shuffling (patches train_capa's SEED).")
    args = parser.parse_args()

    seed = args.seed
    # Propagate to run_haplo_simulation's module global so train_capa() and
    # bootstrap_ci pick up the same seed (they reference the imported constant).
    import scripts.run_haplo_simulation as _haplo
    _haplo.SEED = seed

    device = torch.device(args.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    embs: dict[str, np.ndarray] = {}
    with h5py.File(EMB_PATH, "r") as f:
        for k in f.keys():
            embs[k] = f[k][:]
    logger.info("Loaded %d allele embeddings", len(embs))

    logger.info("Generating directional cohort (n=%d, seed=%d)...", args.n, seed)
    df, _w = generate_directional_cohort(args.n, embs, rng, seed=seed)
    for k, nm in {0: "censored", 1: "GvHD", 2: "Relapse", 3: "TRM"}.items():
        logger.info("  %-10s: %d (%.1f%%)", nm, (df["event_type"] == k).sum(),
                    100 * (df["event_type"] == k).mean())

    idx_trainval, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.10, random_state=seed,
        stratify=df["event_type"].values)
    idx_tr, idx_va = train_test_split(
        idx_trainval, test_size=0.111, random_state=seed,
        stratify=df.iloc[idx_trainval]["event_type"].values)
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_te = df.iloc[idx_te].reset_index(drop=True)
    logger.info("Train=%d  Val=%d  Test=%d", len(idx_tr), len(idx_va), len(idx_te))

    etime_te = df_te["survival_time_days"].values
    etype_te = df_te["event_type"].values.astype(np.int64)

    # --- Cox baselines ---
    logger.info("Cox (binary)...")
    cox_bin = run_cox(df_tr, df_te,
                      ["age_norm", "disease_risk"] + [f"bin_mm_{l}" for l in LOCI],
                      "Cox-binary")
    logger.info("Cox (scalar distances)...")
    cox_dist = run_cox(df_tr, df_te,
                       ["age_norm", "disease_risk"] + [f"dist_{l}" for l in LOCI],
                       "Cox-distances")
    logger.info("Cox (oracle directional features)...")
    cox_oracle = run_cox(df_tr, df_te,
                         ["age_norm", "disease_risk"]
                         + [f"dist_{l}" for l in LOCI]
                         + [f"relu_zproj_{l}" for l in LOCI],
                         "Cox-oracle-direction")

    all_results: dict[str, dict] = {
        "Cox (binary mismatch)": cox_bin,
        "Cox (scalar distances)": cox_dist,
        "Cox (oracle direction)": cox_oracle,
    }

    # --- CAPA on SIGNED difference embeddings ---
    if not args.no_capa:
        logger.info("Building embedding tensors for CAPA...")
        donor_emb, recip_emb = build_embedding_tensors(df, embs)
        cont, cat = build_clinical_tensors(df, device)
        t_bins = np.floor(df["survival_time_days"].values / MAX_DAYS
                          * (N_TIME_BINS - 1)).astype(np.int64).clip(0, N_TIME_BINS - 1)
        times_t = torch.from_numpy(t_bins).long()
        types_t = torch.from_numpy(df["event_type"].values.astype(np.int64)).long()
        idx_tr_t = torch.from_numpy(idx_tr).long()
        idx_va_t = torch.from_numpy(idx_va).long()
        idx_te_t = torch.from_numpy(idx_te).long()

        emb_dim = donor_emb.shape[-1]
        model = CAPAModel(
            embedding_dim=emb_dim, loci=LOCI, clinical_dim=32,
            interaction_dim=128, survival_type="deephit",
            num_events=3, time_bins=N_TIME_BINS,
            event_names=["GvHD", "Relapse", "TRM"],
            num_heads=4, num_layers=2, dropout=0.2, proj_dim=args.proj_dim,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("CAPA trainable params: %d (proj_dim=%d)", n_params, args.proj_dim)

        donor_d = donor_emb.to(device)
        recip_d = recip_emb.to(device)
        best_ep = train_capa(
            donor_d[idx_tr_t], recip_d[idx_tr_t], cont[idx_tr_t], cat[idx_tr_t],
            times_t[idx_tr_t], types_t[idx_tr_t],
            donor_d[idx_va_t], recip_d[idx_va_t], cont[idx_va_t], cat[idx_va_t],
            times_t[idx_va_t], types_t[idx_va_t],
            model, device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
            alpha=0.0, diff_mode=True, signed_diff=True,
        )
        model.eval()
        with torch.no_grad():
            clin_te = model.clinical_encoder(cont[idx_te_t], cat[idx_te_t])
            te_d = (donor_d[idx_te_t] - recip_d[idx_te_t])   # SIGNED
            cif_te = model.cif(te_d, te_d, clin_te).cpu().numpy()
        all_results["CAPA (signed diff)"] = eval_cindex(cif_te, etime_te, etype_te)
        logger.info("CAPA best epoch: %d", best_ep)

    # --- report ---
    W = 84
    print("\n" + "=" * W)
    print("DIRECTIONAL SIMULATION — test-set C-index (n=%d; 80/10/10; seed=%d)" % (args.n, seed))
    print("GvHD driven by relu(signed projection) — scalar distance is direction-blind.")
    print("=" * W)
    print(f"{'Model':<30} {'GvHD':>15}  {'Relapse':>15}  {'TRM':>15}")
    print("-" * W)
    for label, res in all_results.items():
        row = f"{label:<30}"
        for name in ["GvHD", "Relapse", "TRM"]:
            r = res.get(name, {})
            c, lo, hi = r.get("cindex", np.nan), r.get("ci_low", np.nan), r.get("ci_hi", np.nan)
            row += f"  {'—':>15}" if np.isnan(c) else f"  {c:.3f} ({lo:.2f}–{hi:.2f})"
        print(row)
    print("=" * W)
    if "CAPA (signed diff)" in all_results:
        c_capa = all_results["CAPA (signed diff)"]["GvHD"]["cindex"]
        c_dist = cox_dist["GvHD"]["cindex"]
        c_orac = cox_oracle["GvHD"]["cindex"]
        print(f"\nGvHD:  Cox(distances)={c_dist:.3f}  CAPA(signed)={c_capa:.3f}  "
              f"oracle={c_orac:.3f}")
        print(f"  ΔC  CAPA − Cox(distances) = {c_capa - c_dist:+.3f}"
              f"   (CAPA recovers {100*(c_capa-c_dist)/(c_orac-c_dist+1e-9):.0f}% of "
              f"the direction-blind gap)")

    out = {
        "n": args.n, "mode": "directional", "seed": seed,
        "epochs": args.epochs if not args.no_capa else 0,
        "outcome_model": {
            "GvHD": "log h = log(1/60000) + 2.5*relu(z_DRB1) + 1.5*relu(z_DQB1) + 0.3*age; "
                    "z_l = standardised(w_l . (e_recip - e_donor))  [DIRECTIONAL]",
            "TRM":  "log h = log(1/150000) + 2.0*d_DRB1 + 2.0*d_A + 1.0*d_C + 0.5*age",
            "Relapse": "log h = log(1/4000) - 0.8*d_B + 2.5*disease_risk",
            "note": "Scalar distances are symmetric/magnitude-only and cannot represent "
                    "the signed projection direction. CAPA sees signed (e_D - e_R).",
        },
        "results": all_results,
    }
    suffix = "" if seed == SEED else f"_seed{seed}"
    out_path = PROJECT_ROOT / f"data/results/directional_simulation{suffix}.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
