#!/usr/bin/env python3
"""Haploidentical HSCT simulation at N=10,000 — CAPA vs tabular baselines.

SCIENTIFIC QUESTION
-------------------
When all HLA loci are mismatched (haploidentical setting), can CAPA's
cross-attention over raw ESM-2 embeddings outperform tabular Cox models that
only see per-locus scalar distances?

KEY DESIGN
----------
  1. All 5 loci mismatched: binary mismatch indicators are IDENTICAL (all=1)
     for every patient → Cox(binary) cannot discriminate beyond age/disease risk.
  2. Outcome driven by continuous ESM-2 distances WITH cross-locus interaction
     terms:  d_DRB1 × d_DQB1  for GvHD,  d_DRB1 × d_A  for TRM.
  3. Cox (linear distances) recovers main-effect signal but cannot represent
     the multiplicative interactions → sub-optimal C-index.
  4. CAPA cross-attention queries DRB1 against DQB1/A across loci in the
     (batch, n_loci, n_loci) attention map, enabling it to learn the interaction.

OUTCOME MODEL
-------------
  log h_GvHD  = log(1/200000) + 3.5·d_DRB1 + 2.0·d_DQB1 + 2.5·d_DRB1·d_DQB1 + 0.3·z_age
  log h_TRM   = log(1/150000) + 2.0·d_DRB1 + 2.0·d_A   + 1.5·d_DRB1·d_A     + 1.0·d_C + 0.5·z_age
  log h_Rel   = log(1/4000)   - 0.8·d_B    + 2.5·1_high_risk - 0.3·d_DRB1
  (distances normalised by per-locus max pairwise L2 within EURO_FREQS pool)

Usage
-----
    uv run python scripts/run_haplo_simulation.py [--n 10000] [--epochs 500] [--device mps]
    uv run python scripts/run_haplo_simulation.py --no-capa          # Cox only (fast)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import copy
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.model.capa_model import CAPAModel
from capa.model.losses import deephit_loss
from capa.training.evaluate import concordance_index, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
LOCI = ["A", "B", "C", "DRB1", "DQB1"]
MAX_DAYS = 730.0
N_TIME_BINS = 100
T365_BIN = int(365.0 / MAX_DAYS * (N_TIME_BINS - 1))
N_BOOTSTRAP = 1000
EMB_PATH = PROJECT_ROOT / "data/processed/hla_embeddings.h5"

# ---------------------------------------------------------------------------
# European allele frequency pools
# ---------------------------------------------------------------------------
EURO_FREQS: dict[str, dict[str, float]] = {
    "A": {
        "A*02:01": 0.283, "A*01:01": 0.163, "A*03:01": 0.140,
        "A*24:02": 0.089, "A*11:01": 0.056, "A*29:02": 0.037,
        "A*26:01": 0.028, "A*68:01": 0.025, "A*30:01": 0.020,
        "A*31:01": 0.018, "A*32:01": 0.015, "A*25:01": 0.015,
        "A*23:01": 0.014, "A*33:01": 0.010,
    },
    "B": {
        "B*07:02": 0.122, "B*44:02": 0.115, "B*08:01": 0.099,
        "B*35:01": 0.070, "B*51:01": 0.055, "B*57:01": 0.050,
        "B*15:01": 0.045, "B*40:01": 0.043, "B*18:01": 0.040,
        "B*38:01": 0.025, "B*52:01": 0.020, "B*13:01": 0.018,
        "B*27:05": 0.017, "B*49:01": 0.015,
    },
    "C": {
        "C*07:01": 0.195, "C*04:01": 0.125, "C*06:02": 0.095,
        "C*05:01": 0.090, "C*03:04": 0.080, "C*03:03": 0.065,
        "C*12:03": 0.055, "C*08:02": 0.040, "C*02:02": 0.035,
        "C*12:02": 0.025, "C*01:02": 0.020, "C*15:02": 0.015,
        "C*14:02": 0.015, "C*17:01": 0.010,
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


def _normalize_pool(d: dict[str, float]) -> tuple[list[str], np.ndarray]:
    alleles = list(d.keys())
    p = np.array(list(d.values()), dtype=float)
    return alleles, p / p.sum()


_POOLS = {loc: _normalize_pool(EURO_FREQS[loc]) for loc in LOCI}


def _compute_max_l2(embs: dict[str, np.ndarray]) -> dict[str, float]:
    # Use only pool alleles — see run_mechanistic_benchmark.py for rationale
    max_l2: dict[str, float] = {}
    for loc in LOCI:
        pool_alleles = [a for a in _POOLS[loc][0] if a in embs]
        vecs = np.stack([embs[a] for a in pool_alleles])
        dists = np.array([
            np.linalg.norm(vecs[i] - vecs[j])
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))
        ])
        max_l2[loc] = float(dists.max())
        logger.debug("max L2 %s (pool): %.3f", loc, max_l2[loc])
    return max_l2


# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------

def _make_outcome(
    dist: dict[str, float],
    age_norm: float,
    disease_risk: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Haplo competing-risks outcome with cross-locus interaction terms.

    Interaction terms (d_DRB1·d_DQB1 for GvHD, d_DRB1·d_A for TRM) require
    cross-locus information that scalar per-locus distances cannot represent in
    a linear model — but CAPA cross-attention naturally captures them.
    """
    log_h_gvhd = (
        np.log(1 / 200000)                           # λ₀ → ~33% GvHD
        + 3.5 * dist["DRB1"]
        + 2.0 * dist["DQB1"]
        + 2.5 * dist["DRB1"] * dist["DQB1"]          # class II interaction
        + 0.3 * age_norm
    )
    log_h_trm = (
        np.log(1 / 150000)
        + 2.0 * dist["DRB1"]
        + 2.0 * dist["A"]
        + 1.5 * dist["DRB1"] * dist["A"]             # class I × II interaction
        + 1.0 * dist["C"]
        + 0.5 * age_norm
    )
    log_h_rel = (
        np.log(1 / 4000)
        - 0.8 * dist["B"]                            # GvL: B-locus NK effect
        + 2.5 * disease_risk
        - 0.3 * dist["DRB1"]
    )
    rates = [np.exp(np.clip(h, -12, 4)) for h in (log_h_gvhd, log_h_rel, log_h_trm)]
    times = [float(rng.exponential(1.0 / r)) if r > 0 else 1e9 for r in rates]
    t_cens = float(rng.uniform(300, 2000))
    all_t = times + [t_cens]
    winner = int(np.argmin(all_t))
    etype = winner + 1 if winner < 3 else 0
    return min(all_t[winner], MAX_DAYS), etype


# ---------------------------------------------------------------------------
# Cohort generation: haploidentical (ALL 5 loci mismatched)
# ---------------------------------------------------------------------------

def generate_haplo_cohort(
    n: int,
    embs: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Haploidentical cohort: all 5 loci mismatched, homozygous recipients.

    Homozygous recipients ensure alloreactivity = exact L2 between the
    specific donor and recipient allele at each locus (no ambiguity from
    heterozygous pairing). Binary mismatch indicators are ALL=1 for every
    patient, so Cox(binary) reduces to age+disease risk only.
    """
    max_l2 = _compute_max_l2(embs)
    records = []

    for i in range(n):
        dist: dict[str, float] = {}
        donor_alleles: dict[str, str] = {}
        recip_alleles: dict[str, str] = {}

        for loc in LOCI:
            alleles_l, probs_l = _POOLS[loc]
            recip_a = rng.choice(alleles_l, p=probs_l)
            others = [a for a in alleles_l if a != recip_a]
            op = np.array([probs_l[alleles_l.index(a)] for a in others])
            op /= op.sum()
            donor_a = rng.choice(others, p=op)

            dist_raw = float(np.linalg.norm(embs[donor_a] - embs[recip_a]))
            dist[loc] = dist_raw / max_l2[loc]
            donor_alleles[loc] = donor_a
            recip_alleles[loc] = recip_a

        age = float(rng.uniform(20, 70))
        age_norm = (age - 45.0) / 20.0
        disease_risk = int(rng.random() < 0.35)

        t_obs, etype = _make_outcome(dist, age_norm, disease_risk, rng)

        rec: dict = {
            "patient_idx": i,
            "age": age, "age_norm": age_norm, "disease_risk": disease_risk,
            "n_mismatches": 5,
            "survival_time_days": t_obs,
            "event_type": etype,
        }
        for loc in LOCI:
            rec[f"dist_{loc}"] = dist[loc]
            rec[f"bin_mm_{loc}"] = 1          # always 1 — haploidentical
            rec[f"donor_{loc}_1"] = donor_alleles[loc]
            rec[f"donor_{loc}_2"] = donor_alleles[loc]   # homozygous donor
            rec[f"recip_{loc}_1"] = recip_alleles[loc]
            rec[f"recip_{loc}_2"] = recip_alleles[loc]   # homozygous recipient

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Survival evaluation
# ---------------------------------------------------------------------------

def eval_cindex(cif: np.ndarray, etime: np.ndarray, etype: np.ndarray) -> dict:
    event_names = {1: "GvHD", 2: "Relapse", 3: "TRM"}
    results = {}
    for k, name in event_names.items():
        risks = cif[:, k - 1, T365_BIN].astype(np.float64)
        obs = (etype == k).astype(bool)
        if obs.sum() < 5:
            results[name] = {"cindex": float("nan"), "ci_low": float("nan"), "ci_hi": float("nan")}
            continue
        m = bootstrap_ci(
            concordance_index,
            etime.astype(np.float64),
            risks,
            obs,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )
        results[name] = {"cindex": m.value, "ci_low": m.ci_lower, "ci_hi": m.ci_upper}
    return results


# ---------------------------------------------------------------------------
# Cox baselines
# ---------------------------------------------------------------------------

def run_cox(df_tr: pd.DataFrame, df_te: pd.DataFrame, feat_cols: list[str], label: str) -> dict:
    from lifelines import CoxPHFitter

    active_feats = [c for c in feat_cols if df_tr[c].std() > 0]
    dropped = set(feat_cols) - set(active_feats)
    if dropped:
        logger.info("%s: dropping constant columns %s", label, sorted(dropped))

    etime_te = df_te["survival_time_days"].values
    etype_te = df_te["event_type"].values
    event_map = {1: "GvHD", 2: "Relapse", 3: "TRM"}
    cif_te = np.zeros((len(df_te), 3, N_TIME_BINS))
    time_grid = np.linspace(0, MAX_DAYS, N_TIME_BINS)

    for k, name in event_map.items():
        df_tr_k = df_tr[active_feats + ["survival_time_days", "event_type"]].copy()
        df_tr_k["event"] = (df_tr_k["event_type"] == k).astype(int)
        df_tr_k = df_tr_k.drop(columns=["event_type"])
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(df_tr_k, duration_col="survival_time_days", event_col="event")
        except Exception as e:
            logger.warning("%s Cox fit failed for %s: %s", label, name, e)
            continue
        sf = cph.predict_survival_function(df_te[active_feats])
        sf_interp = np.array([
            np.interp(time_grid, sf.index.values, sf.iloc[:, i].values)
            for i in range(len(df_te))
        ])
        cif_te[:, k - 1, :] = 1.0 - sf_interp

    return eval_cindex(cif_te, etime_te, etype_te)


# ---------------------------------------------------------------------------
# CAPA training helpers
# ---------------------------------------------------------------------------

def build_embedding_tensors(df: pd.DataFrame, embs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    emb_dim = next(iter(embs.values())).shape[0]
    n = len(df)
    donor_arr = np.zeros((n, len(LOCI), emb_dim), dtype=np.float32)
    recip_arr = np.zeros((n, len(LOCI), emb_dim), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for j, loc in enumerate(LOCI):
            d1 = embs.get(row[f"donor_{loc}_1"], np.zeros(emb_dim))
            d2 = embs.get(row[f"donor_{loc}_2"], np.zeros(emb_dim))
            r1 = embs.get(row[f"recip_{loc}_1"], np.zeros(emb_dim))
            r2 = embs.get(row[f"recip_{loc}_2"], np.zeros(emb_dim))
            donor_arr[i, j] = (d1 + d2) / 2.0
            recip_arr[i, j] = (r1 + r2) / 2.0
    return torch.from_numpy(donor_arr), torch.from_numpy(recip_arr)


def build_clinical_tensors(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(df)
    cont = np.zeros((n, 4), dtype=np.float32)
    cat = np.zeros((n, 4), dtype=np.int64)
    for i, (_, row) in enumerate(df.iterrows()):
        age_raw = float(row.get("age", 45.0))
        cont[i, 0] = age_raw / 100.0
        cont[i, 1] = 45.0 / 100.0
        cont[i, 2] = 5.0 / 10.0
        cont[i, 3] = float(row.get("disease_risk", 0))
        cat[i, 0] = 2 if row.get("disease_risk", 0) else 1
    return torch.from_numpy(cont).to(device), torch.from_numpy(cat).to(device)


def train_capa(
    donor_tr: torch.Tensor, recip_tr: torch.Tensor,
    cont_tr: torch.Tensor, cat_tr: torch.Tensor,
    times_tr: torch.Tensor, types_tr: torch.Tensor,
    donor_va: torch.Tensor, recip_va: torch.Tensor,
    cont_va: torch.Tensor, cat_va: torch.Tensor,
    times_va: torch.Tensor, types_va: torch.Tensor,
    model: CAPAModel, device: torch.device,
    epochs: int, lr: float, batch_size: int, patience: int,
    alpha: float = 0.0,
    diff_mode: bool = False,
    signed_diff: bool = False,
) -> int:
    """Train CAPA with early stopping on validation C-index (GvHD).

    alpha=0.0 uses pure NLL (ranking term disabled) which is more stable
    than alpha=0.5 when the model becomes confident — the pairwise
    exp(-delta/sigma) ranking term can explode for confidently wrong pairs.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    n = donor_tr.shape[0]
    rng = np.random.default_rng(SEED)

    best_cindex = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state: dict = {}

    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(n)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b = idx[start: start + batch_size]
            bt = torch.from_numpy(b).long()
            d_e = donor_tr[bt].to(device)
            r_e = recip_tr[bt].to(device)
            if diff_mode:
                # Per-locus donor - recipient difference; self-attention on this
                # directly encodes alloreactivity. signed_diff preserves the
                # mismatch *direction* (GvH vs HvD), which |.| would destroy.
                diff_e = (d_e - r_e) if signed_diff else (d_e - r_e).abs()
                d_e = diff_e
                r_e = diff_e
            ct = cont_tr[bt]
            ca = cat_tr[bt]
            t_b = times_tr[bt].to(device)
            ev_b = types_tr[bt].to(device)
            clin = model.clinical_encoder(ct, ca)
            logits = model(d_e, r_e, clin)
            loss = deephit_loss(logits, t_b, ev_b, alpha=alpha, sigma=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        sched.step()

        # Validation C-index every 10 epochs
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                if not diff_mode:
                    va_d = donor_va
                elif signed_diff:
                    va_d = (donor_va - recip_va)
                else:
                    va_d = (donor_va - recip_va).abs()
                va_r = donor_va if not diff_mode else va_d
                clin_va = model.clinical_encoder(cont_va, cat_va)
                cif_va = model.cif(va_d, va_r, clin_va).cpu().numpy()
            etype_va = types_va.cpu().numpy()
            etime_va_days = times_va.cpu().numpy().astype(float) * MAX_DAYS / (N_TIME_BINS - 1)
            risks_gvhd = cif_va[:, 0, T365_BIN]
            obs_gvhd = (etype_va == 1)
            if obs_gvhd.sum() >= 5:
                c = concordance_index(etime_va_days, risks_gvhd, obs_gvhd)
                if c > best_cindex:
                    best_cindex = c
                    best_epoch = ep
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 10
                if ep % 50 == 0:
                    logger.info("Epoch %3d  loss=%.4f  val_C_GvHD=%.3f  best=%.3f@%d",
                                ep, ep_loss / n_batches, c, best_cindex, best_epoch)
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", ep, patience)
                    break

    if best_state:
        model.load_state_dict(best_state)
        logger.info("Loaded best model from epoch %d (val_C_GvHD=%.3f)", best_epoch, best_cindex)
    return best_epoch


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
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--diff-mode", action="store_true",
                        help="Feed |donor-recipient| difference embeddings into "
                             "self-attention instead of raw donor/recipient. "
                             "Gives the model the alloreactivity signal directly.")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    embs: dict[str, np.ndarray] = {}
    with h5py.File(EMB_PATH, "r") as f:
        for k in f.keys():
            embs[k] = f[k][:]
    logger.info("Loaded %d allele embeddings", len(embs))

    logger.info("Generating haploidentical cohort (n=%d, all 5 loci mismatched)...", args.n)
    df = generate_haplo_cohort(args.n, embs, rng)

    event_label = {0: "censored", 1: "GvHD", 2: "Relapse", 3: "TRM"}
    for k, nm in event_label.items():
        logger.info("  %-10s: %d (%.1f%%)", nm, (df["event_type"] == k).sum(),
                    100 * (df["event_type"] == k).mean())

    # 80 / 10 / 10 split (train / val / test)
    idx_trainval, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.10, random_state=SEED,
        stratify=df["event_type"].values,
    )
    idx_tr, idx_va = train_test_split(
        idx_trainval, test_size=0.111, random_state=SEED,
        stratify=df.iloc[idx_trainval]["event_type"].values,
    )
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_va = df.iloc[idx_va].reset_index(drop=True)
    df_te = df.iloc[idx_te].reset_index(drop=True)
    logger.info("Train=%d  Val=%d  Test=%d", len(df_tr), len(df_va), len(df_te))

    etime_te = df_te["survival_time_days"].values
    etype_te = df_te["event_type"].values.astype(np.int64)

    # --- Cox: binary mismatch (constant → only age + disease_risk) ---
    logger.info("Running Cox (binary mismatch)...")
    binary_feats = ["age_norm", "disease_risk"] + [f"bin_mm_{loc}" for loc in LOCI]
    cox_binary = run_cox(df_tr, df_te, binary_feats, "Cox-binary")

    # --- Cox: linear ESM-2 distances (no interaction terms) ---
    logger.info("Running Cox (linear distances)...")
    dist_feats = ["age_norm", "disease_risk"] + [f"dist_{loc}" for loc in LOCI]
    cox_dist = run_cox(df_tr, df_te, dist_feats, "Cox-distances")

    # --- Cox: distances + TRUE interaction terms (oracle ceiling) ---
    # The DGP contains d_DRB1·d_DQB1 (GvHD) and d_DRB1·d_A (TRM).  A linear Cox
    # given these exact product columns is the best a linear model can do — the
    # ceiling that any architecture learning interactions from data should
    # approach.  Validates that the interaction signal is real and recoverable.
    logger.info("Running Cox (distances + true interactions, oracle ceiling)...")
    for d in (df_tr, df_va, df_te, df):
        d["int_DRB1_DQB1"] = d["dist_DRB1"] * d["dist_DQB1"]
        d["int_DRB1_A"] = d["dist_DRB1"] * d["dist_A"]
    oracle_feats = dist_feats + ["int_DRB1_DQB1", "int_DRB1_A"]
    cox_oracle = run_cox(df_tr, df_te, oracle_feats, "Cox-oracle")

    # --- Cox: distances + ALL pairwise interactions (realistic ceiling) ---
    # What a statistician who *suspected* interactions but didn't know the true
    # form would build: all 10 pairwise products.  Tests whether the gain
    # survives without oracle knowledge of which interactions matter.
    logger.info("Running Cox (distances + all pairwise interactions)...")
    pair_cols: list[str] = []
    for a_idx in range(len(LOCI)):
        for b_idx in range(a_idx + 1, len(LOCI)):
            la, lb = LOCI[a_idx], LOCI[b_idx]
            col = f"pair_{la}_{lb}"
            for d in (df_tr, df_va, df_te, df):
                d[col] = d[f"dist_{la}"] * d[f"dist_{lb}"]
            pair_cols.append(col)
    allpair_feats = dist_feats + pair_cols
    cox_allpair = run_cox(df_tr, df_te, allpair_feats, "Cox-allpairs")

    all_results: dict[str, dict] = {
        "Cox (binary mismatch)": cox_binary,
        "Cox (ESM-2 distances)": cox_dist,
        "Cox (distances + true interactions)": cox_oracle,
        "Cox (distances + all pairwise)": cox_allpair,
    }

    # --- CAPA ---
    if not args.no_capa:
        logger.info("Building embedding tensors for CAPA...")
        donor_emb, recip_emb = build_embedding_tensors(df, embs)
        cont, cat = build_clinical_tensors(df, device)
        t_bins = np.floor(
            df["survival_time_days"].values / MAX_DAYS * (N_TIME_BINS - 1)
        ).astype(np.int64).clip(0, N_TIME_BINS - 1)
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
            num_heads=4, num_layers=2, dropout=0.2,
            proj_dim=args.proj_dim,   # 1280 → 64 before MHA: ~330K params vs 27.8M
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("CAPA trainable params: %d (proj_dim=%s)", n_params, args.proj_dim)

        donor_d = donor_emb.to(device)
        recip_d = recip_emb.to(device)

        best_ep = train_capa(
            donor_d[idx_tr_t], recip_d[idx_tr_t],
            cont[idx_tr_t], cat[idx_tr_t],
            times_t[idx_tr_t], types_t[idx_tr_t],
            donor_d[idx_va_t], recip_d[idx_va_t],
            cont[idx_va_t], cat[idx_va_t],
            times_t[idx_va_t], types_t[idx_va_t],
            model, device,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
            alpha=0.0,   # pure NLL — ranking term exp(-Δ/0.1) explodes for confident wrong pairs
            diff_mode=args.diff_mode,
        )

        model.eval()
        with torch.no_grad():
            clin_te = model.clinical_encoder(cont[idx_te_t], cat[idx_te_t])
            te_d = donor_d[idx_te_t] if not args.diff_mode else (donor_d[idx_te_t] - recip_d[idx_te_t]).abs()
            te_r = recip_d[idx_te_t] if not args.diff_mode else te_d
            cif_te = model.cif(te_d, te_r, clin_te).cpu().numpy()

        label = "CAPA (diff self-attn)" if args.diff_mode else "CAPA (cross-attention)"
        all_results[label] = eval_cindex(cif_te, etime_te, etype_te)
        logger.info("CAPA best epoch: %d", best_ep)

    # --- Print results ---
    W = 84
    print("\n" + "=" * W)
    print("HAPLOIDENTICAL SIMULATION — test-set C-index  (n=%d; 80/10/10 split; seed=42)" % args.n)
    print("Outcome: cross-locus interaction (d_DRB1·d_DQB1 for GvHD, d_DRB1·d_A for TRM).")
    print("All patients: binary mismatch all=1 → Cox(binary) reduces to age+disease risk.")
    print("=" * W)
    print(f"{'Model':<32} {'GvHD':>14}  {'Relapse':>14}  {'TRM':>14}")
    print("-" * W)
    for label, res in all_results.items():
        row = f"{label:<32}"
        for name in ["GvHD", "Relapse", "TRM"]:
            r = res.get(name, {})
            c = r.get("cindex", float("nan"))
            lo = r.get("ci_low", float("nan"))
            hi = r.get("ci_hi", float("nan"))
            row += f"  {'—':>14}" if np.isnan(c) else f"  {c:.3f} ({lo:.2f}–{hi:.2f})"
        print(row)
    print("=" * W)

    print("\nΔ C-index  Cox(distances) − Cox(binary)  [ESM-2 main-effect gain]:")
    for name in ["GvHD", "Relapse", "TRM"]:
        c_d = cox_dist.get(name, {}).get("cindex", float("nan"))
        c_b = cox_binary.get(name, {}).get("cindex", float("nan"))
        delta = c_d - c_b if not (np.isnan(c_d) or np.isnan(c_b)) else float("nan")
        marker = " ✓" if not np.isnan(delta) and delta > 0.03 else ""
        print(f"  {name:<10}: {delta:+.3f}{marker}")

    if "CAPA (cross-attention)" in all_results:
        print("\nΔ C-index  CAPA − Cox(distances)  [cross-attention interaction gain]:")
        for name in ["GvHD", "Relapse", "TRM"]:
            c_c = all_results["CAPA (cross-attention)"].get(name, {}).get("cindex", float("nan"))
            c_d = cox_dist.get(name, {}).get("cindex", float("nan"))
            delta = c_c - c_d if not (np.isnan(c_c) or np.isnan(c_d)) else float("nan")
            marker = " ✓" if not np.isnan(delta) and delta > 0.02 else ""
            print(f"  {name:<10}: {delta:+.3f}{marker}")

    out = {
        "n": args.n,
        "mode": "haploidentical",
        "epochs": args.epochs if not args.no_capa else 0,
        "seed": SEED,
        "outcome_model": {
            "GvHD":    "log h = log(1/200000) + 3.5*d_DRB1 + 2.0*d_DQB1 + 2.5*d_DRB1*d_DQB1 + 0.3*age_norm",
            "TRM":     "log h = log(1/150000) + 2.0*d_DRB1 + 2.0*d_A   + 1.5*d_DRB1*d_A     + 1.0*d_C + 0.5*age_norm",
            "Relapse": "log h = log(1/4000)   - 0.8*d_B   + 2.5*disease_risk - 0.3*d_DRB1",
            "note":    "Pool-normalised distances [0,1]. Interaction terms not in Cox(distances) features.",
        },
        "results": {k: v for k, v in all_results.items()},
    }
    out_path = PROJECT_ROOT / "data/results/haplo_simulation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
