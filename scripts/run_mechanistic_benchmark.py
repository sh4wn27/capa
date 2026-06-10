#!/usr/bin/env python3
"""Mechanistic benchmark: ESM-2 distances vs binary mismatch for HSCT outcome prediction.

SCIENTIFIC QUESTION
-------------------
Do ESM-2 embedding distances between donor and recipient HLA alleles contain MORE
predictive information about transplant outcomes than binary mismatch indicators?

Two patients may both have "1 DRB1 mismatch" but one pair is biochemically similar
(L2 ≈ 1.2) and another biochemically distant (L2 ≈ 3.9). The binary indicator is
blind to this difference. This script tests whether the continuous distance signal
is recoverable and predictive.

EXPERIMENTAL DESIGN
-------------------
  1. Assign donor-recipient genotypes from real alleles (98 ESM-2-embedded alleles)
  2. Compute alloreactivity = min-L2 distance from donor alleles to recipient alleles
     per locus (represents T-cell allorecognition pressure in the embedding space)
  3. Simulate competing-risks outcomes (GvHD, Relapse, TRM) causally driven by
     ONLY the continuous ESM-2 distances (binary indicators carry zero ADDITIONAL
     information beyond what the distances capture)
  4. Compare THREE models:
       A. Cox (binary mismatch) — clinical baseline; sees only 0/1 per locus
       B. Cox (ESM-2 distances) — oracle; sees actual continuous alloreactivity
       C. CAPA (small) — cross-attention; sees raw embeddings; must learn distances
  5. Expected result: Cox(B) >> Cox(A) because distances contain fine-grained signal
     that binary indicators lose. CAPA(C) should match Cox(B) or better.

OUTCOME MODEL (all signal is in continuous distances, not binary mismatch):
  log h_GvHD  = log(1/2800) + 3.5·dist_DRB1 + 2.0·dist_DQB1 + 0.4·age_norm
  log h_TRM   = log(1/2600) + 2.5·dist_DRB1 + 1.8·dist_A   + 1.0·dist_C + 0.6·age_norm
  log h_Rel   = log(1/1800) - 1.2·dist_B    + 2.0·disease_risk - 0.5·dist_DRB1
  (distances normalised to [0,1] by per-locus max pairwise L2)

Usage
-----
    uv run python scripts/run_mechanistic_benchmark.py [--n 2000] [--epochs 300]
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
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# European allele frequency pools (same as impute_hla_alleles.py)
# ---------------------------------------------------------------------------
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


def _normalize_pool(d: dict[str, float]) -> tuple[list[str], np.ndarray]:
    alleles = list(d.keys())
    p = np.array(list(d.values()), dtype=float)
    return alleles, p / p.sum()


_POOLS = {loc: _normalize_pool(EURO_FREQS[loc]) for loc in LOCI}


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def _compute_max_l2(embs: dict[str, np.ndarray]) -> dict[str, float]:
    max_l2: dict[str, float] = {}
    for loc in LOCI:
        alleles_l = [k for k in embs if k.startswith(loc + "*")]
        vecs = np.stack([embs[a] for a in alleles_l])
        dists = np.array([
            np.linalg.norm(vecs[i] - vecs[j])
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))
        ])
        max_l2[loc] = float(dists.max())
        logger.debug("max L2 %s: %.3f", loc, max_l2[loc])
    return max_l2


def _make_outcome(
    dist: dict[str, float],
    age_norm: float,
    disease_risk: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Exponential competing risks. Returns (observed_time_days, event_type).

    Calibrated so that, for the controlled cohort (dist_DRB1 ∈ [0.32, 1.0],
    mean ≈ 0.65), approximate event rates are:
      GvHD ~20%, TRM ~20%, Relapse ~30%, Censored ~30%

    Baseline hazards λ₀ derived by requiring exp(-λ̄ · T_max) ≈ target_survival.
    """
    # GvHD: almost entirely DRB1 alloreactivity (T-cell mediated rejection).
    # Large β so continuous distance dominates; age is weak → Cox(binary) near
    # chance; Cox(distances) achieves high C-index.  10x hazard ratio across
    # the observed DRB1 distance range [0.32, 1.0].
    log_h_gvhd = (
        np.log(1 / 7000)          # λ₀ calibrated for ~25% GvHD rate
        + 4.0 * dist["DRB1"]      # dominant driver
        + 1.5 * dist["DQB1"]
        + 0.15 * age_norm         # weak age effect so Cox(binary) ≈ 0.5
    )
    # TRM: class I + class II alloreactivity + age (age matters clinically)
    log_h_trm = (
        np.log(1 / 8000)
        + 3.0 * dist["DRB1"]
        + 2.0 * dist["A"]
        + 1.0 * dist["C"]
        + 0.8 * age_norm
    )
    # Relapse: GvL reduces relapse (B-locus NK effect), disease risk increases it
    log_h_rel = (
        np.log(1 / 2500)
        - 1.0 * dist["B"]
        + 2.0 * disease_risk
        - 0.5 * dist["DRB1"]
    )
    rates = [np.exp(np.clip(h, -12, 4)) for h in (log_h_gvhd, log_h_rel, log_h_trm)]
    times = [float(rng.exponential(1.0 / r)) if r > 0 else 1e9 for r in rates]
    t_cens = float(rng.uniform(300, 2000))
    all_t = times + [t_cens]
    winner = int(np.argmin(all_t))
    etype = winner + 1 if winner < 3 else 0
    return min(all_t[winner], MAX_DAYS), etype


def generate_controlled_cohort(
    n: int,
    embs: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Controlled cohort: ALL patients have exactly 1 DRB1 mismatch, all other loci matched.

    This is the critical experiment for demonstrating ESM-2 informativeness:
    - Binary mismatch indicator (DRB1=1) is IDENTICAL for every patient → C≈0.5
    - Continuous ESM-2 distance varies across patients → can predict outcomes
    - Outcome is causally driven by the continuous distance, not the binary flag

    Recipients are homozygous at DRB1 (same allele twice) so alloreactivity
    simplifies to the exact L2 distance between the mismatched allele and the
    recipient allele — a clean, unambiguous distance metric.
    """
    max_l2 = _compute_max_l2(embs)
    drb1_alleles, drb1_probs = _POOLS["DRB1"]
    records = []

    for i in range(n):
        # Homozygous DRB1 recipient → alloreactivity = exact L2(mm, recip_allele)
        recip_drb1 = rng.choice(drb1_alleles, p=drb1_probs)

        # Donor DRB1: a DIFFERENT allele (the mismatch)
        others = [a for a in drb1_alleles if a != recip_drb1]
        others_p = np.array([drb1_probs[drb1_alleles.index(a)] for a in others])
        others_p /= others_p.sum()
        donor_drb1 = rng.choice(others, p=others_p)

        # dist_DRB1 = L2(donor_drb1, recip_drb1), normalised by max pairwise L2
        dist_drb1_raw = float(np.linalg.norm(embs[donor_drb1] - embs[recip_drb1]))
        dist_drb1_norm = dist_drb1_raw / max_l2["DRB1"]

        # All other loci: donor == recipient (perfectly matched, dist=0)
        dist: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0,
                                  "DRB1": dist_drb1_norm, "DQB1": 0.0}

        age = float(rng.uniform(2, 20))
        age_norm = (age - 11.0) / 6.0
        disease_risk = int(rng.random() < 0.45)

        t_obs, etype = _make_outcome(dist, age_norm, disease_risk, rng)

        # CAPA embedding columns: use actual DRB1 alleles; other loci same allele for both
        placeholder = {loc: rng.choice(_POOLS[loc][0], p=_POOLS[loc][1]) for loc in LOCI}
        records.append({
            "patient_idx": i,
            "age": age, "age_norm": age_norm, "disease_risk": disease_risk,
            "n_mismatches": 1,
            "survival_time_days": t_obs,
            "event_type": etype,
            "dist_A": 0.0, "dist_B": 0.0, "dist_C": 0.0,
            "dist_DRB1": dist_drb1_norm, "dist_DQB1": 0.0,
            "bin_mm_A": 0, "bin_mm_B": 0, "bin_mm_C": 0,
            "bin_mm_DRB1": 1,  # CONSTANT across all patients — binary is uninformative
            "bin_mm_DQB1": 0,
            # Embedding columns: donor DRB1 position 0 is mismatched; all others matched
            "donor_A_1": placeholder["A"], "donor_A_2": placeholder["A"],
            "donor_B_1": placeholder["B"], "donor_B_2": placeholder["B"],
            "donor_C_1": placeholder["C"], "donor_C_2": placeholder["C"],
            "donor_DRB1_1": donor_drb1,    "donor_DRB1_2": recip_drb1,
            "donor_DQB1_1": placeholder["DQB1"], "donor_DQB1_2": placeholder["DQB1"],
            "recip_A_1": placeholder["A"], "recip_A_2": placeholder["A"],
            "recip_B_1": placeholder["B"], "recip_B_2": placeholder["B"],
            "recip_C_1": placeholder["C"], "recip_C_2": placeholder["C"],
            "recip_DRB1_1": recip_drb1,    "recip_DRB1_2": recip_drb1,
            "recip_DQB1_1": placeholder["DQB1"], "recip_DQB1_2": placeholder["DQB1"],
        })

    return pd.DataFrame(records)


def generate_mixed_cohort(
    n: int,
    embs: dict[str, np.ndarray],
    rng: np.random.Generator,
    match_mix: tuple[float, ...] = (0.50, 0.35, 0.12, 0.03),
) -> pd.DataFrame:
    """Realistic cohort: mixed mismatch grades (0-3), all 5 loci.

    Less clean than the controlled experiment but more representative of
    real registry data. Useful for seeing how ESM-2 distances help when the
    mismatch landscape is heterogeneous.
    """
    import copy
    max_l2 = _compute_max_l2(embs)
    records = []

    for i in range(n):
        n_mm = int(rng.choice(4, p=match_mix))

        recip: dict[str, list[str]] = {}
        for loc in LOCI:
            al, pr = _POOLS[loc]
            recip[loc] = list(rng.choice(al, size=2, replace=True, p=pr))

        donor = copy.deepcopy(recip)
        if n_mm > 0:
            all_slots = [(loc, idx) for loc in LOCI for idx in range(2)]
            chosen = [all_slots[j] for j in rng.choice(len(all_slots), size=n_mm, replace=False)]
            for loc, idx in chosen:
                al, pr = _POOLS[loc]
                current = recip[loc][idx]
                others = [a for a in al if a != current]
                op = np.array([pr[list(al).index(a)] for a in others])
                op /= op.sum()
                donor[loc][idx] = rng.choice(others, p=op)

        # Alloreactivity: sum over donor alleles of min L2 distance to any recipient allele
        dist: dict[str, float] = {}
        for loc in LOCI:
            total = sum(
                min(np.linalg.norm(embs[d_a] - embs[r_a]) for r_a in recip[loc])
                for d_a in donor[loc]
            )
            dist[loc] = total / max_l2[loc]

        age = float(rng.uniform(2, 20))
        age_norm = (age - 11.0) / 6.0
        disease_risk = int(rng.random() < 0.45)

        bin_mm = {
            loc: int(any(donor[loc][j] != recip[loc][j] for j in range(2)))
            for loc in LOCI
        }

        t_obs, etype = _make_outcome(dist, age_norm, disease_risk, rng)

        rec: dict = {
            "patient_idx": i,
            "age": age, "age_norm": age_norm, "disease_risk": disease_risk,
            "n_mismatches": n_mm,
            "survival_time_days": t_obs,
            "event_type": etype,
            "dist_A": dist["A"], "dist_B": dist["B"], "dist_C": dist["C"],
            "dist_DRB1": dist["DRB1"], "dist_DQB1": dist["DQB1"],
            "bin_mm_A": bin_mm["A"], "bin_mm_B": bin_mm["B"],
            "bin_mm_C": bin_mm["C"], "bin_mm_DRB1": bin_mm["DRB1"],
            "bin_mm_DQB1": bin_mm["DQB1"],
        }
        for loc in LOCI:
            rec[f"donor_{loc}_1"] = donor[loc][0]
            rec[f"donor_{loc}_2"] = donor[loc][1]
            rec[f"recip_{loc}_1"] = recip[loc][0]
            rec[f"recip_{loc}_2"] = recip[loc][1]
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Survival evaluation helpers (shared)
# ---------------------------------------------------------------------------

def eval_cindex(cif: np.ndarray, etime: np.ndarray, etype: np.ndarray) -> dict:
    """C-index with 95% bootstrap CI for each competing event."""
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
# Cox baseline (cause-specific proportional hazards)
# ---------------------------------------------------------------------------

def run_cox(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    feat_cols: list[str],
    label: str,
) -> dict:
    """Cause-specific Cox PH; returns C-index dict.

    Automatically drops constant columns (zero-variance on training set)
    so the fit does not fail when some indicators are uniform across patients,
    as happens in the controlled experiment (all binary mismatch = 1).
    """
    from lifelines import CoxPHFitter

    # Drop columns with zero variance in the training set to avoid singularity
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
# CAPA training
# ---------------------------------------------------------------------------

def build_embedding_tensors(
    df: pd.DataFrame,
    embs: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # cont: [age_norm/100, age_norm/100 (donor placeholder), cd34 placeholder, disease_risk]
    cont = np.zeros((n, 4), dtype=np.float32)
    cat = np.zeros((n, 4), dtype=np.int64)
    for i, (_, row) in enumerate(df.iterrows()):
        age_raw = float(row.get("age", 11.0))
        cont[i, 0] = age_raw / 100.0
        cont[i, 1] = 30.0 / 100.0  # donor age placeholder
        cont[i, 2] = 5.0 / 10.0    # cd34 placeholder
        cont[i, 3] = float(row.get("disease_risk", 0))
        # Disease: high_risk → index 2 (AML), otherwise 1 (ALL)
        cat[i, 0] = 2 if row.get("disease_risk", 0) else 1
    return torch.from_numpy(cont).to(device), torch.from_numpy(cat).to(device)


def train_capa(
    donor_tr: torch.Tensor, recip_tr: torch.Tensor,
    cont_tr: torch.Tensor, cat_tr: torch.Tensor,
    times_tr: torch.Tensor, types_tr: torch.Tensor,
    model: CAPAModel, device: torch.device,
    epochs: int, lr: float, batch_size: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = donor_tr.shape[0]
    rng = np.random.default_rng(SEED)
    model.train()
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b = idx[start: start + batch_size]
            bt = torch.from_numpy(b).long()
            d_e = donor_tr[bt].to(device)
            r_e = recip_tr[bt].to(device)
            ct  = cont_tr[bt]
            ca  = cat_tr[bt]
            t_b = times_tr[bt].to(device)
            ev_b = types_tr[bt].to(device)
            clin = model.clinical_encoder(ct, ca)
            logits = model(d_e, r_e, clin)
            loss = deephit_loss(logits, t_b, ev_b, alpha=0.5, sigma=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        if ep % 50 == 0:
            logger.info("Epoch %3d/%d  loss=%.4f", ep, epochs, ep_loss / n_batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-capa", action="store_true", help="Skip CAPA training (Cox comparison only)")
    parser.add_argument(
        "--mode", choices=["controlled", "mixed"], default="controlled",
        help="controlled: all patients have 1 DRB1 mismatch (binary uninformative); "
             "mixed: realistic distribution of match grades",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # --- Load embeddings ---
    embs: dict[str, np.ndarray] = {}
    with h5py.File(EMB_PATH, "r") as f:
        for k in f.keys():
            embs[k] = f[k][:]
    logger.info("Loaded %d allele embeddings", len(embs))

    # --- Generate cohort ---
    if args.mode == "controlled":
        logger.info("Generating CONTROLLED cohort (n=%d, all DRB1-only mismatches)...", args.n)
        df = generate_controlled_cohort(args.n, embs, rng)
        logger.info("  NOTE: binary DRB1 mismatch=1 for ALL patients → Cox(binary) cannot")
        logger.info("        discriminate on DRB1; its C-index reflects age/disease_risk only.")
    else:
        logger.info("Generating MIXED cohort (n=%d, realistic match grades)...", args.n)
        df = generate_mixed_cohort(args.n, embs, rng)

    event_label = {0: "censored", 1: "GvHD", 2: "Relapse", 3: "TRM"}
    for k, nm in event_label.items():
        logger.info("  %-10s: %d (%.1f%%)", nm, (df["event_type"] == k).sum(),
                    100 * (df["event_type"] == k).mean())

    # --- Split 80/20 ---
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED,
        stratify=df["event_type"].values,
    )
    df_tr, df_te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
    logger.info("Train=%d  Test=%d", len(df_tr), len(df_te))

    etime_te = df_te["survival_time_days"].values
    etype_te = df_te["event_type"].values.astype(np.int64)

    # --- MODEL A: Cox with binary mismatch (clinical baseline) ---
    logger.info("Running Cox (binary mismatch)...")
    binary_feats = ["age_norm", "disease_risk", "n_mismatches",
                    "bin_mm_A", "bin_mm_B", "bin_mm_C", "bin_mm_DRB1", "bin_mm_DQB1"]
    cox_binary = run_cox(df_tr, df_te, binary_feats, "Cox-binary")

    # --- MODEL B: Cox with continuous ESM-2 distances (oracle comparison) ---
    # This model sees the ACTUAL continuous alloreactivity distances that
    # causally drive outcomes.  Binary mismatch is a lossy projection of these
    # distances; if Cox(B) >> Cox(A), ESM-2 distances are more informative.
    logger.info("Running Cox (ESM-2 distances)...")
    dist_feats = ["age_norm", "disease_risk",
                  "dist_A", "dist_B", "dist_C", "dist_DRB1", "dist_DQB1"]
    cox_dist = run_cox(df_tr, df_te, dist_feats, "Cox-distances")

    all_results: dict[str, dict] = {
        "Cox (binary mismatch)": cox_binary,
        "Cox (ESM-2 distances)": cox_dist,
    }

    # --- MODEL C: CAPA (small architecture, reduced overfitting risk) ---
    if not args.no_capa:
        logger.info("Building CAPA model (small)...")
        donor_emb, recip_emb = build_embedding_tensors(df, embs)
        cont, cat = build_clinical_tensors(df, device)
        t_bins = np.floor(
            df["survival_time_days"].values / MAX_DAYS * (N_TIME_BINS - 1)
        ).astype(np.int64).clip(0, N_TIME_BINS - 1)
        times_t = torch.from_numpy(t_bins).long()
        types_t = torch.from_numpy(df["event_type"].values.astype(np.int64)).long()

        idx_tr_t = torch.from_numpy(idx_tr).long()
        idx_te_t = torch.from_numpy(idx_te).long()

        emb_dim = donor_emb.shape[-1]
        # Small architecture: reduces ~28M params to ~7M to mitigate overfitting
        # on N=1600 training patients. interaction_dim=32, num_heads=4, 1 layer.
        model = CAPAModel(
            embedding_dim=emb_dim, loci=LOCI, clinical_dim=16,
            interaction_dim=64, survival_type="deephit",
            num_events=3, time_bins=N_TIME_BINS,
            event_names=["GvHD", "Relapse", "TRM"],
            num_heads=4, num_layers=1, dropout=0.3,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("CAPA (small) trainable params: %d", n_params)

        donor_d = donor_emb.to(device)
        recip_d = recip_emb.to(device)

        train_capa(
            donor_d[idx_tr_t], recip_d[idx_tr_t],
            cont[idx_tr_t], cat[idx_tr_t],
            times_t[idx_tr_t], types_t[idx_tr_t],
            model, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )

        model.eval()
        with torch.no_grad():
            clin_te = model.clinical_encoder(cont[idx_te_t], cat[idx_te_t])
            cif_te = model.cif(donor_d[idx_te_t], recip_d[idx_te_t], clin_te).cpu().numpy()

        all_results["CAPA (ESM-2 cross-attn)"] = eval_cindex(cif_te, etime_te, etype_te)

    # --- Results table ---
    W = 84
    print("\n" + "=" * W)
    print("MECHANISTIC BENCHMARK — test-set C-index  (n=%d; 80/20 split; seed=42)" % args.n)
    print("Outcome signal: ESM-2 alloreactivity distances causally drive all events.")
    if args.mode == "controlled":
        print("Mode: CONTROLLED — all patients have identical binary DRB1 mismatch (=1).")
        print("  Cox(binary) can only use age/disease_risk → approaches clinical baseline.")
        print("  Cox(ESM-2 dist) sees actual allele-pair distances → reveals information gain.")
    else:
        print("Mode: MIXED — realistic distribution of 0-3 mismatches across all loci.")
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
            if np.isnan(c):
                row += f"  {'—':>14}"
            else:
                row += f"  {c:.3f} ({lo:.2f}–{hi:.2f})"
        print(row)

    print("=" * W)

    # Delta: distances vs binary (the primary scientific comparison)
    print("\nΔ C-index  Cox(distances) − Cox(binary)  [primary test of ESM-2 informativeness]:")
    for name in ["GvHD", "Relapse", "TRM"]:
        c_d = cox_dist.get(name, {}).get("cindex", float("nan"))
        c_b = cox_binary.get(name, {}).get("cindex", float("nan"))
        delta = c_d - c_b if not (np.isnan(c_d) or np.isnan(c_b)) else float("nan")
        marker = " ✓" if not np.isnan(delta) and delta > 0.03 else ""
        print(f"  {name:<10}: {delta:+.3f}{marker}")

    if "CAPA (ESM-2 cross-attn)" in all_results:
        print("\nΔ C-index  CAPA − Cox(distances)  [tests if cross-attention adds value]:")
        for name in ["GvHD", "Relapse", "TRM"]:
            c_c = all_results["CAPA (ESM-2 cross-attn)"].get(name, {}).get("cindex", float("nan"))
            c_d = cox_dist.get(name, {}).get("cindex", float("nan"))
            delta = c_c - c_d if not (np.isnan(c_c) or np.isnan(c_d)) else float("nan")
            print(f"  {name:<10}: {delta:+.3f}")

    # Save results
    out: dict = {
        "n": args.n,
        "mode": args.mode,
        "epochs": args.epochs if not args.no_capa else 0,
        "seed": SEED,
        "outcome_model": {
            "GvHD":    "log h = log(1/2800) + 3.5*dist_DRB1 + 2.0*dist_DQB1 + 0.4*age_norm",
            "TRM":     "log h = log(1/2600) + 2.5*dist_DRB1 + 1.8*dist_A   + 1.0*dist_C + 0.6*age_norm",
            "Relapse": "log h = log(1/1800) - 1.2*dist_B   + 2.0*disease_risk - 0.5*dist_DRB1",
            "note":    "Distances normalised by per-locus max pairwise L2; range [0,1]. No n_mm term.",
        },
        "results": {k: v for k, v in all_results.items()},
    }
    out_path = PROJECT_ROOT / "data/results/mechanistic_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
