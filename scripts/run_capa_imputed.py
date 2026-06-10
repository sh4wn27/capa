#!/usr/bin/env python3
"""Train and evaluate CAPA on UCI BMT with frequency-imputed HLA alleles.

Run impute_hla_alleles.py first to generate data/processed/bmt_imputed_hla.csv.

This script demonstrates end-to-end CAPA pipeline functionality on real
transplant outcomes. Allele assignments are frequency-based imputations
(no outcome-specific signal), so CAPA performance is expected to equal
the clinical-only baseline (~0.78 relapse C-index on the held-out test set).
This establishes a lower bound; real allele typing data would be required
to realise ESM-2's potential contribution.

Usage
-----
    uv run python scripts/run_capa_imputed.py [--epochs 150] [--device cpu]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.model.capa_model import CAPAModel, DISEASE_CATEGORIES, STEM_CELL_SOURCE_CATEGORIES
from capa.model.losses import deephit_loss
from capa.training.evaluate import concordance_index, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match run_real_baselines.py for comparability)
# ---------------------------------------------------------------------------
MAX_DAYS = 730.0
N_TIME_BINS = 100
EVAL_TIMES = np.array([182.5, 365.0, 547.5])
T365_BIN = int(365.0 / MAX_DAYS * (N_TIME_BINS - 1))
N_BOOTSTRAP = 1000
SEED = 42
LOCI = ["A", "B", "C", "DRB1", "DQB1"]

DISEASE_MAP: dict[str, int] = {v: i for i, v in enumerate(DISEASE_CATEGORIES)}
SC_MAP: dict[int, int] = {0: 1, 1: 2}  # BMT raw 0=BM→idx1, 1=PBSC→idx2

EMB_PATH = PROJECT_ROOT / "data/processed/hla_embeddings.h5"
CSV_PATH = PROJECT_ROOT / "data/processed/bmt_imputed_hla.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings() -> dict[str, np.ndarray]:
    """Load pre-computed ESM-2 embeddings from HDF5 cache."""
    embs: dict[str, np.ndarray] = {}
    with h5py.File(EMB_PATH, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    logger.info("Loaded %d allele embeddings (dim=%d)", len(embs), next(iter(embs.values())).shape[0])
    return embs


def build_embedding_tensors(
    df: pd.DataFrame,
    embs: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build donor and recipient embedding matrices.

    Each patient gets a (n_loci, emb_dim) tensor produced by
    mean-pooling the two alleles at each locus.
    """
    emb_dim = next(iter(embs.values())).shape[0]
    n = len(df)
    donor_arr = np.zeros((n, len(LOCI), emb_dim), dtype=np.float32)
    recip_arr = np.zeros((n, len(LOCI), emb_dim), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, loc in enumerate(LOCI):
            d1 = embs.get(row[f"donor_{loc}_1"], np.zeros(emb_dim))
            d2 = embs.get(row[f"donor_{loc}_2"], np.zeros(emb_dim))
            r1 = embs.get(row[f"recipient_{loc}_1"], np.zeros(emb_dim))
            r2 = embs.get(row[f"recipient_{loc}_2"], np.zeros(emb_dim))
            donor_arr[i, j] = (d1 + d2) / 2.0
            recip_arr[i, j] = (r1 + r2) / 2.0

    return torch.from_numpy(donor_arr), torch.from_numpy(recip_arr)


def build_clinical_tensors(
    df: pd.DataFrame,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cont, cat_indices) tensors for ClinicalEncoder."""
    n = len(df)
    cont = np.zeros((n, 4), dtype=np.float32)
    cat_idx = np.zeros((n, 4), dtype=np.int64)

    for i, (_, row) in enumerate(df.iterrows()):
        cont[i, 0] = float(row.get("recipient_age", 0) or 0) / 100.0
        cont[i, 1] = float(row.get("donor_age", 0) or 0) / 100.0
        cont[i, 2] = float(row.get("cd34_dose", 0) or 0) / 10.0
        cont[i, 3] = float(row.get("sex_mismatch_f2m", 0) or 0)

        # Disease: map string to DISEASE_CATEGORIES index
        raw_dis = str(row.get("disease", "")).strip()
        dis_str = {"chronic": "CML", "lymphoma": "NHL", "nonmalignant": "other"}.get(raw_dis, raw_dis)
        cat_idx[i, 0] = DISEASE_MAP.get(dis_str, 0)

        # Conditioning: not in BMT → unknown (0)
        cat_idx[i, 1] = 0

        # Donor type: not in BMT → unknown (0)
        cat_idx[i, 2] = 0

        # Stem cell source: 0=BM, 1=PBSC
        sc_raw = int(row.get("stem_cell_source", 0) or 0)
        cat_idx[i, 3] = SC_MAP.get(sc_raw, 0)

    return (
        torch.from_numpy(cont).to(device),
        torch.from_numpy(cat_idx).to(device),
    )


def make_event_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    relapse = df["relapse"].fillna(0).astype(int).values
    dead    = df["dead"].fillna(0).astype(int).values
    gvhd    = df["acute_gvhd_iii_iv"].fillna(0).astype(int).values
    etime   = np.clip(df["survival_time_days"].astype(float).values, 0.0, MAX_DAYS)

    etype = np.zeros(len(df), dtype=np.int64)
    etype[(gvhd == 1) & (relapse == 0) & (dead == 0)] = 1
    etype[(dead == 1) & (relapse == 0)] = 3
    etype[relapse == 1] = 2
    return etype, etime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_capa(
    donor_tr: torch.Tensor,
    recip_tr: torch.Tensor,
    cont_tr: torch.Tensor,
    cat_tr: torch.Tensor,
    times_tr: torch.Tensor,
    types_tr: torch.Tensor,
    model: CAPAModel,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
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
            b = idx[start : start + batch_size]
            b_t = torch.from_numpy(b).long()

            d_emb   = donor_tr[b_t].to(device)
            r_emb   = recip_tr[b_t].to(device)
            cont_b  = cont_tr[b_t]
            cat_b   = cat_tr[b_t]
            t_b     = times_tr[b_t].to(device)
            ev_b    = types_tr[b_t].to(device)

            clin = model.clinical_encoder(cont_b, cat_b)
            logits = model(d_emb, r_emb, clin)
            loss = deephit_loss(logits, t_b, ev_b, alpha=0.5, sigma=0.1)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            n_batches += 1

        if ep % 25 == 0:
            logger.info("Epoch %3d/%d  loss=%.4f", ep, epochs, ep_loss / n_batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not CSV_PATH.exists():
        logger.error("Missing %s — run impute_hla_alleles.py first", CSV_PATH)
        sys.exit(1)

    # --- Load data ---
    df = pd.read_csv(CSV_PATH)
    logger.info("Loaded imputed dataset: %d patients", len(df))

    embs = load_embeddings()
    donor_emb, recip_emb = build_embedding_tensors(df, embs)
    cont, cat = build_clinical_tensors(df, device)
    etype, etime = make_event_labels(df)

    # Time → bin index
    t_bins = np.floor(etime / MAX_DAYS * (N_TIME_BINS - 1)).astype(np.int64).clip(0, N_TIME_BINS - 1)
    times_t = torch.from_numpy(t_bins).long()
    types_t = torch.from_numpy(etype).long()

    # --- Stratified 80/20 split (seed=42, stratify on event type) ---
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED,
        stratify=etype,
    )
    logger.info("Train: %d  Test: %d", len(idx_tr), len(idx_te))

    # --- Build CAPA model ---
    emb_dim = donor_emb.shape[-1]
    model = CAPAModel(
        embedding_dim=emb_dim,
        loci=LOCI,
        clinical_dim=32,
        interaction_dim=128,
        survival_type="deephit",
        num_events=3,
        time_bins=N_TIME_BINS,
        event_names=["GvHD", "Relapse", "TRM"],
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CAPAModel  trainable params: %d", n_params)

    idx_tr_t = torch.from_numpy(idx_tr).long()
    idx_te_t = torch.from_numpy(idx_te).long()

    # Move embedding tensors to device for training
    donor_emb_d = donor_emb.to(device)
    recip_emb_d = recip_emb.to(device)

    # --- Train ---
    train_capa(
        donor_emb_d[idx_tr_t],
        recip_emb_d[idx_tr_t],
        cont[idx_tr_t],
        cat[idx_tr_t],
        times_t[idx_tr_t],
        types_t[idx_tr_t],
        model,
        device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # --- Evaluate on test set ---
    model.eval()
    with torch.no_grad():
        d_te = donor_emb_d[idx_te_t]
        r_te = recip_emb_d[idx_te_t]
        clin_te = model.clinical_encoder(cont[idx_te_t], cat[idx_te_t])
        cif_te = model.cif(d_te, r_te, clin_te).cpu().numpy()  # (n_te, K, T)

    etime_te = etime[idx_te]
    etype_te = etype[idx_te]

    EVENT_NAMES = ["GvHD", "Relapse", "TRM"]
    print("\n" + "=" * 70)
    print("CAPA (frequency-imputed HLA) — test-set results")
    print("=" * 70)
    print(f"{'Event':<12} {'C-index':>8}  {'95% CI':>18}")
    print("-" * 70)

    results: dict[str, dict] = {}
    for k_ev, name in enumerate(EVENT_NAMES):
        risks = cif_te[:, k_ev, T365_BIN].astype(np.float64)
        obs   = (etype_te == (k_ev + 1)).astype(bool)
        m = bootstrap_ci(
            concordance_index,
            etime_te.astype(np.float64),
            risks,
            obs,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )
        print(f"{name:<12} {m.value:>8.3f}  ({m.ci_lower:.3f}, {m.ci_upper:.3f})")
        results[name] = {"cindex": m.value, "ci_low": m.ci_lower, "ci_hi": m.ci_upper}

    print("=" * 70)
    print("\nNOTE: Imputed alleles carry no outcome-specific signal.")
    print("Expected performance ≈ clinical-only baseline (Relapse ~0.78 on this split).")
    print("This run validates end-to-end CAPA pipeline functionality.\n")

    return results


if __name__ == "__main__":
    main()
