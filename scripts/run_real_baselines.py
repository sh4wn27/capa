#!/usr/bin/env python3
"""Run competing-risks baselines on the real UCI BMT dataset.

Trains Cox PH (cause-specific), Fine-Gray subdistribution hazard, and a
flat-feature DeepHit MLP on the real 187-patient UCI BMT cohort.  Evaluates
with time-dependent C-index and Integrated Brier Score (1000-bootstrap 95 % CI).

Outputs
-------
- Printed results table to stdout
- JSON results to data/results/real_baselines.json
- Paper-ready LaTeX snippet printed to stdout

Usage
-----
    uv run python scripts/run_real_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capa.data.loader import load_bmt
from capa.data.splits import make_splits
from capa.model.baselines import CoxPHBaseline, FineGrayBaseline
from capa.training.evaluate import (
    MetricWithCI,
    bootstrap_ci,
    concordance_index,
    integrated_brier_score,
    brier_score,
    evaluate_all,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EVENTS = 3
EVENT_NAMES = ["GvHD", "Relapse", "TRM"]
# event_type 1=GvHD, 2=Relapse, 3=TRM, 0=censored
MAX_DAYS = 730.0
N_TIME_BINS = 100
TIME_BINS = np.linspace(0.0, MAX_DAYS, N_TIME_BINS)
EVAL_TIMES = np.array([182.5, 365.0, 547.5])  # 6 mo, 1 yr, 18 mo
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Event encoding
# ---------------------------------------------------------------------------

def make_event_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Derive competing-risk event types and observation times.

    Priority hierarchy (highest overrides):
      1 (highest) – Relapse      → event_type = 2
      2           – TRM          → event_type = 3  (dead, no relapse)
      3 (lowest)  – GvHD III/IV  → event_type = 1  (alive, no relapse)
      0           – Censored

    Time is survival_time_days (overall observation endpoint).
    """
    n = len(df)
    etype = np.zeros(n, dtype=np.int64)
    etime = df["survival_time_days"].astype(float).values.copy()

    relapse = df["relapse"].fillna(0).astype(int).values
    dead    = df["dead"].fillna(0).astype(int).values
    gvhd    = df["acute_gvhd_iii_iv"].fillna(0).astype(int).values

    # Priority 3 (GvHD)
    mask_gvhd = (gvhd == 1) & (relapse == 0) & (dead == 0)
    etype[mask_gvhd] = 1

    # Priority 2 (TRM — overrides GvHD)
    mask_trm = (dead == 1) & (relapse == 0)
    etype[mask_trm] = 3

    # Priority 1 (Relapse — overrides all)
    etype[relapse == 1] = 2

    # Clamp times to MAX_DAYS
    etime = np.clip(etime, 0.0, MAX_DAYS)

    counts = {i: int((etype == i).sum()) for i in range(4)}
    logger.info("Event distribution (0=cens,1=GvHD,2=rel,3=TRM): %s", counts)
    return etype, etime


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a fully numeric feature matrix for tabular baselines.

    Features included
    -----------------
    Continuous (4): recipient_age, donor_age, log1p(cd34_dose), body_mass
    Binary (9): sex, stem_cell_source, malignant_disease, high_risk,
                sex_mismatch_f2m, donor_cmv, recipient_cmv, abo_match,
                retransplant_after_relapse
    HLA (4): hla_match_score, hla_mismatched, n_antigen_mismatches+1,
             n_allele_mismatches+1
    Disease one-hot (4): ALL, AML, chronic, lymphoma
        (nonmalignant is the reference category)
    """
    feats = pd.DataFrame(index=df.index)

    # --- Continuous ---
    feats["recipient_age"] = df["recipient_age"].astype(float)
    feats["donor_age"]     = df["donor_age"].astype(float)
    cd34 = df["cd34_dose"].astype(float)
    feats["log_cd34_dose"] = np.log1p(cd34.fillna(cd34.median()))
    bm = df["recipient_body_mass_kg"].astype(float)
    feats["recipient_body_mass"] = bm.fillna(bm.median())

    # --- Binary clinical ---
    binary_cols = [
        "recipient_sex", "stem_cell_source", "malignant_disease",
        "high_risk", "sex_mismatch_f2m", "donor_cmv", "recipient_cmv",
        "abo_match", "retransplant_after_relapse",
    ]
    for col in binary_cols:
        feats[col] = df[col].fillna(0).astype(float)

    # --- HLA (aggregate scores only — dataset has no per-allele strings) ---
    feats["hla_match_score"]    = df["hla_match_score"].fillna(0).astype(float)
    feats["hla_mismatched"]     = df["hla_mismatched"].fillna(0).astype(float)
    # sentinel -1 ≡ 0 mismatches; add 1 to shift to [0, …]
    feats["n_antigen_mm"] = df["n_antigen_mismatches"].fillna(-1).astype(float) + 1.0
    feats["n_allele_mm"]  = df["n_allele_mismatches"].fillna(-1).astype(float) + 1.0

    # --- Disease one-hot ---
    for cat in ["ALL", "AML", "chronic", "lymphoma"]:
        feats[f"disease_{cat}"] = (df["disease"].astype(str) == cat).astype(float)

    return feats


# ---------------------------------------------------------------------------
# Simple flat-feature DeepHit MLP (no cross-attention; no allele strings)
# ---------------------------------------------------------------------------

class DeepHitMLP(nn.Module):
    """Flat-feature DeepHit: clinical + HLA scores → joint survival PMF."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_events: int = 3,
        time_bins: int = 100,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_events = num_events
        self.time_bins = time_bins
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, num_events * time_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        logits = self.head(h)                      # (B, K*T)
        B = x.shape[0]
        joint = F.softmax(logits.view(B, -1), dim=-1)
        joint = joint.view(B, self.num_events, self.time_bins)
        return joint                               # (B, K, T)

    def cif(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(self.forward(x), dim=2)  # (B, K, T)


def _deephit_loss(
    joint: torch.Tensor,       # (B, K, T)
    times: torch.Tensor,       # (B,) int  bin index
    types: torch.Tensor,       # (B,) int  0=cens 1..K=event
    alpha: float = 0.5,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Combined NLL + pairwise ranking loss (DeepHit, Lee et al. 2018)."""
    B, K, T = joint.shape
    eps = 1e-8

    # ── NLL ──────────────────────────────────────────────────────────────────
    nll_terms: list[torch.Tensor] = []
    for k in range(1, K + 1):
        mask = types == k                           # (B,)
        if mask.sum() == 0:
            continue
        t_k = times[mask]                          # (n_k,)
        p_kt = joint[mask, k - 1, :]              # (n_k, T)
        idx = t_k.clamp(0, T - 1).long()
        p_at_t = p_kt.gather(1, idx.unsqueeze(1)).squeeze(1).clamp(min=eps)
        nll_terms.append(-p_at_t.log().mean())

    nll = torch.stack(nll_terms).sum() if nll_terms else joint.sum() * 0.0

    # ── Pairwise ranking ─────────────────────────────────────────────────────
    cif = torch.cumsum(joint, dim=2)              # (B, K, T)
    rank_terms: list[torch.Tensor] = []
    for k in range(1, K + 1):
        mask_k = types == k
        if mask_k.sum() < 2:
            continue
        t_i = times[mask_k].float()               # event subjects
        r_i = cif[mask_k, k - 1, :]              # CIF for event k (n_k, T)
        t_all = times.float()
        r_all = cif[:, k - 1, :]                  # (B, T)
        for m in range(len(t_i)):
            ti = t_i[m]
            comparable = t_all > ti               # j with t_j > t_i
            if comparable.sum() == 0:
                continue
            ti_idx = ti.long().clamp(0, T - 1)
            risk_i_at_ti = r_i[m, ti_idx]         # scalar
            risk_j_at_ti = r_all[comparable, ti_idx]  # (n_comp,)
            eta = torch.exp(-(risk_i_at_ti - risk_j_at_ti) / sigma)
            rank_terms.append(eta.mean())

    rank_loss = torch.stack(rank_terms).mean() if rank_terms else joint.sum() * 0.0

    return nll + alpha * rank_loss


def train_deephit(
    X_train: np.ndarray,
    times_train: np.ndarray,
    types_train: np.ndarray,
    X_val: np.ndarray,
    times_val: np.ndarray,
    types_val: np.ndarray,
    in_dim: int,
    n_epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 20,
    device_str: str = "cpu",
) -> DeepHitMLP:
    """Train a flat-feature DeepHit MLP with early stopping on val C-index."""
    device = torch.device(device_str)
    model = DeepHitMLP(in_dim=in_dim, hidden=64, num_events=NUM_EVENTS,
                       time_bins=N_TIME_BINS).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs, eta_min=lr * 0.01)

    # Convert times to bin indices
    def _to_bins(t: np.ndarray) -> np.ndarray:
        return np.searchsorted(TIME_BINS, t).clip(0, N_TIME_BINS - 1).astype(np.int64)

    t_train_bins = _to_bins(times_train)
    t_val_bins   = _to_bins(times_val)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    t_tr = torch.tensor(t_train_bins, dtype=torch.long, device=device)
    y_tr = torch.tensor(types_train, dtype=torch.long, device=device)

    X_vl = torch.tensor(X_val, dtype=torch.float32, device=device)

    n = len(X_train)
    best_cindex = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start: start + batch_size]
            x_b = X_tr[batch_idx]
            t_b = t_tr[batch_idx]
            y_b = y_tr[batch_idx]
            optim.zero_grad()
            joint = model(x_b)
            loss = _deephit_loss(joint, t_b, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        # Validate on val set — mean C-index across events with ≥2 observations
        model.eval()
        with torch.no_grad():
            cif_val = model.cif(X_vl).cpu().numpy()  # (n_val, K, T)
        t365_bin = int(np.searchsorted(TIME_BINS, 365.0))
        ci_values = []
        for k_ev in range(NUM_EVENTS):
            obs_k = (types_val == k_ev + 1).astype(bool)
            if obs_k.sum() >= 2:
                risks_k = cif_val[:, k_ev, t365_bin]
                ci_k = concordance_index(times_val.astype(np.float64), risks_k, obs_k)
                if not np.isnan(ci_k):
                    ci_values.append(ci_k)
        mean_ci = float(np.mean(ci_values)) if ci_values else 0.5
        if mean_ci > best_cindex:
            best_cindex = mean_ci
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("DeepHit early stop at epoch %d (best val C-index=%.4f)", epoch, best_cindex)
            break

        if epoch % 20 == 0:
            logger.info("DeepHit epoch %d/%d  loss=%.4f  val_c=%.4f", epoch, n_epochs,
                        total_loss / max(n_batches, 1), best_cindex)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info("DeepHit training complete (best val C-index=%.4f)", best_cindex)
    return model


def predict_deephit_cif(
    model: DeepHitMLP,
    X: np.ndarray,
    device_str: str = "cpu",
) -> np.ndarray:
    """Return CIF (n, K, T) for all time bins from DeepHit model."""
    device = torch.device(device_str)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        cif = model.cif(x_t).cpu().numpy()
    return cif  # (n, K, T)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    cif: np.ndarray,
    times_test: np.ndarray,
    types_test: np.ndarray,
    label: str,
) -> dict:
    """Evaluate CIF (n, K, T) and return a dict of MetricWithCI per event."""
    results: dict[str, dict] = {}
    for k, name in enumerate(EVENT_NAMES):
        observed = (types_test == k + 1).astype(bool)
        cif_k = cif[:, k, :]

        # Risk score at 365-day bin
        t365 = int(np.searchsorted(TIME_BINS, 365.0))
        risks = cif_k[:, t365].astype(np.float64)

        ci = bootstrap_ci(
            concordance_index,
            times_test.astype(np.float64),
            risks,
            observed,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_SEED,
        )

        # IBS over eval times
        def _ibs(cif_arr, times, obs):
            return integrated_brier_score(
                cif_arr, times, obs, EVAL_TIMES, TIME_BINS
            )

        ibs = bootstrap_ci(
            _ibs,
            cif_k.astype(np.float64),
            times_test.astype(np.float64),
            observed,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_SEED + k,
        )

        results[name] = {"cindex": ci, "ibs": ibs}
        logger.info(
            "  [%s] %s  C-index=%s  IBS=%s",
            label, name, ci, ibs,
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = PROJECT_ROOT / "data" / "raw" / "bone-marrow.arff"
    out_dir = PROJECT_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading %s", data_path)
    df = load_bmt(data_path)
    logger.info("Loaded %d patients, %d columns", len(df), len(df.columns))

    # ── 2. Event labels ───────────────────────────────────────────────────────
    etype_all, etime_all = make_event_labels(df)
    df["_etype"] = etype_all
    df["_etime"] = etime_all

    # ── 3. Feature matrix ──────────────────────────────────────────────────────
    X_all = build_feature_matrix(df)
    feature_names = list(X_all.columns)
    logger.info("Features (%d): %s", len(feature_names), feature_names)

    # ── 4. Stratified split ───────────────────────────────────────────────────
    df["_row_id"] = np.arange(len(df))
    train_df, val_df, test_df = make_splits(
        df, val_fraction=0.15, test_fraction=0.15, random_seed=RANDOM_SEED
    )
    logger.info(
        "Splits: train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # Re-align features by row ID
    train_ids = train_df["_row_id"].values
    val_ids   = val_df["_row_id"].values
    test_ids  = test_df["_row_id"].values

    X_np = X_all.values.astype(np.float64)

    X_tr_raw = X_np[train_ids]
    X_vl_raw = X_np[val_ids]
    X_te_raw = X_np[test_ids]

    etype_tr = etype_all[train_ids]
    etime_tr = etime_all[train_ids]
    etype_vl = etype_all[val_ids]
    etime_vl = etime_all[val_ids]
    etype_te = etype_all[test_ids]
    etime_te = etime_all[test_ids]

    # Impute any remaining NaNs with training column means
    col_means = np.nanmean(X_tr_raw, axis=0)
    for arr in [X_tr_raw, X_vl_raw, X_te_raw]:
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Standardise using training statistics
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_vl = scaler.transform(X_vl_raw)
    X_te = scaler.transform(X_te_raw)

    # Build DataFrames for lifelines (preserve column names)
    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    X_vl_df = pd.DataFrame(X_vl, columns=feature_names)
    X_te_df = pd.DataFrame(X_te, columns=feature_names)

    all_results: dict[str, dict] = {}

    # ── 5. Cox PH baseline ────────────────────────────────────────────────────
    logger.info("=== Cox PH (cause-specific) ===")
    cox = CoxPHBaseline(num_events=NUM_EVENTS, penalizer=0.1)
    cox.fit(X_tr_df, etime_tr, etype_tr)
    cif_cox = cox.predict_cif(X_te_df, TIME_BINS)   # (n_test, K, T)
    all_results["Cox PH"] = evaluate_model(cif_cox, etime_te, etype_te, "Cox PH")

    # ── 6. Fine-Gray baseline ────────────────────────────────────────────────
    logger.info("=== Fine-Gray ===")
    fg = FineGrayBaseline(num_events=NUM_EVENTS, penalizer=0.1)
    fg.fit(X_tr_df, etime_tr, etype_tr)
    cif_fg = fg.predict_cif(X_te_df, TIME_BINS)
    all_results["Fine-Gray"] = evaluate_model(cif_fg, etime_te, etype_te, "Fine-Gray")

    # ── 7. RSF (optional) ────────────────────────────────────────────────────
    try:
        from capa.model.baselines import RandomSurvivalForestBaseline
        logger.info("=== Random Survival Forest ===")
        rsf = RandomSurvivalForestBaseline(num_events=NUM_EVENTS, n_estimators=500,
                                           min_samples_leaf=5, random_state=RANDOM_SEED)
        rsf.fit(X_tr, etime_tr, etype_tr)
        cif_rsf = rsf.predict_cif(X_te, TIME_BINS)
        all_results["RSF"] = evaluate_model(cif_rsf, etime_te, etype_te, "RSF")
    except ImportError:
        logger.warning("scikit-survival not installed — skipping RSF baseline.")

    # ── 8. DeepHit (flat features) ───────────────────────────────────────────
    logger.info("=== DeepHit (flat clinical + HLA features) ===")
    in_dim = X_tr.shape[1]
    model_dh = train_deephit(
        X_tr, etime_tr, etype_tr,
        X_vl, etime_vl, etype_vl,
        in_dim=in_dim,
        n_epochs=200,
        lr=1e-3,
        batch_size=32,
        patience=25,
    )
    # Evaluate on train+val combined for final test (standard practice for deep models)
    X_trainval = np.vstack([X_tr, X_vl])
    etime_trainval = np.concatenate([etime_tr, etime_vl])
    etype_trainval = np.concatenate([etype_tr, etype_vl])

    model_dh_final = train_deephit(
        X_trainval, etime_trainval, etype_trainval,
        X_vl, etime_vl, etype_vl,  # val set just for early-stop criterion
        in_dim=in_dim,
        n_epochs=200,
        lr=1e-3,
        batch_size=32,
        patience=30,
    )
    cif_dh = predict_deephit_cif(model_dh_final, X_te)
    all_results["DeepHit (flat)"] = evaluate_model(cif_dh, etime_te, etype_te, "DeepHit (flat)")

    # ── 9. Print results table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS — Test set (n={}) — C-index and IBS (95 % CI, {} bootstrap)".format(
        len(X_te), N_BOOTSTRAP))
    print("=" * 80)
    header = f"{'Model':<28} {'GvHD C-index':>16} {'Rel C-index':>16} {'TRM C-index':>16}"
    print(header)
    print("-" * 80)
    for model_name, res in all_results.items():
        row = f"{model_name:<28}"
        for event in EVENT_NAMES:
            ci = res[event]["cindex"]
            row += f"  {ci.value:.3f} ({ci.ci_lower:.3f}–{ci.ci_upper:.3f})"
        print(row)

    print("\n")
    header2 = f"{'Model':<28} {'GvHD IBS':>20} {'Rel IBS':>20} {'TRM IBS':>20}"
    print(header2)
    print("-" * 84)
    for model_name, res in all_results.items():
        row = f"{model_name:<28}"
        for event in EVENT_NAMES:
            ibs = res[event]["ibs"]
            row += f"  {ibs.value:.3f} ({ibs.ci_lower:.3f}–{ibs.ci_upper:.3f})"
        print(row)

    # ── 10. Print LaTeX table snippet ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LATEX TABLE SNIPPET (main.tex Table 1 replacement)")
    print("=" * 80)
    latex_names = {
        "Cox PH": "Cox PH (cause-specific)",
        "Fine-Gray": "Fine--Gray",
        "RSF": "Random Survival Forest",
        "DeepHit (flat)": "DeepHit (tabular HLA)",
    }
    for model_name, res in all_results.items():
        lname = latex_names.get(model_name, model_name)
        parts = []
        for event in EVENT_NAMES:
            ci = res[event]["cindex"]
            parts.append(f"{ci.value:.3f} ({ci.ci_lower:.3f}--{ci.ci_upper:.3f})")
        print(f"    {lname:<32} & {' & '.join(parts)} \\\\")

    print("\n% IBS table:")
    for model_name, res in all_results.items():
        lname = latex_names.get(model_name, model_name)
        parts = []
        for event in EVENT_NAMES:
            ibs = res[event]["ibs"]
            parts.append(f"{ibs.value:.3f} ({ibs.ci_lower:.3f}--{ibs.ci_upper:.3f})")
        print(f"    {lname:<32} & {' & '.join(parts)} \\\\")

    # ── 11. Save JSON ─────────────────────────────────────────────────────────
    def _serialize(obj):
        if isinstance(obj, MetricWithCI):
            return obj.to_dict()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(type(obj))

    out_path = out_dir / "real_baselines.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_vl)),
                "n_test": int(len(X_te)),
                "time_bins": TIME_BINS.tolist(),
                "eval_times": EVAL_TIMES.tolist(),
                "event_names": EVENT_NAMES,
                "results": all_results,
            },
            fh,
            indent=2,
            default=_serialize,
        )
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
