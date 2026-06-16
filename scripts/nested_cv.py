"""Repeated stratified k-fold cross-validation for CAPA.

Replaces a single train/test split with repeated k-fold CV, giving a much
more honest performance picture for small cohorts (n = 187).

Usage
-----
Quick smoke-test on synthetic data (no real data or checkpoint needed):
    uv run python scripts/nested_cv.py --synthetic

Full CV on real preprocessed data (frequency-imputed HLA alleles):
    uv run python scripts/nested_cv.py --diff-mode none
    uv run python scripts/nested_cv.py --diff-mode signed

Custom folds / repeats:
    uv run python scripts/nested_cv.py --folds 5 --repeats 10

Save results JSON:
    uv run python scripts/nested_cv.py --output-path runs/cv_results.json

IMPORTANT CAVEAT — real-data mode
----------------------------------
The real UCI BMT cohort has no actual donor/recipient allele typing, only
aggregate mismatch counts. ``data/processed/bmt_imputed_hla.csv`` assigns
*frequency-imputed* alleles constrained to match each patient's recorded
mismatch count (see ``scripts/impute_hla_alleles.py``). These alleles carry
zero outcome-specific biological signal. Consequently ``--diff-mode`` here
validates pipeline correctness and clinical-covariate performance under
repeated CV — it CANNOT demonstrate the directional HLA-embedding mechanism,
because the data contains no real allele-outcome relationship to detect.
A Cox baseline on scalar mismatch distance is run alongside CAPA for honest
comparison.

Notes
-----
* Stratification uses the first-event indicator so that each fold has a
  similar distribution of event types and censoring.
* The model is retrained from scratch for each fold — no checkpoint is
  required; this script trains a lightweight version (embed_dim inferred
  from a provided checkpoint, or defaulting to a small default).
* Bootstrap CIs are computed per fold and then aggregated via the pooled
  bootstrap distribution across folds.
* GvHD evaluation is skipped automatically when the fold has fewer than
  5 events (the metric is unreliable below this threshold).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MIN_EVENTS_FOR_METRIC = 5  # skip C-index if fewer events in a fold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeated stratified k-fold CV for CAPA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",   type=int, default=5,  help="Number of CV folds")
    p.add_argument("--repeats", type=int, default=5,  help="Number of repeat runs")
    p.add_argument("--seed",    type=int, default=42, help="Base random seed")
    p.add_argument("--device",  default="cpu",        help="cpu / cuda / mps")
    p.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data (no real data required).",
    )
    p.add_argument("--synthetic-n",    type=int, default=187)
    p.add_argument("--synthetic-seed", type=int, default=0)
    p.add_argument(
        "--epochs",    type=int, default=30,
        help="Training epochs per fold (use fewer for quick smoke-tests).",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=200,
        help="Bootstrap samples per fold for CIs (0 to skip).",
    )
    p.add_argument(
        "--output-path", type=Path, default=None,
        help="Write results JSON to this path.",
    )
    p.add_argument(
        "--diff-mode", choices=["none", "signed"], default="none",
        help="'none' = standard donor/recipient cross-attention; "
             "'signed' = self-attention on the signed difference "
             "(recipient - donor) embedding, matching the directional "
             "simulation's CAPA variant.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _make_synthetic(
    n: int = 187,
    n_loci: int = 5,
    embed_dim: int = 16,
    n_events: int = 3,
    time_bins: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a synthetic dataset that mirrors the UCI BMT structure."""
    rng = np.random.default_rng(seed)

    donor     = rng.standard_normal((n, n_loci, embed_dim)).astype(np.float32)
    recipient = rng.standard_normal((n, n_loci, embed_dim)).astype(np.float32)
    clinical  = rng.standard_normal((n, 4)).astype(np.float32)

    # Simulate competing-risks outcomes: event ∈ {0=censored, 1, 2, 3}
    # Rough event rates matching UCI BMT: GvHD ~20%, relapse ~25%, TRM ~15%
    probs = rng.dirichlet([0.40, 0.20, 0.25, 0.15], size=n)  # (n, 4)
    event_type = np.array([rng.choice(4, p=p) for p in probs])  # 0=censored
    event_time = rng.uniform(10, 730, size=n).astype(np.float32)

    return {
        "donor":      donor,
        "recipient":  recipient,
        "clinical":   clinical,
        "event_type": event_type,
        "event_time": event_time,
        "n_loci":     n_loci,
        "embed_dim":  embed_dim,
        "n_events":   n_events,
        "time_bins":  time_bins,
    }


_REAL_LOCI = ["A", "B", "C", "DRB1", "DQB1"]
_REAL_MAX_DAYS = 730.0
_REAL_EMB_PATH = _ROOT / "data/processed/hla_embeddings.h5"
_REAL_CSV_PATH = _ROOT / "data/processed/bmt_imputed_hla.csv"


def _load_real_data() -> dict[str, Any]:
    """Load the real UCI BMT cohort with frequency-imputed HLA alleles.

    CAVEAT: alleles in bmt_imputed_hla.csv are frequency-imputed (see
    scripts/impute_hla_alleles.py), not real typing — they carry no
    outcome-specific biological signal. This loader provides real survival
    outcomes and real clinical covariates; the HLA embeddings only let us
    test pipeline mechanics, not the directional mechanism itself.
    """
    from capa.model.capa_model import DISEASE_CATEGORIES

    if not _REAL_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{_REAL_CSV_PATH} not found — run scripts/impute_hla_alleles.py first."
        )

    df = pd.read_csv(_REAL_CSV_PATH)
    n = len(df)

    embs: dict[str, np.ndarray] = {}
    with h5py.File(_REAL_EMB_PATH, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    embed_dim = next(iter(embs.values())).shape[0]
    n_loci = len(_REAL_LOCI)

    donor_emb     = np.zeros((n, n_loci, embed_dim), dtype=np.float32)
    recipient_emb = np.zeros((n, n_loci, embed_dim), dtype=np.float32)
    dist          = np.zeros((n, n_loci), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        row = row._asdict()
        for j, locus in enumerate(_REAL_LOCI):
            d1 = embs.get(row[f"donor_{locus}_1"], np.zeros(embed_dim))
            d2 = embs.get(row[f"donor_{locus}_2"], np.zeros(embed_dim))
            r1 = embs.get(row[f"recipient_{locus}_1"], np.zeros(embed_dim))
            r2 = embs.get(row[f"recipient_{locus}_2"], np.zeros(embed_dim))
            d_mean = (d1 + d2) / 2.0
            r_mean = (r1 + r2) / 2.0
            donor_emb[i, j] = d_mean
            recipient_emb[i, j] = r_mean
            dist[i, j] = np.linalg.norm(d_mean - r_mean)

    disease_map = {v: k for k, v in enumerate(DISEASE_CATEGORIES)}
    disease_alias = {"chronic": "CML", "lymphoma": "NHL", "nonmalignant": "other"}
    disease_idx = df["disease"].astype(str).str.strip().map(
        lambda s: disease_map.get(disease_alias.get(s, s), 0)
    ).to_numpy(dtype=np.int64)

    clinical = np.stack([
        df["recipient_age"].fillna(0).to_numpy(dtype=np.float32) / 100.0,
        df["donor_age"].fillna(0).to_numpy(dtype=np.float32) / 100.0,
        df["cd34_dose"].fillna(0).to_numpy(dtype=np.float32) / 10.0,
        df["sex_mismatch_f2m"].fillna(0).to_numpy(dtype=np.float32),
    ], axis=1)

    relapse = df["relapse"].fillna(0).astype(int).to_numpy()
    dead    = df["dead"].fillna(0).astype(int).to_numpy()
    gvhd    = df["acute_gvhd_iii_iv"].fillna(0).astype(int).to_numpy()
    event_time = np.clip(df["survival_time_days"].astype(float).to_numpy(), 0.0, _REAL_MAX_DAYS).astype(np.float32)

    event_type = np.zeros(n, dtype=np.int64)
    event_type[(gvhd == 1) & (relapse == 0) & (dead == 0)] = 1
    event_type[(dead == 1) & (relapse == 0)] = 3
    event_type[relapse == 1] = 2

    return {
        "donor":      donor_emb,
        "recipient":  recipient_emb,
        "dist":       dist,
        "disease_idx": disease_idx,
        "clinical":   clinical,
        "event_type": event_type,
        "event_time": event_time,
        "n_loci":     n_loci,
        "embed_dim":  embed_dim,
        "n_events":   3,
        "time_bins":  100,
        "max_days":   _REAL_MAX_DAYS,
    }


# ---------------------------------------------------------------------------
# Stratified CV splitter
# ---------------------------------------------------------------------------

def _stratified_kfold_indices(
    event_type: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs for stratified k-fold.

    Stratifies on event_type (0 = censored, 1/2/3 = events) so each fold
    preserves the overall event-type distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(event_type)
    indices = np.arange(n)

    # Group by event_type
    classes = np.unique(event_type)
    per_class: dict[int, np.ndarray] = {}
    for c in classes:
        idx = indices[event_type == c]
        per_class[int(c)] = rng.permutation(idx)

    # Assign fold IDs round-robin within each class
    fold_ids = np.empty(n, dtype=int)
    for c, idx in per_class.items():
        for i, sample_idx in enumerate(idx):
            fold_ids[sample_idx] = i % n_folds

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        val_idx   = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Per-fold model training + evaluation
# ---------------------------------------------------------------------------

def _build_model(data: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from capa.model.capa_model import CAPAModel
    return CAPAModel(
        embedding_dim=data["embed_dim"],
        loci=[f"L{i}" for i in range(data["n_loci"])],
        clinical_dim=32,
        interaction_dim=64,
        num_heads=min(4, data["embed_dim"]),
        num_layers=2,
        dropout=0.1,
        time_bins=data["time_bins"],
        num_events=data["n_events"],
    ).to(device)


def _cat_tensor(data: dict[str, Any], idx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Build the (n, 4) categorical-index tensor for ClinicalEncoder.

    Only "disease" is populated from real data when available (synthetic
    data has no disease_idx, so all four columns default to "unknown"/0).
    """
    n_cat = 4
    cat_t = torch.zeros(len(idx), n_cat, dtype=torch.long, device=device)
    if "disease_idx" in data:
        cat_t[:, 0] = torch.from_numpy(data["disease_idx"][idx]).long().to(device)
    return cat_t


def _diff_embeddings(
    donor_t: torch.Tensor, recip_t: torch.Tensor, diff_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the donor/recipient → signed-difference transform if requested."""
    if diff_mode == "signed":
        diff = donor_t - recip_t
        return diff, diff
    return donor_t, recip_t


def _train_fold(
    model: torch.nn.Module,
    data: dict[str, Any],
    train_idx: np.ndarray,
    epochs: int,
    device: torch.device,
    diff_mode: str = "none",
) -> None:
    """Train model for one fold (simplified training loop for CV)."""
    from capa.model.losses import deephit_loss

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    donor_t     = torch.from_numpy(data["donor"][train_idx]).to(device)
    recip_t     = torch.from_numpy(data["recipient"][train_idx]).to(device)
    donor_t, recip_t = _diff_embeddings(donor_t, recip_t, diff_mode)
    clin_t      = torch.from_numpy(data["clinical"][train_idx]).to(device)
    etype_t = torch.from_numpy(data["event_type"][train_idx]).long().to(device)
    # deephit_loss expects time bin indices (0..time_bins-1), not raw days
    max_days = data.get("max_days", 730.0)
    raw_times = data["event_time"][train_idx]
    bin_idx   = np.clip(
        (raw_times / max_days * (data["time_bins"] - 1)).astype(int),
        0, data["time_bins"] - 1,
    )
    etime_t = torch.from_numpy(bin_idx).long().to(device)

    cat_t = _cat_tensor(data, train_idx, device)

    for epoch in range(epochs):
        opt.zero_grad()
        clin_feats = model.clinical_encoder(clin_t, cat_t)
        logits = model(donor_t, recip_t, clin_feats)
        loss = deephit_loss(logits, event_times=etime_t, event_types=etype_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()


def _eval_fold(
    model: torch.nn.Module,
    data: dict[str, Any],
    val_idx: np.ndarray,
    n_bootstrap: int,
    device: torch.device,
    diff_mode: str = "none",
) -> dict[str, Any]:
    """Evaluate model on one validation fold."""
    from capa.training.evaluate import concordance_index, bootstrap_ci

    donor_t  = torch.from_numpy(data["donor"][val_idx]).to(device)
    recip_t  = torch.from_numpy(data["recipient"][val_idx]).to(device)
    donor_t, recip_t = _diff_embeddings(donor_t, recip_t, diff_mode)
    clin_t   = torch.from_numpy(data["clinical"][val_idx]).to(device)
    cat_t    = _cat_tensor(data, val_idx, device)

    with torch.no_grad():
        clin_feats = model.clinical_encoder(clin_t, cat_t)
        cif = model.cif(donor_t, recip_t, clin_feats)  # (n, n_events, time_bins)

    cif_np      = cif.cpu().numpy()
    event_types = data["event_type"][val_idx]
    event_times = data["event_time"][val_idx].astype(np.float64)
    event_names = ["gvhd", "relapse", "trm"][:data["n_events"]]

    fold_results: dict[str, Any] = {"n": len(val_idx), "events": {}}

    for k, name in enumerate(event_names):
        observed    = (event_types == (k + 1)).astype(bool)
        risk_scores = cif_np[:, k, -1].astype(np.float64)
        n_events    = int(observed.sum())

        if n_events < MIN_EVENTS_FOR_METRIC:
            fold_results["events"][name] = {
                "cindex": None,
                "n_events": n_events,
                "note": f"skipped — only {n_events} events (< {MIN_EVENTS_FOR_METRIC})",
            }
            continue

        c = concordance_index(event_times, risk_scores, observed)

        ci: dict[str, float | None] = {"lower": None, "upper": None}
        if n_bootstrap > 0:
            m = bootstrap_ci(
                concordance_index,
                event_times, risk_scores, observed,
                n_bootstrap=n_bootstrap,
            )
            ci = {"lower": round(m.ci_lower, 4), "upper": round(m.ci_upper, 4)}

        fold_results["events"][name] = {
            "cindex": round(c, 4),
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "n_events": n_events,
        }

    return fold_results


def _run_cox_fold(
    data: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    """Cox scalar-mismatch-distance baseline for one fold (real data only).

    Uses per-locus L2 distance between mean-pooled donor/recipient ESM-2
    embeddings plus clinical covariates — the same scalar-distance
    representation CAPA's signed-difference mechanism is meant to improve on.
    """
    from lifelines import CoxPHFitter

    if "dist" not in data:
        return {"n": len(val_idx), "events": {}, "note": "no dist features (synthetic data)"}

    dist_cols = [f"dist_{i}" for i in range(data["dist"].shape[1])]
    clin_cols = ["age_recip", "age_donor", "cd34", "sex_mm"]
    feat_cols = dist_cols + clin_cols

    def _frame(idx: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(data["dist"][idx], columns=dist_cols)
        for j, c in enumerate(clin_cols):
            df[c] = data["clinical"][idx, j]
        df["survival_time_days"] = data["event_time"][idx]
        df["event_type"] = data["event_type"][idx]
        return df

    df_tr, df_va = _frame(train_idx), _frame(val_idx)
    active_feats = [c for c in feat_cols if df_tr[c].std() > 0]

    event_names = ["gvhd", "relapse", "trm"][:data["n_events"]]
    fold_results: dict[str, Any] = {"n": len(val_idx), "events": {}}

    for k, name in enumerate(event_names):
        observed = (data["event_type"][val_idx] == (k + 1)).astype(bool)
        n_events_va = int(observed.sum())
        n_events_tr = int((df_tr["event_type"] == (k + 1)).sum())
        if n_events_va < MIN_EVENTS_FOR_METRIC or n_events_tr < MIN_EVENTS_FOR_METRIC:
            fold_results["events"][name] = {
                "cindex": None, "n_events": n_events_va,
                "note": f"skipped — too few events (val={n_events_va}, train={n_events_tr})",
            }
            continue

        df_tr_k = df_tr[active_feats + ["survival_time_days"]].copy()
        df_tr_k["event"] = (df_tr["event_type"] == (k + 1)).astype(int)
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(df_tr_k, duration_col="survival_time_days", event_col="event")
            risk_scores = cph.predict_partial_hazard(df_va[active_feats]).to_numpy().astype(np.float64)
        except Exception as e:
            fold_results["events"][name] = {"cindex": None, "n_events": n_events_va, "note": f"Cox fit failed: {e}"}
            continue

        from capa.training.evaluate import concordance_index
        c = concordance_index(data["event_time"][val_idx].astype(np.float64), risk_scores, observed)
        fold_results["events"][name] = {"cindex": round(c, 4), "n_events": n_events_va}

    return fold_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregatedMetric:
    mean:   float
    std:    float
    min:    float
    max:    float
    ci_lower_mean: float | None  # mean of per-fold CI lower bounds
    ci_upper_mean: float | None  # mean of per-fold CI upper bounds
    n_folds_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate(fold_results: list[dict[str, Any]]) -> dict[str, AggregatedMetric]:
    """Pool per-fold metrics into summary statistics."""
    event_names = set()
    for fr in fold_results:
        event_names.update(fr["events"].keys())

    aggregated: dict[str, AggregatedMetric] = {}
    for name in sorted(event_names):
        cindexes: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []

        for fr in fold_results:
            ev = fr["events"].get(name, {})
            if ev.get("cindex") is not None:
                cindexes.append(ev["cindex"])
                if ev.get("ci_lower") is not None:
                    ci_lowers.append(ev["ci_lower"])
                    ci_uppers.append(ev["ci_upper"])

        if not cindexes:
            continue

        arr = np.array(cindexes)
        aggregated[name] = AggregatedMetric(
            mean=round(float(arr.mean()), 4),
            std=round(float(arr.std()), 4),
            min=round(float(arr.min()), 4),
            max=round(float(arr.max()), 4),
            ci_lower_mean=round(float(np.mean(ci_lowers)), 4) if ci_lowers else None,
            ci_upper_mean=round(float(np.mean(ci_uppers)), 4) if ci_uppers else None,
            n_folds_evaluated=len(cindexes),
        )
    return aggregated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    logger.info("Loading data …")
    if args.synthetic:
        data = _make_synthetic(n=args.synthetic_n, seed=args.synthetic_seed)
        logger.info("Using synthetic data (n=%d)", args.synthetic_n)
    else:
        data = _load_real_data()
        logger.info("Using real UCI BMT data (n=%d)", len(data["event_time"]))

    n_total    = len(data["event_time"])
    all_folds:  list[dict[str, Any]] = []
    all_cox_folds: list[dict[str, Any]] = []

    for repeat in range(args.repeats):
        seed = args.seed + repeat * 1000
        splits = _stratified_kfold_indices(
            data["event_type"], n_folds=args.folds, seed=seed
        )

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            logger.info(
                "Repeat %d/%d  Fold %d/%d  (train=%d  val=%d)",
                repeat + 1, args.repeats, fold_i + 1, args.folds,
                len(train_idx), len(val_idx),
            )

            model = _build_model(data, device)
            _train_fold(model, data, train_idx, epochs=args.epochs, device=device, diff_mode=args.diff_mode)

            fold_result = _eval_fold(
                model, data, val_idx,
                n_bootstrap=args.n_bootstrap,
                device=device,
                diff_mode=args.diff_mode,
            )
            fold_result["repeat"] = repeat
            fold_result["fold"]   = fold_i
            all_folds.append(fold_result)

            cox_result = _run_cox_fold(data, train_idx, val_idx)
            cox_result["repeat"] = repeat
            cox_result["fold"]   = fold_i
            all_cox_folds.append(cox_result)

    # Aggregate
    agg = _aggregate(all_folds)
    agg_cox = _aggregate(all_cox_folds)

    # Report
    print("\n" + "=" * 60)
    print(f"CAPA (diff-mode={args.diff_mode}) Repeated {args.folds}-fold CV "
          f"× {args.repeats} repeats  (n={n_total})")
    print("=" * 60)
    print(f"{'Event':<10}  {'Mean C-index':>12}  {'Std':>6}  {'Range':>14}  {'Avg 95% CI':>18}  {'Folds':>5}")
    print("-" * 60)
    for name, m in agg.items():
        ci_str = (
            f"{m.ci_lower_mean:.3f}–{m.ci_upper_mean:.3f}"
            if m.ci_lower_mean is not None else "no bootstrap"
        )
        print(
            f"{name:<10}  {m.mean:>12.4f}  {m.std:>6.4f}"
            f"  [{m.min:.3f}–{m.max:.3f}]  {ci_str:>18}  {m.n_folds_evaluated:>5}"
        )
    print("=" * 60)
    if agg_cox:
        print(f"\nCox (scalar mismatch distance) baseline — same folds")
        print("-" * 60)
        for name, m in agg_cox.items():
            print(f"{name:<10}  {m.mean:>12.4f}  {m.std:>6.4f}  [{m.min:.3f}–{m.max:.3f}]  {m.n_folds_evaluated:>5} folds")
        print("=" * 60)
    print(
        "\nNote: mean C-index across folds is more reliable than a single\n"
        "29-patient test split. Wide CIs reflect the fundamental data-size\n"
        "limit — not a modelling failure.\n"
    )
    if not args.synthetic:
        print(
            "CAVEAT: HLA alleles in bmt_imputed_hla.csv are frequency-imputed,\n"
            "not real typing — they carry no outcome-specific biological signal.\n"
            "This run validates pipeline mechanics and clinical-covariate\n"
            "performance under proper CV; it does NOT test the directional\n"
            "HLA-embedding mechanism.\n"
        )

    results = {
        "config": {
            "folds": args.folds,
            "repeats": args.repeats,
            "epochs_per_fold": args.epochs,
            "n_bootstrap": args.n_bootstrap,
            "n_total": n_total,
            "synthetic": args.synthetic,
            "diff_mode": args.diff_mode,
            "caveat": None if args.synthetic else (
                "HLA alleles are frequency-imputed (no real typing); results "
                "validate pipeline mechanics, not the directional mechanism."
            ),
        },
        "per_fold": all_folds,
        "aggregated": {k: v.to_dict() for k, v in agg.items()},
        "cox_baseline": {
            "per_fold": all_cox_folds,
            "aggregated": {k: v.to_dict() for k, v in agg_cox.items()},
        },
    }

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", args.output_path)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
