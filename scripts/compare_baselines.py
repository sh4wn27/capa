"""Train all CAPA baseline models and generate a comparison table.

Usage
-----
Quick smoke-test with synthetic data (no real data required):
    uv run python scripts/compare_baselines.py --synthetic

Specify which models to include:
    uv run python scripts/compare_baselines.py --synthetic --models finegray cox capa_onehot

Skip bootstrap CIs (much faster):
    uv run python scripts/compare_baselines.py --synthetic --n-bootstrap 0

Save output to a named run:
    uv run python scripts/compare_baselines.py --synthetic --run-name baseline_smoke

RSF requires scikit-survival:
    uv add scikit-survival
    uv run python scripts/compare_baselines.py --synthetic --models rsf
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Resolve project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVENT_NAMES = ["GvHD", "Relapse", "TRM"]
_HLA_LOCI = ["A", "B", "C", "DRB1", "DQB1"]
_ALL_MODELS = ["finegray", "cox", "rsf", "capa_onehot", "capa"]

# Synthetic allele pool: 10 alleles per locus
_ALLELE_POOL: dict[str, list[str]] = {
    locus: [f"{locus}*{i:02d}:01" for i in range(1, 11)]
    for locus in _HLA_LOCI
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and compare all CAPA baseline models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data (no real BMT dataset needed).")
    p.add_argument("--n", type=int, default=200,
                   help="Number of synthetic subjects.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-bins", type=int, default=50)
    p.add_argument("--n-events", type=int, default=3)

    p.add_argument("--models", nargs="+", default=_ALL_MODELS,
                   choices=_ALL_MODELS + ["all"],
                   help="Which models to include.")
    p.add_argument("--epochs", type=int, default=20,
                   help="Max epochs for deep models.")
    p.add_argument("--embedding-dim", type=int, default=32,
                   help="Embedding dim for CAPA-OneHot and CAPA (synthetic).")
    p.add_argument("--interaction-dim", type=int, default=32)
    p.add_argument("--device", default="cpu")

    p.add_argument("--n-bootstrap", type=int, default=100,
                   help="Bootstrap replicates for CIs (0 = skip).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _rand_allele(locus: str, rng: np.random.Generator) -> str:
    pool = _ALLELE_POOL[locus]
    return pool[int(rng.integers(0, len(pool)))]


def make_synthetic_dataset(
    n: int,
    time_bins: int,
    num_events: int,
    seed: int,
) -> dict[str, Any]:
    """Generate a synthetic dataset for all baseline model types.

    Returns a dict with keys
    -------------------------
    tabular_X : pd.DataFrame (n, n_features)
        HLA mismatch counts + clinical features for Cox/Fine-Gray/RSF.
    times : np.ndarray (n,) int
        Event/censoring time bin indices.
    event_types : np.ndarray (n,) int
        0=censored, 1..K=event type.
    donor_alleles : list[list[str]]  (n, n_loci)
        Donor HLA allele strings for deep models.
    recipient_alleles : list[list[str]]  (n, n_loci)
        Recipient HLA allele strings.
    clinical_cont : np.ndarray (n, 4) float
        Normalised continuous+binary clinical features (age_r/100, age_d/100,
        cd34/10, sex_mismatch).
    """
    rng = np.random.default_rng(seed)

    # HLA alleles
    donor_alleles = [
        [_rand_allele(loc, rng) for loc in _HLA_LOCI]
        for _ in range(n)
    ]
    recipient_alleles = [
        [_rand_allele(loc, rng) for loc in _HLA_LOCI]
        for _ in range(n)
    ]

    # Tabular mismatch features
    mismatch_cols = {}
    for j, locus in enumerate(_HLA_LOCI):
        mm = np.array(
            [int(donor_alleles[i][j] != recipient_alleles[i][j]) for i in range(n)],
            dtype=np.float64,
        )
        mismatch_cols[f"mm_{locus}"] = mm

    age_r = rng.uniform(0, 1, n)          # normalised age_recipient/100
    age_d = rng.uniform(0, 1, n)
    cd34  = rng.uniform(0, 1, n)
    sexmm = rng.integers(0, 2, n).astype(np.float64)
    disease    = rng.integers(0, 5, n).astype(np.float64)
    cond       = rng.integers(0, 3, n).astype(np.float64)
    donor_type = rng.integers(0, 4, n).astype(np.float64)
    sc_source  = rng.integers(0, 3, n).astype(np.float64)

    tabular_X = pd.DataFrame({
        **mismatch_cols,
        "age_recipient": age_r,
        "age_donor": age_d,
        "cd34_dose": cd34,
        "sex_mismatch": sexmm,
        "disease": disease,
        "conditioning": cond,
        "donor_type": donor_type,
        "stem_cell_source": sc_source,
    })

    # Clinical continuous features for deep models (4 features)
    clinical_cont = np.stack([age_r, age_d, cd34, sexmm], axis=1).astype(np.float32)

    # Survival outcomes
    times      = rng.integers(1, time_bins, n).astype(np.int64)
    event_types = rng.integers(0, num_events + 1, n).astype(np.int64)

    return {
        "tabular_X": tabular_X,
        "times": times,
        "event_types": event_types,
        "donor_alleles": donor_alleles,
        "recipient_alleles": recipient_alleles,
        "clinical_cont": clinical_cont,
    }


def split_dataset(data: dict[str, Any], seed: int) -> tuple[dict, dict, dict]:
    """80/10/10 train/val/test split."""
    n = len(data["times"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train: n_train + n_val],
        "test":  idx[n_train + n_val:],
    }

    def _slice(d: dict, i: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, pd.DataFrame):
                out[k] = v.iloc[i].reset_index(drop=True)
            elif isinstance(v, np.ndarray):
                out[k] = v[i]
            elif isinstance(v, list):
                out[k] = [v[j] for j in i]
            else:
                out[k] = v
        return out

    return _slice(data, splits["train"]), _slice(data, splits["val"]), _slice(data, splits["test"])


# ---------------------------------------------------------------------------
# DataLoaders for deep models
# ---------------------------------------------------------------------------

class _OneHotDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for CAPAOneHot: allele indices + clinical + survival."""

    def __init__(
        self,
        donor_idx: np.ndarray,   # (n, n_loci) int
        recip_idx: np.ndarray,
        clinical: np.ndarray,    # (n, C) float
        times: np.ndarray,
        event_types: np.ndarray,
    ) -> None:
        self.donor_idx   = torch.tensor(donor_idx,   dtype=torch.long)
        self.recip_idx   = torch.tensor(recip_idx,   dtype=torch.long)
        self.clinical    = torch.tensor(clinical,    dtype=torch.float32)
        self.times       = torch.tensor(times,       dtype=torch.long)
        self.event_types = torch.tensor(event_types, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "donor_allele_indices":     self.donor_idx[idx],
            "recipient_allele_indices": self.recip_idx[idx],
            "clinical_features":        self.clinical[idx],
            "event_times":              self.times[idx],
            "event_types":              self.event_types[idx],
        }


class _CAPADataset(Dataset):  # type: ignore[type-arg]
    """Dataset for full CAPA model: float embeddings + clinical + survival."""

    def __init__(
        self,
        donor_emb: np.ndarray,   # (n, n_loci, D) float
        recip_emb: np.ndarray,
        clinical: np.ndarray,    # (n, C) float
        times: np.ndarray,
        event_types: np.ndarray,
    ) -> None:
        self.donor_emb   = torch.tensor(donor_emb,   dtype=torch.float32)
        self.recip_emb   = torch.tensor(recip_emb,   dtype=torch.float32)
        self.clinical    = torch.tensor(clinical,    dtype=torch.float32)
        self.times       = torch.tensor(times,       dtype=torch.long)
        self.event_types = torch.tensor(event_types, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "donor_embeddings":     self.donor_emb[idx],
            "recipient_embeddings": self.recip_emb[idx],
            "clinical_features":    self.clinical[idx],
            "event_times":          self.times[idx],
            "event_types":          self.event_types[idx],
        }


def make_onehot_loaders(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    vocab: "AlleleVocabulary",  # type: ignore[name-defined]
    batch_size: int = 32,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build DataLoaders for CAPAOneHotBaseline."""
    from capa.model.baselines import AlleleVocabulary

    def _encode_alleles(allele_lists: list[list[str]]) -> np.ndarray:
        n = len(allele_lists)
        n_loci = len(allele_lists[0])
        arr = np.zeros((n, n_loci), dtype=np.int64)
        for i, row in enumerate(allele_lists):
            for j, a in enumerate(row):
                arr[i, j] = vocab.encode(a)
        return arr

    def _make_loader(d: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
        di = _encode_alleles(d["donor_alleles"])
        ri = _encode_alleles(d["recipient_alleles"])
        ds = _OneHotDataset(di, ri, d["clinical_cont"], d["times"], d["event_types"])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _make_loader(train_data, True), _make_loader(val_data, False)


def make_capa_loaders(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    embedding_dim: int,
    seed: int,
    batch_size: int = 32,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build DataLoaders for the full CAPA model (random synthetic embeddings)."""
    rng = np.random.default_rng(seed)

    def _make_loader(d: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
        n = len(d["times"])
        n_loci = len(_HLA_LOCI)
        donor_emb = rng.standard_normal((n, n_loci, embedding_dim)).astype(np.float32)
        recip_emb = rng.standard_normal((n, n_loci, embedding_dim)).astype(np.float32)
        ds = _CAPADataset(
            donor_emb, recip_emb,
            d["clinical_cont"], d["times"], d["event_types"],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _make_loader(train_data, True), _make_loader(val_data, False)


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def _print_comparison_table(
    results: dict[str, dict[str, Any]],
    event_names: list[str],
) -> None:
    """Print a rich comparison table to stdout."""
    sep = "─" * 80
    print(f"\n{'='*80}")
    print("  CAPA Baseline Comparison")
    print(f"{'='*80}")

    # Header
    col_w = 12
    header_events = "  ".join(f"{'C-idx ' + e:>{col_w}}" for e in event_names)
    ibs_events    = "  ".join(f"{'IBS ' + e:>{col_w}}" for e in event_names)
    print(f"\n{'Model':<30}  {header_events}")
    print(f"{'':30}  {ibs_events}")
    print(sep)

    for model_name, res in results.items():
        if "error" in res:
            print(f"{model_name:<30}  ERROR: {res['error']}")
            continue
        cindex_vals = "  ".join(
            f"{res['cindex'].get(e, float('nan')):>{col_w}.4f}"
            for e in event_names
        )
        ibs_vals = "  ".join(
            f"{res['ibs'].get(e, float('nan')):>{col_w}.4f}"
            for e in event_names
        )
        train_s = res.get("train_time_s", float("nan"))
        print(f"{model_name:<30}  {cindex_vals}   ({train_s:.1f}s)")
        print(f"{'':30}  {ibs_vals}")
        print()

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = _ALL_MODELS

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run dir
    import datetime
    from capa.config import get_config
    cfg = get_config()
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_runs_dir = args.runs_dir if args.runs_dir is not None else cfg.training.runs_dir
    runs_dir = base_runs_dir / f"compare_{run_name}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", runs_dir)

    time_bins = args.time_bins
    num_events = args.n_events

    # ── Data ─────────────────────────────────────────────────────────
    if not args.synthetic:
        logger.error("Real-data pipeline not yet implemented. Use --synthetic.")
        sys.exit(1)

    logger.info("Generating synthetic dataset (n=%d, time_bins=%d)", args.n, time_bins)
    data = make_synthetic_dataset(args.n, time_bins, num_events, args.seed)
    train_data, val_data, test_data = split_dataset(data, args.seed)
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_data["times"]), len(val_data["times"]), len(test_data["times"]),
    )

    time_bins_arr = np.arange(time_bins, dtype=np.float64)

    # ── Evaluate helper ───────────────────────────────────────────────
    from capa.training.evaluate import evaluate_all

    def _eval(cif: np.ndarray) -> dict[str, Any]:
        result = evaluate_all(
            cif=cif,
            event_times=test_data["times"].astype(np.float64),
            event_types=test_data["event_types"].astype(np.int64),
            event_names=_EVENT_NAMES[:num_events],
            time_bins=time_bins_arr,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        cindex_map = {em.event_name: em.cindex.value for em in result.events}
        ibs_map    = {em.event_name: em.ibs.value if em.ibs else float("nan")
                      for em in result.events}
        return {"cindex": cindex_map, "ibs": ibs_map, "full": result.to_dict()}

    comparison: dict[str, Any] = {}

    # ── Fine-Gray ─────────────────────────────────────────────────────
    if "finegray" in models_to_run:
        logger.info("Fitting Fine-Gray…")
        from capa.model.baselines import FineGrayBaseline
        fg = FineGrayBaseline(num_events=num_events)
        t0 = time.perf_counter()
        try:
            fg.fit(train_data["tabular_X"], train_data["times"].astype(float),
                   train_data["event_types"])
            test_X_fg = test_data["tabular_X"]
            cif_fg = fg.predict_cif(test_X_fg, time_bins_arr)
            elapsed = time.perf_counter() - t0
            row = _eval(cif_fg)
            row["train_time_s"] = elapsed
            comparison[fg.name] = row
            logger.info("Fine-Gray done (%.1fs)", elapsed)
        except Exception as exc:
            logger.error("Fine-Gray failed: %s", exc)
            comparison["Fine-Gray"] = {"error": str(exc)}

    # ── Cox PH ────────────────────────────────────────────────────────
    if "cox" in models_to_run:
        logger.info("Fitting Cox PH…")
        from capa.model.baselines import CoxPHBaseline
        cox = CoxPHBaseline(num_events=num_events)
        t0 = time.perf_counter()
        try:
            cox.fit(train_data["tabular_X"], train_data["times"].astype(float),
                    train_data["event_types"])
            cif_cox = cox.predict_cif(test_data["tabular_X"], time_bins_arr)
            elapsed = time.perf_counter() - t0
            row = _eval(cif_cox)
            row["train_time_s"] = elapsed
            comparison[cox.name] = row
            logger.info("Cox PH done (%.1fs)", elapsed)
        except Exception as exc:
            logger.error("Cox PH failed: %s", exc)
            comparison["Cox PH (cause-specific)"] = {"error": str(exc)}

    # ── RSF ───────────────────────────────────────────────────────────
    if "rsf" in models_to_run:
        logger.info("Fitting Random Survival Forest…")
        from capa.model.baselines import RandomSurvivalForestBaseline
        rsf = RandomSurvivalForestBaseline(num_events=num_events, n_estimators=50)
        t0 = time.perf_counter()
        try:
            X_np = train_data["tabular_X"].to_numpy(dtype=np.float64)
            rsf.fit(X_np, train_data["times"].astype(float), train_data["event_types"])
            cif_rsf = rsf.predict_cif(
                test_data["tabular_X"].to_numpy(dtype=np.float64), time_bins_arr
            )
            elapsed = time.perf_counter() - t0
            row = _eval(cif_rsf)
            row["train_time_s"] = elapsed
            comparison[rsf.name] = row
            logger.info("RSF done (%.1fs)", elapsed)
        except ImportError as exc:
            logger.warning("RSF skipped: %s", exc)
            comparison["Random Survival Forest"] = {"error": str(exc)}
        except Exception as exc:
            logger.error("RSF failed: %s", exc)
            comparison["Random Survival Forest"] = {"error": str(exc)}

    # ── CAPA-OneHot ───────────────────────────────────────────────────
    if "capa_onehot" in models_to_run:
        logger.info("Fitting CAPA-OneHot (ablation)…")
        from capa.model.baselines import AlleleVocabulary, CAPAOneHotBaseline

        # Build vocabulary from training alleles
        vocab = AlleleVocabulary()
        all_train_alleles = [
            a
            for row in train_data["donor_alleles"] + train_data["recipient_alleles"]
            for a in row
        ]
        vocab.fit(all_train_alleles)
        logger.info("Vocabulary size: %d alleles", vocab.size)

        train_ldr, val_ldr = make_onehot_loaders(
            train_data, val_data, vocab, batch_size=32
        )

        oh = CAPAOneHotBaseline(
            num_events=num_events,
            time_bins=time_bins,
            embedding_dim=args.embedding_dim,
            n_loci=len(_HLA_LOCI),
            raw_clinical_dim=4,     # age_r, age_d, cd34, sex_mismatch
            clinical_dim=16,
            interaction_dim=args.interaction_dim,
            max_epochs=args.epochs,
            patience=max(5, args.epochs // 4),
            device=args.device,
        )
        t0 = time.perf_counter()
        try:
            oh.fit(train_ldr, val_ldr, vocab)

            # Encode test alleles
            def _enc(allele_lists: list[list[str]]) -> np.ndarray:
                n = len(allele_lists)
                arr = np.zeros((n, len(_HLA_LOCI)), dtype=np.int64)
                for i, row in enumerate(allele_lists):
                    for j, a in enumerate(row):
                        arr[i, j] = vocab.encode(a)
                return arr

            d_idx = _enc(test_data["donor_alleles"])
            r_idx = _enc(test_data["recipient_alleles"])
            clin  = test_data["clinical_cont"]
            cif_oh = oh.predict_cif(d_idx, r_idx, clin, time_bins_arr)
            elapsed = time.perf_counter() - t0
            row = _eval(cif_oh)
            row["train_time_s"] = elapsed
            comparison[oh.name] = row
            logger.info("CAPA-OneHot done (%.1fs)", elapsed)
        except Exception as exc:
            import traceback
            logger.error("CAPA-OneHot failed: %s\n%s", exc, traceback.format_exc())
            comparison["CAPA-OneHot (ablation)"] = {"error": str(exc)}

    # ── Full CAPA (synthetic embeddings) ─────────────────────────────
    if "capa" in models_to_run:
        logger.info("Fitting CAPA (synthetic embeddings)…")
        from capa.model.capa_model import CAPAModel
        from capa.training.trainer import Trainer

        capa_train_ldr, capa_val_ldr = make_capa_loaders(
            train_data, val_data,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
            batch_size=32,
        )

        capa_model = CAPAModel(
            embedding_dim=args.embedding_dim,
            loci=_HLA_LOCI,
            clinical_dim=4,
            interaction_dim=args.interaction_dim,
            survival_type="deephit",
            num_events=num_events,
            time_bins=time_bins,
            num_heads=4,
            num_layers=2,
        )
        trainer = Trainer(
            model=capa_model,
            train_loader=capa_train_ldr,
            val_loader=capa_val_ldr,
            max_epochs=args.epochs,
            patience=max(5, args.epochs // 4),
            runs_dir=runs_dir / "capa_run",
            checkpoint_every=0,
            device=args.device,
        )
        t0 = time.perf_counter()
        try:
            trainer.fit()

            # Build test embeddings (random — same seed offset)
            rng = np.random.default_rng(args.seed + 999)
            n_test = len(test_data["times"])
            d_emb = rng.standard_normal(
                (n_test, len(_HLA_LOCI), args.embedding_dim)
            ).astype(np.float32)
            r_emb = rng.standard_normal(
                (n_test, len(_HLA_LOCI), args.embedding_dim)
            ).astype(np.float32)
            c_emb = test_data["clinical_cont"]

            device = torch.device(args.device)
            import torch.nn.functional as F
            capa_model.eval()
            with torch.no_grad():
                dt = torch.tensor(d_emb, device=device)
                rt = torch.tensor(r_emb, device=device)
                ct = torch.tensor(c_emb, device=device)
                logits = capa_model(dt, rt, ct)
                bsz = logits.shape[0]
                joint = F.softmax(logits.view(bsz, -1), dim=-1).view(
                    bsz, num_events, time_bins
                )
                cif_capa = torch.cumsum(joint, dim=2).cpu().numpy()

            elapsed = time.perf_counter() - t0
            row = _eval(cif_capa)
            row["train_time_s"] = elapsed
            comparison["CAPA (synthetic emb)"] = row
            logger.info("CAPA done (%.1fs)", elapsed)
        except Exception as exc:
            import traceback
            logger.error("CAPA failed: %s\n%s", exc, traceback.format_exc())
            comparison["CAPA (synthetic emb)"] = {"error": str(exc)}

    # ── Print table ───────────────────────────────────────────────────
    display = {
        name: {
            "cindex": res.get("cindex", {}),
            "ibs": res.get("ibs", {}),
            "train_time_s": res.get("train_time_s", float("nan")),
            **({} if "error" not in res else {"error": res["error"]}),
        }
        for name, res in comparison.items()
    }
    _print_comparison_table(display, _EVENT_NAMES[:num_events])

    # ── Save JSON ─────────────────────────────────────────────────────
    save_dict: dict[str, Any] = {
        "n_subjects": args.n,
        "n_train": len(train_data["times"]),
        "n_test": len(test_data["times"]),
        "time_bins": time_bins,
        "num_events": num_events,
        "event_names": _EVENT_NAMES[:num_events],
        "models": {},
    }
    for name, res in comparison.items():
        if "error" in res:
            save_dict["models"][name] = {"error": res["error"]}
        else:
            save_dict["models"][name] = {
                "cindex": res["cindex"],
                "ibs": res["ibs"],
                "train_time_s": res.get("train_time_s"),
                "full_evaluation": res.get("full"),
            }

    out_path = runs_dir / "comparison.json"
    with out_path.open("w") as f:
        json.dump(save_dict, f, indent=2)
    logger.info("Comparison results saved to %s", out_path)


if __name__ == "__main__":
    main()
