"""CAPA training entry point.

Usage
-----
Basic run (CPU, all defaults from config):
    uv run python scripts/train.py

Override common knobs:
    uv run python scripts/train.py --device cuda --lr 5e-4 --epochs 100

Resume from a checkpoint:
    uv run python scripts/train.py --resume runs/my_run/checkpoint_epoch_0050.pt

Dry-run to verify data pipeline without training:
    uv run python scripts/train.py --dry-run

Synthetic dataset (no real data required):
    uv run python scripts/train.py --synthetic --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- resolve project root so `capa` is importable when run as a script ------
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CAPA competing-risks survival model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config file
    p.add_argument("--config", type=Path, default=None,
                   help="Path to a YAML config file (e.g. configs/default.yaml). "
                        "CLI flags override YAML values; YAML overrides pydantic defaults.")

    # Run identity
    p.add_argument("--run-name", default=None,
                   help="Sub-directory name inside runs_dir. Defaults to a timestamp.")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help="Override the runs root directory from config.")

    # Data
    p.add_argument("--synthetic", action="store_true",
                   help="Use a small synthetic dataset instead of the real UCI BMT data.")
    p.add_argument("--synthetic-n", type=int, default=200,
                   help="Number of synthetic subjects (only with --synthetic).")
    p.add_argument("--synthetic-seed", type=int, default=42)

    # Architecture
    p.add_argument("--embedding-dim", type=int, default=None,
                   help="Override embedding_dim (default from config, 1280 for ESM-2).")
    p.add_argument("--loci", nargs="+", default=None,
                   help="HLA loci to use, e.g. --loci A B DRB1")
    p.add_argument("--interaction-dim", type=int, default=None)
    p.add_argument("--survival-type", choices=["deephit", "cause_specific"],
                   default="deephit")

    # Training
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs from config.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning_rate from config.")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--alpha", type=float, default=None,
                   help="DeepHit ranking loss weight (0–1).")
    p.add_argument("--sigma", type=float, default=None,
                   help="DeepHit ranking loss bandwidth.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=10)

    # Resume
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint .pt file to resume from.")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Build data pipeline and model, then exit without training.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic DataLoader (no real data needed)
# ---------------------------------------------------------------------------

def _make_synthetic_loaders(
    n: int,
    embedding_dim: int,
    n_loci: int,
    clinical_dim: int,
    time_bins: int,
    num_events: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Generate a tiny synthetic dataset for smoke-testing the training loop."""
    from typing import Any

    rng = torch.Generator()
    rng.manual_seed(seed)

    donor    = torch.randn(n, n_loci, embedding_dim)
    recip    = torch.randn(n, n_loci, embedding_dim)
    clinical = torch.randn(n, clinical_dim)
    times    = torch.randint(0, time_bins, (n,))
    types    = torch.randint(0, num_events + 1, (n,))  # 0=censored

    # 80/20 split
    split = int(0.8 * n)
    def _loader(idx: range, shuffle: bool) -> DataLoader[Any]:
        ds = _DictDataset(donor[idx], recip[idx], clinical[idx], times[idx], types[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _loader(range(split), True), _loader(range(split, n), False)


class _DictDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Thin wrapper that yields the standard batch dict from tensors."""

    def __init__(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
        times: torch.Tensor,
        types: torch.Tensor,
    ) -> None:
        self._donor    = donor
        self._recip    = recip
        self._clinical = clinical
        self._times    = times
        self._types    = types

    def __len__(self) -> int:
        return len(self._donor)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "donor_embeddings":     self._donor[idx],
            "recipient_embeddings": self._recip[idx],
            "clinical_features":    self._clinical[idx],
            "event_times":          self._times[idx],
            "event_types":          self._types[idx],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Reproducibility ───────────────────────────────────────────────
    from capa.config import get_config
    cfg = get_config(config_file=args.config)

    seed = args.seed if args.seed is not None else cfg.training.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Random seed: %d", seed)

    # ── Resolve run directory ─────────────────────────────────────────
    import datetime
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_runs_dir = args.runs_dir if args.runs_dir is not None else cfg.training.runs_dir
    runs_dir = base_runs_dir / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", runs_dir)

    # ── Architecture settings ─────────────────────────────────────────
    embedding_dim   = args.embedding_dim or cfg.embedding.embedding_dim
    loci            = args.loci          or cfg.model.hla_loci
    n_loci          = len(loci)
    interaction_dim = args.interaction_dim or cfg.model.interaction_dim
    clinical_dim    = cfg.model.clinical_dim
    time_bins       = cfg.model.time_bins
    num_events      = cfg.model.num_events

    # ── Training hyperparameters ──────────────────────────────────────
    lr           = args.lr           or cfg.training.learning_rate
    weight_decay = args.weight_decay or cfg.training.weight_decay
    max_epochs   = args.epochs       or cfg.training.max_epochs
    patience     = args.patience     or cfg.training.patience
    batch_size   = args.batch_size   or cfg.training.batch_size
    alpha        = args.alpha        if args.alpha is not None else cfg.training.alpha
    sigma        = args.sigma        if args.sigma is not None else cfg.training.sigma

    # ── Data ──────────────────────────────────────────────────────────
    if args.synthetic:
        logger.info(
            "Using synthetic dataset  (n=%d, embedding_dim=%d, n_loci=%d)",
            args.synthetic_n, embedding_dim, n_loci,
        )
        train_loader, val_loader = _make_synthetic_loaders(
            n=args.synthetic_n,
            embedding_dim=embedding_dim,
            n_loci=n_loci,
            clinical_dim=clinical_dim,
            time_bins=time_bins,
            num_events=num_events,
            batch_size=batch_size,
            seed=args.synthetic_seed,
        )
    else:
        # Real data path — requires preprocessing to have been run first
        from capa.data.loader import load_bmt_dataset
        from capa.data.splits import make_splits
        from capa.embeddings.cache import EmbeddingCache

        logger.info("Loading UCI BMT dataset from %s", cfg.data.bmt_path)
        try:
            df = load_bmt_dataset(cfg.data.bmt_path)
        except FileNotFoundError:
            logger.error(
                "BMT dataset not found at %s.\n"
                "Run:  uv run python scripts/preprocess.py\n"
                "Or use --synthetic for a smoke-test without real data.",
                cfg.data.bmt_path,
            )
            sys.exit(1)

        train_df, val_df, _ = make_splits(
            df,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            random_seed=seed,
        )
        logger.info("Split: train=%d  val=%d", len(train_df), len(val_df))

        # Build DataLoaders from the real dataset
        # (requires embedding cache + feature engineering — placeholder)
        logger.error(
            "Real-data DataLoader construction not yet implemented.\n"
            "Use --synthetic to train on synthetic data."
        )
        sys.exit(1)

    if args.dry_run:
        logger.info("--dry-run: data pipeline OK, exiting before training.")
        return

    # ── Model ─────────────────────────────────────────────────────────
    from capa.model.capa_model import CAPAModel
    model = CAPAModel(
        embedding_dim=embedding_dim,
        loci=loci,
        clinical_dim=clinical_dim,
        interaction_dim=interaction_dim,
        survival_type=args.survival_type,
        num_events=num_events,
        time_bins=time_bins,
        num_heads=cfg.model.interaction_heads,
        num_layers=cfg.model.interaction_layers,
        dropout=cfg.model.dropout,
        use_pos_embed=cfg.model.interaction_pos_embed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters  survival=%s", n_params, args.survival_type)

    # ── Trainer ───────────────────────────────────────────────────────
    from capa.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        lr_patience=cfg.training.lr_patience,
        lr_factor=cfg.training.lr_factor,
        alpha=alpha,
        sigma=sigma,
        max_grad_norm=cfg.training.max_grad_norm,
        runs_dir=runs_dir,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        survival_type=args.survival_type,
    )

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training  (max_epochs=%d  patience=%d)", max_epochs, patience)
    history = trainer.fit()

    # ── Summary ───────────────────────────────────────────────────────
    best_cindex = max(history["val_cindex"]) if history["val_cindex"] else float("nan")
    final_epoch = len(history["train_loss"])
    logger.info(
        "Training complete  |  epochs=%d  best_val_cindex=%.4f  "
        "checkpoints in %s",
        final_epoch, best_cindex, runs_dir,
    )


if __name__ == "__main__":
    main()
