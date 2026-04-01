"""Full data preprocessing pipeline: load → clean → split → save."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAPA data preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config override (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    parse_args()

    from capa.config import get_config
    from capa.data.loader import load_bmt
    from capa.data.splits import make_splits

    cfg = get_config()

    if not cfg.data.bmt_path.exists():
        logger.error(
            "BMT dataset not found at %s. Download from: "
            "https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children",
            cfg.data.bmt_path,
        )
        raise SystemExit(1)

    df = load_bmt(cfg.data.bmt_path)

    # TODO: Set event_col once column mapping is confirmed.
    train, val, test = make_splits(
        df,
        event_col="event_type",  # TODO: update with actual column name
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        random_seed=cfg.data.random_seed,
    )

    cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cfg.data.processed_dir / "train.parquet", index=False)
    val.to_parquet(cfg.data.processed_dir / "val.parquet", index=False)
    test.to_parquet(cfg.data.processed_dir / "test.parquet", index=False)

    logger.info("Preprocessing complete. Splits saved to %s", cfg.data.processed_dir)


if __name__ == "__main__":
    main()
