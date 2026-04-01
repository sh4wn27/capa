"""Training entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the CAPA model")
    parser.add_argument("--config", type=Path, default=None, help="YAML config override")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    from capa.config import get_config
    cfg = get_config()

    if args.seed is not None:
        cfg.training.random_seed = args.seed

    logger.info("Training config: %s", cfg.training)
    # TODO: Build DataLoaders, instantiate CAPAModel, call Trainer.fit()


if __name__ == "__main__":
    main()
