"""Evaluation entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained CAPA checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model .pt file")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger.info("Evaluating checkpoint: %s", args.checkpoint)
    # TODO: Load model, run on test split, compute C-index and Brier scores


if __name__ == "__main__":
    main()
