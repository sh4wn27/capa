"""Generate all publication-ready figures programmatically."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate CAPA paper figures")
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURES_DIR,
        help="Output directory for figures",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figures → %s", args.out)
    # TODO: Implement figure generation functions (EDA, UMAP, CIF curves, attention maps)


if __name__ == "__main__":
    main()
