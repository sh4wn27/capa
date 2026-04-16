"""CAPA evaluation entry point.

Usage
-----
Evaluate a trained checkpoint against synthetic data (quick smoke-test):
    uv run python scripts/evaluate.py --checkpoint runs/my_run/best_model.pt --synthetic

Evaluate against real test data (requires preprocessing):
    uv run python scripts/evaluate.py --checkpoint runs/my_run/best_model.pt

Skip bootstrap CIs for a fast pass:
    uv run python scripts/evaluate.py --checkpoint runs/my_run/best_model.pt \\
        --synthetic --n-bootstrap 0

Save results JSON to a custom path:
    uv run python scripts/evaluate.py --checkpoint runs/my_run/best_model.pt \\
        --synthetic --output-path runs/my_run/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

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
        description="Evaluate a trained CAPA competing-risks model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML config file (e.g. configs/default.yaml). "
             "CLI flags override YAML values; YAML overrides pydantic defaults.",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a .pt checkpoint saved by the Trainer.",
    )
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps")

    # Data source
    p.add_argument(
        "--synthetic", action="store_true",
        help="Evaluate on a tiny synthetic test set (no real data required).",
    )
    p.add_argument("--synthetic-n", type=int, default=100)
    p.add_argument("--synthetic-seed", type=int, default=99)

    # Architecture (must match the checkpoint)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--loci", nargs="+", default=None)
    p.add_argument("--interaction-dim", type=int, default=None)
    p.add_argument(
        "--survival-type", choices=["deephit", "cause_specific"],
        default="deephit",
    )

    # Evaluation settings
    p.add_argument(
        "--n-bootstrap", type=int, default=200,
        help="Bootstrap iterations for CIs (0 = skip CIs, much faster).",
    )
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--n-calibration-bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # Output
    p.add_argument(
        "--output-path", type=Path, default=None,
        help=(
            "Where to write results.json.  Defaults to "
            "<checkpoint_dir>/results.json."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def _make_synthetic_batch(
    n: int,
    embedding_dim: int,
    n_loci: int,
    clinical_dim: int,
    time_bins: int,
    num_events: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return a single big dict-batch (no DataLoader needed for evaluation)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {
        "donor_embeddings":     torch.randn(n, n_loci, embedding_dim),
        "recipient_embeddings": torch.randn(n, n_loci, embedding_dim),
        "clinical_features":    torch.randn(n, clinical_dim),
        "event_times":          torch.randint(0, time_bins, (n,)),
        "event_types":          torch.randint(0, num_events + 1, (n,)),
    }


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def _print_results(result: "EvaluationResult") -> None:  # noqa: F821
    from capa.training.evaluate import EvaluationResult  # local import

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  CAPA Evaluation  │  n={result.n_subjects} subjects")
    print(sep)

    for em in result.events:
        ci = em.cindex
        pct = int(ci.ci_level * 100)
        print(f"\n  Event: {em.event_name}")
        if ci.n_bootstrap > 0:
            print(
                f"    C-index : {ci.value:.4f}  "
                f"[{pct}% CI {ci.ci_lower:.4f}–{ci.ci_upper:.4f}]"
                f"  (n_boot={ci.n_bootstrap})"
            )
        else:
            print(f"    C-index : {ci.value:.4f}")

        if em.ibs is not None:
            ibs = em.ibs
            if ibs.n_bootstrap > 0:
                print(
                    f"    IBS     : {ibs.value:.4f}  "
                    f"[{pct}% CI {ibs.ci_lower:.4f}–{ibs.ci_upper:.4f}]"
                )
            else:
                print(f"    IBS     : {ibs.value:.4f}")

        print(f"    Brier scores per time point:")
        for t, bs in sorted(em.brier_scores.items()):
            if bs.n_bootstrap > 0:
                print(
                    f"      t={t:.1f}  BS={bs.value:.4f}  "
                    f"[{pct}% CI {bs.ci_lower:.4f}–{bs.ci_upper:.4f}]"
                )
            else:
                print(f"      t={t:.1f}  BS={bs.value:.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    from capa.config import get_config
    from capa.model.capa_model import CAPAModel
    from capa.training.evaluate import evaluate_all
    from capa.training.trainer import CheckpointState

    cfg = get_config(config_file=args.config)

    # ── Architecture settings ─────────────────────────────────────────
    embedding_dim   = args.embedding_dim   or cfg.embedding.embedding_dim
    loci            = args.loci            or cfg.model.hla_loci
    n_loci          = len(loci)
    interaction_dim = args.interaction_dim or cfg.model.interaction_dim
    clinical_dim    = cfg.model.clinical_dim
    time_bins       = cfg.model.time_bins
    num_events      = cfg.model.num_events
    event_names     = ["GvHD", "Relapse", "TRM"][:num_events]

    device = torch.device(args.device)

    # ── Build model ───────────────────────────────────────────────────
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
        dropout=0.0,   # eval mode — dropout off anyway, but be explicit
    )

    # ── Load checkpoint ───────────────────────────────────────────────
    logger.info("Loading checkpoint from %s", args.checkpoint)
    ckpt: CheckpointState = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()
    logger.info(
        "Checkpoint loaded  (epoch=%d  best_cindex=%.4f)",
        ckpt.epoch, ckpt.best_cindex,
    )

    # ── Test data ─────────────────────────────────────────────────────
    if args.synthetic:
        logger.info(
            "Using synthetic test set  (n=%d)", args.synthetic_n,
        )
        batch = _make_synthetic_batch(
            n=args.synthetic_n,
            embedding_dim=embedding_dim,
            n_loci=n_loci,
            clinical_dim=clinical_dim,
            time_bins=time_bins,
            num_events=num_events,
            seed=args.synthetic_seed,
        )
    else:
        logger.error(
            "Real-data test-set loading is not yet implemented.\n"
            "Use --synthetic for a quick evaluation smoke-test."
        )
        sys.exit(1)

    # ── Run model → get CIF ───────────────────────────────────────────
    import torch.nn.functional as F

    with torch.no_grad():
        donor    = batch["donor_embeddings"].to(device)
        recip    = batch["recipient_embeddings"].to(device)
        clinical = batch["clinical_features"].to(device)
        logits   = model(donor, recip, clinical)  # (n, K, T)

        n_sub, K, T = logits.shape
        if args.survival_type == "deephit":
            joint = F.softmax(logits.view(n_sub, -1), dim=-1).view(n_sub, K, T)
            cif_t = torch.cumsum(joint, dim=2).cpu().numpy()
        else:
            from capa.model.survival import hazards_to_cif
            cif_t = hazards_to_cif(logits).cpu().numpy()   # (n, K, T)

    event_times = batch["event_times"].numpy().astype(np.float64)
    event_types = batch["event_types"].numpy().astype(np.int64)
    time_bins_arr = np.arange(time_bins, dtype=np.float64)

    # ── Evaluate ──────────────────────────────────────────────────────
    logger.info("Running evaluation (n_bootstrap=%d)…", args.n_bootstrap)
    result = evaluate_all(
        cif=cif_t,
        event_times=event_times,
        event_types=event_types,
        event_names=event_names,
        time_bins=time_bins_arr,
        eval_times=None,   # auto quartiles
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        n_calibration_bins=args.n_calibration_bins,
        seed=args.seed,
    )

    # ── Print table ───────────────────────────────────────────────────
    _print_results(result)

    # ── Save JSON ─────────────────────────────────────────────────────
    output_path = (
        args.output_path
        if args.output_path is not None
        else args.checkpoint.parent / "results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
