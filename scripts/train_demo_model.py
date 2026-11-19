#!/usr/bin/env python3
"""Train the signed-difference CAPA model served by the public demo.

The demo (HF Space ``coconutmocha/capa``) needs a checkpoint whose outputs
respond to HLA input.  No public dataset has allele-level typing with
outcomes, so the demo model is trained on the directional simulation of
``scripts/run_directional_simulation.py`` (manuscript ``sec:directional``),
with one change: each locus is mismatched with probability ``p_mismatch``
(default 0.5) instead of always, so that matched loci (signed difference
= 0) are in-distribution at inference time.

The model is the manuscript's primary architecture: frozen ESM-2 allele
embeddings -> signed per-locus difference (donor - recipient) -> locus
self-attention with ``proj_dim=64`` -> DeepHit head.

Inputs the model actually uses: the five HLA loci and recipient age.  The
simulation holds donor age and CD34 dose constant and draws a binary disease
risk; the checkpoint records this under ``clinical_adapter`` so the server
can build matching inputs.

**Outputs are illustrative.**  The GvHD hazard in the simulation is driven by
a random, seeded direction in embedding space, not by any known biology.

Usage
-----
    uv run python scripts/train_demo_model.py --device mps
    uv run python scripts/train_demo_model.py --out hf-space/runs/best/model.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_directional_simulation import generate_directional_cohort  # noqa: E402
from scripts.run_haplo_simulation import (  # noqa: E402
    EMB_PATH,
    LOCI,
    MAX_DAYS,
    N_TIME_BINS,
    SEED,
    T365_BIN,
    build_clinical_tensors,
    build_embedding_tensors,
    eval_cindex,
)

from capa.model.capa_model import CAPAModel  # noqa: E402
from capa.model.losses import deephit_loss  # noqa: E402
from capa.training.evaluate import concordance_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT = PROJECT_ROOT / "runs" / "demo" / "model.pt"

# Architecture of the manuscript's primary model (469,580 parameters).
MODEL_CONFIG: dict[str, object] = {
    "embedding_dim": 1280,
    "loci": LOCI,
    "clinical_dim": 32,
    "interaction_dim": 128,
    "survival_type": "deephit",
    "num_events": 3,
    "time_bins": N_TIME_BINS,
    "event_names": ["GvHD", "Relapse", "TRM"],
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.2,
    "proj_dim": 64,
}


def mean_val_cindex(
    model: CAPAModel, diff: torch.Tensor, cont: torch.Tensor, cat: torch.Tensor,
    etime_days: np.ndarray, etype: np.ndarray,
) -> float:
    """Mean 1-year C-index over the three events (no bootstrap)."""
    model.eval()
    with torch.no_grad():
        cif = model.cif(diff, diff, model.clinical_encoder(cont, cat)).cpu().numpy()
    cs = [
        concordance_index(etime_days, cif[:, k, T365_BIN].astype(np.float64), etype == k + 1)
        for k in range(3)
    ]
    return float(np.nanmean(cs))


def train(
    model: CAPAModel,
    diff_tr: torch.Tensor, cont_tr: torch.Tensor, cat_tr: torch.Tensor,
    times_tr: torch.Tensor, types_tr: torch.Tensor,
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    epochs: int, lr: float, batch_size: int, patience: int, seed: int,
) -> tuple[int, float]:
    """Pure-NLL DeepHit training with early stopping on mean validation C.

    The paper's ``train_capa`` selects on GvHD alone, which for the demo kept
    an epoch-10 model that had not learned relapse at all.  Selecting on the
    mean over all three events keeps every displayed curve meaningful.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    rng = np.random.default_rng(seed)
    best_c, best_ep, since = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] = {}
    n = diff_tr.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        for b in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            bt = torch.from_numpy(b).long()
            d = diff_tr[bt]
            logits = model(d, d, model.clinical_encoder(cont_tr[bt], cat_tr[bt]))
            loss = deephit_loss(logits, times_tr[bt].to(d.device), types_tr[bt].to(d.device),
                                alpha=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 10 == 0:
            c = mean_val_cindex(model, *val)
            if c > best_c:
                best_c, best_ep, since = c, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 10
            if ep % 50 == 0:
                logger.info("epoch %3d  loss=%.4f  val mean C=%.3f  best=%.3f@%d",
                            ep, loss.item(), c, best_c, best_ep)
            if since >= patience:
                logger.info("early stop at epoch %d", ep)
                break
    model.load_state_dict(best_state)
    return best_ep, best_c


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--p-mismatch", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    """Generate the demo cohort, train, evaluate on a held-out split, save."""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with h5py.File(EMB_PATH, "r") as f:
        embs = {k: f[k][:] for k in f.keys()}
    df, _w = generate_directional_cohort(
        args.n, embs, rng, seed=args.seed, p_mismatch=args.p_mismatch,
    )
    logger.info("cohort n=%d, events=%s, mean mismatches=%.2f", len(df),
                np.bincount(df["event_type"], minlength=4).tolist(),
                df["n_mismatches"].mean())

    idx_trval, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.10, random_state=args.seed,
        stratify=df["event_type"].values)
    idx_tr, idx_va = train_test_split(
        idx_trval, test_size=0.111, random_state=args.seed,
        stratify=df.iloc[idx_trval]["event_type"].values)

    donor, recip = build_embedding_tensors(df, embs)
    donor, recip = donor.to(device), recip.to(device)
    cont, cat = build_clinical_tensors(df, device)
    t_bins = np.floor(df["survival_time_days"].values / MAX_DAYS * (N_TIME_BINS - 1))
    times = torch.from_numpy(t_bins.astype(np.int64).clip(0, N_TIME_BINS - 1))
    types = torch.from_numpy(df["event_type"].values.astype(np.int64))
    tr, va, te = (torch.from_numpy(i).long() for i in (idx_tr, idx_va, idx_te))

    model = CAPAModel(**MODEL_CONFIG).to(device)  # type: ignore[arg-type]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable params: %d", n_params)

    diff = donor - recip  # signed, direction-preserving
    etime_days = df["survival_time_days"].values.astype(np.float64)
    etype = df["event_type"].values
    val = (diff[va], cont[va], cat[va], etime_days[idx_va], etype[idx_va])
    best_epoch, best_val = train(
        model, diff[tr], cont[tr], cat[tr], times[tr], types[tr], val,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        patience=args.patience, seed=args.seed,
    )
    logger.info("best epoch %d (val mean C=%.3f)", best_epoch, best_val)

    model.eval()
    with torch.no_grad():
        cif_te = model.cif(diff[te], diff[te], model.clinical_encoder(cont[te], cat[te]))
    etime_te = df["survival_time_days"].values[idx_te]
    etype_te = df["event_type"].values[idx_te]
    test = eval_cindex(cif_te.cpu().numpy(), etime_te, etype_te)
    logger.info("held-out C-index: %s",
                {k: round(v["cindex"], 3) for k, v in test.items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "version": f"demo-signed-diff-sim-seed{args.seed}",
        "config": MODEL_CONFIG,
        "input_mode": "signed_diff",
        "clinical_adapter": {
            "name": "directional_sim",
            "cont": ["age_recipient/100", "0.45 (fixed donor age)",
                     "0.5 (fixed CD34)", "disease_risk (0/1)"],
            "cat0": "2 if disease_risk else 1",
            "disease_risk_default": 0,
        },
        "training": {
            "data": "directional simulation (not patient data)",
            "n": args.n, "p_mismatch": args.p_mismatch, "seed": args.seed,
            "best_epoch": best_epoch, "val_mean_cindex": best_val, "n_params": n_params,
            "selection": "mean 1-year C-index over GvHD, relapse, TRM",
            "test_cindex": {k: v["cindex"] for k, v in test.items()},
        },
    }, args.out)
    logger.info("saved %s", args.out)
    print(json.dumps({k: round(v["cindex"], 3) for k, v in test.items()}))


if __name__ == "__main__":
    main()
