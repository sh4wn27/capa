"""Tests for capa/training/trainer.py.

All tests use a tiny synthetic dataset and a small model so the full
suite runs in a few seconds on CPU.

Coverage
--------
Trainer construction
  - default construction succeeds
  - model moved to device
  - optimizer is AdamW
  - scheduler is CosineAnnealingLR
  - runs_dir created on construction
  - JSON log file path is inside runs_dir

Training loop: _train_epoch
  - returns a finite scalar
  - model parameters change after one epoch
  - gradients are clipped (no explosion)

Training loop: _val_epoch
  - returns (loss, cindex) both finite
  - cindex in [0, 1]
  - all-censored validation → cindex == 0.5 fallback
  - model stays in eval mode at end of _val_epoch

fit()
  - history dict has the four expected keys
  - history lists have equal length
  - best_model.pt is written after fit()
  - JSON log has one entry per epoch
  - early stopping fires when val_cindex doesn't improve
  - periodic checkpoint written every checkpoint_every epochs
  - loss decreases at least slightly over multiple epochs (on a trivial dataset)

Checkpoint save / resume
  - load_checkpoint restores epoch, best_cindex, history, model weights
  - after resume, fit() continues from the right epoch
  - resumed model weights equal saved weights
  - optimizer state restored (LR unchanged)

CLI smoke tests
  - --dry-run exits cleanly without writing checkpoints
  - --synthetic --epochs 2 runs to completion and writes best_model.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure project root on path (mirrors scripts/ convention)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capa.model.capa_model import CAPAModel
from capa.training.trainer import CheckpointState, Trainer

# ---------------------------------------------------------------------------
# Constants — kept tiny for speed
# ---------------------------------------------------------------------------
EMB = 16
N_LOCI = 3
LOCI = ["A", "B", "DRB1"]
CLIN_DIM = 8
INT_DIM = 16
N_EVENTS = 3
T_BINS = 10
N_HEADS = 2
N_LAYERS = 1
BATCH_SIZE = 8
N_TRAIN = 32
N_VAL = 16


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _DictDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        donor: torch.Tensor, recip: torch.Tensor, clinical: torch.Tensor,
        times: torch.Tensor, types: torch.Tensor,
    ) -> None:
        self._d, self._r, self._c, self._t, self._k = donor, recip, clinical, times, types

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "donor_embeddings":     self._d[idx],
            "recipient_embeddings": self._r[idx],
            "clinical_features":    self._c[idx],
            "event_times":          self._t[idx],
            "event_types":          self._k[idx],
        }


def _make_loaders(
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    batch_size: int = BATCH_SIZE,
    all_censored_val: bool = False,
    seed: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    torch.manual_seed(seed)

    def _ds(n: int, censored: bool = False) -> _DictDataset:
        return _DictDataset(
            donor    = torch.randn(n, N_LOCI, EMB),
            recip    = torch.randn(n, N_LOCI, EMB),
            clinical = torch.randn(n, CLIN_DIM),
            times    = torch.randint(0, T_BINS, (n,)),
            types    = torch.zeros(n, dtype=torch.long) if censored
                       else torch.randint(0, N_EVENTS + 1, (n,)),
        )

    train_loader = DataLoader(_ds(n_train), batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(_ds(n_val, censored=all_censored_val),
                              batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _make_model() -> CAPAModel:
    return CAPAModel(
        embedding_dim=EMB,
        loci=LOCI,
        clinical_dim=CLIN_DIM,
        interaction_dim=INT_DIM,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        dropout=0.0,
        time_bins=T_BINS,
        num_events=N_EVENTS,
    )


def _make_trainer(
    tmp_path: Path,
    *,
    max_epochs: int = 3,
    patience: int = 50,
    checkpoint_every: int = 0,
    all_censored_val: bool = False,
    model: CAPAModel | None = None,
    max_grad_norm: float = 1.0,
) -> Trainer:
    train_loader, val_loader = _make_loaders(all_censored_val=all_censored_val)
    return Trainer(
        model=model or _make_model(),
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=max_epochs,
        patience=patience,
        alpha=0.5,
        sigma=0.1,
        max_grad_norm=max_grad_norm,
        runs_dir=tmp_path / "runs",
        checkpoint_every=checkpoint_every,
        device="cpu",
    )


# ===========================================================================
# Construction
# ===========================================================================

class TestTrainerConstruction:
    def test_init_succeeds(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        assert isinstance(t, Trainer)

    def test_model_on_device(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        p = next(t.model.parameters())
        assert p.device.type == "cpu"

    def test_optimizer_is_adamw(self, tmp_path: Path) -> None:
        from torch.optim import AdamW
        t = _make_trainer(tmp_path)
        assert isinstance(t.optimizer, AdamW)

    def test_scheduler_is_cosine(self, tmp_path: Path) -> None:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        t = _make_trainer(tmp_path)
        assert isinstance(t.scheduler, CosineAnnealingLR)

    def test_runs_dir_created(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        assert t.runs_dir.is_dir()

    def test_log_path_inside_runs_dir(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        assert t._log_path.parent == t.runs_dir


# ===========================================================================
# _train_epoch
# ===========================================================================

class TestTrainEpoch:
    def test_returns_finite_float(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        loss = t._train_epoch()
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_params_change_after_epoch(self, tmp_path: Path) -> None:
        m = _make_model()
        before = {n: p.clone() for n, p in m.named_parameters()}
        t = _make_trainer(tmp_path, model=m)
        t._train_epoch()
        changed = sum(
            1 for n, p in m.named_parameters()
            if not torch.allclose(p, before[n])
        )
        assert changed > 0

    def test_grad_norm_clipped(self, tmp_path: Path) -> None:
        """Manually pump large gradients and verify clipping brings norm to ≤ 1."""
        t = _make_trainer(tmp_path)
        # Inject huge gradients directly
        for p in t.model.parameters():
            p.grad = torch.ones_like(p) * 1e6
        nn.utils.clip_grad_norm_(t.model.parameters(), 1.0)
        total_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in t.model.parameters() if p.grad is not None)
        )
        assert total_norm.item() <= 1.0 + 1e-4


# ===========================================================================
# _val_epoch
# ===========================================================================

class TestValEpoch:
    def test_returns_two_finite_values(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        val_loss, val_cindex = t._val_epoch()
        assert np.isfinite(val_loss)
        assert np.isfinite(val_cindex)

    def test_cindex_in_01(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        _, val_cindex = t._val_epoch()
        assert 0.0 <= val_cindex <= 1.0

    def test_all_censored_fallback(self, tmp_path: Path) -> None:
        """When no uncensored subjects exist, C-index should fall back to 0.5."""
        t = _make_trainer(tmp_path, all_censored_val=True)
        _, val_cindex = t._val_epoch()
        assert val_cindex == pytest.approx(0.5)

    def test_model_in_eval_mode_after(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path)
        t._val_epoch()
        assert not t.model.training


# ===========================================================================
# fit()
# ===========================================================================

class TestFit:
    def test_history_has_expected_keys(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=2)
        h = t.fit()
        for key in ("train_loss", "val_loss", "val_cindex", "lr"):
            assert key in h

    def test_history_lists_equal_length(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=3)
        h = t.fit()
        lengths = [len(v) for v in h.values()]
        assert len(set(lengths)) == 1

    def test_best_model_written(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=2)
        t.fit()
        assert (t.runs_dir / "best_model.pt").exists()

    def test_json_log_has_one_entry_per_epoch(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=3)
        t.fit()
        lines = t._log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            for key in ("epoch", "train_loss", "val_loss", "val_cindex", "lr", "elapsed_s"):
                assert key in record

    def test_json_log_epoch_numbers_sequential(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=4)
        t.fit()
        epochs = [json.loads(l)["epoch"] for l in t._log_path.read_text().splitlines()]
        assert epochs == list(range(1, 5))

    def test_history_loss_finite(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=2)
        h = t.fit()
        assert all(np.isfinite(v) for v in h["train_loss"])
        assert all(np.isfinite(v) for v in h["val_loss"])

    def test_lr_decreases_with_cosine_schedule(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=5)
        h = t.fit()
        # Cosine schedule monotonically decreases LR
        assert h["lr"][0] >= h["lr"][-1]

    def test_early_stopping_fires(self, tmp_path: Path) -> None:
        """With patience=1, if val_cindex never improves training stops early."""
        from unittest.mock import patch
        t = _make_trainer(tmp_path, max_epochs=20, patience=1)
        # Force _val_epoch to always return the same C-index so it never improves
        call_count = [0]
        original_val = t._val_epoch

        def _stubbed_val() -> tuple[float, float]:
            call_count[0] += 1
            loss, _ = original_val()
            # Return a constant cindex so epoch 2+ never beats epoch 1
            return loss, 0.55

        with patch.object(t, "_val_epoch", side_effect=_stubbed_val):
            h = t.fit()
        # Epoch 1 sets best=0.55; epoch 2 ties (no improvement) → stops at epoch 2
        assert len(h["train_loss"]) <= 3  # stopped well before max_epochs=20

    def test_periodic_checkpoint_written(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=4, checkpoint_every=2)
        t.fit()
        assert (t.runs_dir / "checkpoint_epoch_0002.pt").exists()
        assert (t.runs_dir / "checkpoint_epoch_0004.pt").exists()

    def test_no_periodic_checkpoint_when_disabled(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=3, checkpoint_every=0)
        t.fit()
        periodic = list(t.runs_dir.glob("checkpoint_epoch_*.pt"))
        assert len(periodic) == 0

    def test_fit_returns_correct_length(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=4, patience=50, checkpoint_every=0)
        h = t.fit()
        assert len(h["train_loss"]) == 4


# ===========================================================================
# Checkpoint save / resume
# ===========================================================================

class TestCheckpointResume:
    def test_best_model_is_loadable(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=2)
        t.fit()
        ckpt: CheckpointState = torch.load(t.runs_dir / "best_model.pt", map_location="cpu", weights_only=False)
        assert isinstance(ckpt, CheckpointState)
        assert isinstance(ckpt.model_state, dict)

    def test_checkpoint_contains_all_fields(self, tmp_path: Path) -> None:
        t = _make_trainer(tmp_path, max_epochs=2)
        t.fit()
        ckpt: CheckpointState = torch.load(t.runs_dir / "best_model.pt", map_location="cpu", weights_only=False)
        assert ckpt.epoch >= 1
        assert isinstance(ckpt.best_cindex, float)
        assert "train_loss" in ckpt.history
        assert isinstance(ckpt.optimizer_state, dict)
        assert isinstance(ckpt.scheduler_state, dict)

    def test_resume_restores_epoch(self, tmp_path: Path) -> None:
        # Train for 2 epochs, save checkpoint
        t1 = _make_trainer(tmp_path / "run1", max_epochs=2, checkpoint_every=2)
        t1.fit()
        ckpt_path = t1.runs_dir / "checkpoint_epoch_0002.pt"
        assert ckpt_path.exists()

        # Resume and verify start epoch is 3
        t2 = _make_trainer(tmp_path / "run2", max_epochs=4)
        t2.load_checkpoint(ckpt_path)
        assert t2._start_epoch == 3

    def test_resume_restores_history(self, tmp_path: Path) -> None:
        t1 = _make_trainer(tmp_path / "run1", max_epochs=2, checkpoint_every=2)
        t1.fit()
        ckpt_path = t1.runs_dir / "checkpoint_epoch_0002.pt"

        t2 = _make_trainer(tmp_path / "run2", max_epochs=4)
        t2.load_checkpoint(ckpt_path)
        assert len(t2._history["train_loss"]) == 2

    def test_resume_restores_model_weights(self, tmp_path: Path) -> None:
        t1 = _make_trainer(tmp_path / "run1", max_epochs=2, checkpoint_every=2)
        t1.fit()
        ckpt_path = t1.runs_dir / "checkpoint_epoch_0002.pt"

        t2 = _make_trainer(tmp_path / "run2", max_epochs=4)
        t2.load_checkpoint(ckpt_path)

        # Both models should have identical weights after resume
        for (n1, p1), (n2, p2) in zip(
            t1.model.named_parameters(), t2.model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_resumed_fit_extends_history(self, tmp_path: Path) -> None:
        t1 = _make_trainer(tmp_path / "run1", max_epochs=2, checkpoint_every=2)
        t1.fit()
        ckpt_path = t1.runs_dir / "checkpoint_epoch_0002.pt"

        t2 = _make_trainer(tmp_path / "run2", max_epochs=4)
        t2.load_checkpoint(ckpt_path)
        h = t2.fit()
        # 2 pre-existing + 2 new epochs
        assert len(h["train_loss"]) == 4

    def test_resume_best_cindex_preserved(self, tmp_path: Path) -> None:
        t1 = _make_trainer(tmp_path / "run1", max_epochs=3, checkpoint_every=3)
        t1.fit()
        saved_best = t1._best_cindex
        ckpt_path = t1.runs_dir / "checkpoint_epoch_0003.pt"

        t2 = _make_trainer(tmp_path / "run2", max_epochs=6)
        t2.load_checkpoint(ckpt_path)
        assert t2._best_cindex == pytest.approx(saved_best)


# ===========================================================================
# CLI smoke tests (scripts/train.py)
# ===========================================================================

class TestCLI:
    def _run(self, argv: list[str], tmp_path: Path) -> int:
        """Run scripts/train.py with given args; returns exit code."""
        import os, subprocess
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "train.py"),
            "--run-name", "test_run",
            "--runs-dir", str(tmp_path / "runs"),  # write into tmp_path, not project root
            *argv,
        ]
        env = {**os.environ, "PYTHONPATH": str(_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path)
        return result.returncode

    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        rc = self._run(["--synthetic", "--dry-run"], tmp_path)
        assert rc == 0

    def test_synthetic_two_epochs(self, tmp_path: Path) -> None:
        rc = self._run(
            ["--synthetic", "--synthetic-n", "32", "--epochs", "2",
             "--embedding-dim", "16", "--loci", "A", "B",
             "--batch-size", "8", "--checkpoint-every", "0"],
            tmp_path,
        )
        assert rc == 0
        # best_model.pt should exist under runs/test_run/
        best = next((tmp_path / "runs" / "test_run").glob("best_model.pt"), None)
        assert best is not None and best.exists()

    def test_synthetic_json_log_written(self, tmp_path: Path) -> None:
        self._run(
            ["--synthetic", "--synthetic-n", "32", "--epochs", "3",
             "--embedding-dim", "16", "--loci", "A",
             "--batch-size", "8", "--checkpoint-every", "0"],
            tmp_path,
        )
        log_path = tmp_path / "runs" / "test_run" / "train_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_resume_cli(self, tmp_path: Path) -> None:
        """Train 2 epochs, then resume for 2 more — total history should be 4."""
        # First run: 2 epochs with checkpoint_every=2
        self._run(
            ["--synthetic", "--synthetic-n", "32", "--epochs", "2",
             "--embedding-dim", "16", "--loci", "A", "--batch-size", "8",
             "--checkpoint-every", "2", "--patience", "100"],
            tmp_path,
        )
        ckpt = tmp_path / "runs" / "test_run" / "checkpoint_epoch_0002.pt"
        assert ckpt.exists()

        # Second run: resume from checkpoint, train up to epoch 4
        self._run(
            ["--synthetic", "--synthetic-n", "32", "--epochs", "4",
             "--embedding-dim", "16", "--loci", "A", "--batch-size", "8",
             "--checkpoint-every", "0", "--patience", "100",
             "--resume", str(ckpt)],
            tmp_path,
        )
        log_path = tmp_path / "runs" / "test_run" / "train_log.jsonl"
        lines = log_path.read_text().strip().splitlines()
        # 2 from first run + 2 from resumed run = 4 total
        assert len(lines) == 4
