"""Training loop for the CAPA competing-risks model.

Features
--------
* AdamW optimiser with configurable LR and weight decay
* ReduceLROnPlateau LR schedule (factor & patience configurable; matches paper)
* Early stopping on validation C-index (higher is better; patience configurable)
* Gradient clipping at ``max_grad_norm``
* Checkpoint saving: best model + periodic every ``checkpoint_every`` epochs
* Resume from any checkpoint (model weights, optimiser state, scheduler state,
  epoch counter, best metric, history)
* Structured console logging (one line per epoch) + JSON log file

Batch format
------------
DataLoaders must yield dicts with keys:
  ``donor_embeddings``    — float32 (batch, n_loci, embedding_dim)
  ``recipient_embeddings``— float32 (batch, n_loci, embedding_dim)
  ``clinical_features``  — float32 (batch, clinical_dim)
  ``event_times``        — long    (batch,)
  ``event_types``        — long    (batch,)   0=censored, 1..K=event

C-index for early stopping
--------------------------
At the end of each epoch the validation set is scored with a fast cause-specific
C-index (cause 1 by default, usually the highest-incidence event).  The metric
is computed from the model's predicted CIF at the *median* event-time bin, which
is a single-number risk score comparable across subjects.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from capa.model.losses import deephit_loss
from capa.training.evaluate import concordance_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

@dataclass
class CheckpointState:
    """Everything needed to resume training from a checkpoint.

    Attributes
    ----------
    epoch : int
        The epoch that was just completed (1-indexed).
    best_cindex : float
        Best validation C-index seen so far.
    history : dict[str, list[float]]
        Training history lists.
    model_state : dict[str, Any]
        ``model.state_dict()``
    optimizer_state : dict[str, Any]
        ``optimizer.state_dict()``
    scheduler_state : dict[str, Any]
        ``scheduler.state_dict()``
    """

    epoch: int
    best_cindex: float
    history: dict[str, list[float]]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Train a CAPA model with early stopping on validation C-index.

    Parameters
    ----------
    model : nn.Module
        The CAPA model (or any module accepting the standard batch format).
    train_loader : DataLoader
        Yields batches of training data.
    val_loader : DataLoader
        Yields batches of validation data.
    learning_rate : float
        Initial AdamW learning rate.
    weight_decay : float
        AdamW L2 regularisation coefficient.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience in epochs (C-index not improving).
    lr_patience : int
        ReduceLROnPlateau patience (validation loss not improving).
    lr_factor : float
        ReduceLROnPlateau multiplicative decay factor (< 1).
    alpha : float
        DeepHit loss ranking weight (0 = pure NLL, 1 = pure ranking).
    sigma : float
        DeepHit ranking loss bandwidth.
    max_grad_norm : float
        Gradient-clipping max norm.
    esm_params : list[torch.nn.Parameter] | None
        If provided, these parameters are added as a second AdamW parameter
        group with learning rate ``esm_lr_scale × learning_rate``.  Used for
        ESM-2 partial fine-tuning (V2: ``esm_finetune_layers > 0``).
    esm_lr_scale : float
        Learning rate multiplier for the ESM-2 fine-tune parameter group.
        Default is 0.01 (i.e. LR is 100× smaller than the main network).
    runs_dir : Path
        Directory to write checkpoints and the JSON log file.
    checkpoint_every : int
        Save a ``checkpoint_epoch_<N>.pt`` every this many epochs (0 = never).
    device : str
        Torch device string.
    cindex_event : int
        Which competing event (1-indexed) to use for early-stopping C-index.
    survival_type : str
        ``"deephit"`` or ``"cause_specific"`` — determines how to compute CIF
        from model output for the C-index calculation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        esm_params: list[nn.Parameter] | None = None,
        esm_lr_scale: float = 0.01,
        alpha: float = 0.5,
        sigma: float = 0.1,
        max_grad_norm: float = 1.0,
        runs_dir: Path = Path("runs"),
        checkpoint_every: int = 10,
        device: str = "cpu",
        cindex_event: int = 1,
        survival_type: str = "deephit",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.alpha = alpha
        self.sigma = sigma
        self.max_grad_norm = max_grad_norm
        self.runs_dir = Path(runs_dir)
        self.checkpoint_every = checkpoint_every
        self.cindex_event = cindex_event
        self.survival_type = survival_type
        self.device = torch.device(device)

        param_groups: list[dict[str, object]] = [
            {"params": list(model.parameters()), "lr": learning_rate},
        ]
        if esm_params:
            param_groups.append({
                "params": esm_params,
                "lr": learning_rate * esm_lr_scale,
                "name": "esm_finetune",
            })
            logger.info(
                "ESM-2 fine-tune param group: %d params at lr=%.2e",
                sum(p.numel() for p in esm_params),  # type: ignore[union-attr]
                learning_rate * esm_lr_scale,
            )
        self.optimizer = AdamW(
            param_groups,
            weight_decay=weight_decay,
        )
        # Paper: "LR decayed by 0.5 when val loss did not improve for 10 epochs"
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
        )

        self.model.to(self.device)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # JSON log: one dict per epoch, appended as newline-delimited JSON
        self._log_path = self.runs_dir / "train_log.jsonl"

        # State that can be overwritten on resume
        self._start_epoch: int = 1
        self._best_cindex: float = -1.0
        self._epochs_no_improve: int = 0
        self._history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_cindex": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        """Run the training loop from ``_start_epoch`` to ``max_epochs``.

        Returns
        -------
        dict[str, list[float]]
            History dict with keys ``train_loss``, ``val_loss``,
            ``val_cindex``, ``lr``.
        """
        logger.info(
            "Training on %s  |  train=%d batches  val=%d batches  max_epochs=%d",
            self.device, len(self.train_loader), len(self.val_loader), self.max_epochs,
        )

        for epoch in range(self._start_epoch, self.max_epochs + 1):
            t0 = time.perf_counter()
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_loss = self._train_epoch()
            val_loss, val_cindex = self._val_epoch()
            self.scheduler.step(val_loss)  # ReduceLROnPlateau monitors val loss

            elapsed = time.perf_counter() - t0
            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["val_cindex"].append(val_cindex)
            self._history["lr"].append(current_lr)

            logger.info(
                "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_cindex=%.4f  lr=%.2e  (%.1fs)",
                epoch, self.max_epochs,
                train_loss, val_loss, val_cindex, current_lr, elapsed,
            )
            self._write_log(epoch, train_loss, val_loss, val_cindex, current_lr, elapsed)

            # Periodic checkpoint
            if self.checkpoint_every > 0 and epoch % self.checkpoint_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch:04d}.pt", epoch)

            # Best-model checkpoint + early stopping (higher C-index is better)
            if val_cindex > self._best_cindex:
                self._best_cindex = val_cindex
                self._epochs_no_improve = 0
                self._save_checkpoint("best_model.pt", epoch)
                logger.info("  ↑ new best C-index=%.4f", self._best_cindex)
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, self.patience,
                    )
                    break

        return dict(self._history)

    def load_checkpoint(self, path: Path) -> None:
        """Resume training from a saved :class:`CheckpointState`.

        Parameters
        ----------
        path : Path
            Path to the checkpoint ``.pt`` file produced by
            :meth:`_save_checkpoint`.
        """
        ckpt: CheckpointState = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.model_state)
        self.optimizer.load_state_dict(ckpt.optimizer_state)
        self.scheduler.load_state_dict(ckpt.scheduler_state)
        self._start_epoch = ckpt.epoch + 1
        self._best_cindex = ckpt.best_cindex
        self._history = ckpt.history
        # Recompute epochs_no_improve from history
        cindex_hist = self._history.get("val_cindex", [])
        if cindex_hist:
            best_pos = int(np.argmax(cindex_hist))
            self._epochs_no_improve = len(cindex_hist) - 1 - best_pos
        logger.info(
            "Resumed from %s  (epoch %d, best_cindex=%.4f)",
            path, ckpt.epoch, ckpt.best_cindex,
        )

    # ------------------------------------------------------------------
    # Internal: one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        """Run one validation pass.

        Returns
        -------
        val_loss : float
            Mean DeepHit loss over all validation batches.
        val_cindex : float
            Cause-specific C-index for ``cindex_event`` (or 0.5 if no
            uncensored subjects in the validation set).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_times: list[np.ndarray] = []
        all_risks: list[np.ndarray] = []
        all_observed: list[np.ndarray] = []

        for batch in self.val_loader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            n_batches += 1

            # Accumulate predictions for C-index
            donor    = batch["donor_embeddings"].to(self.device)
            recip    = batch["recipient_embeddings"].to(self.device)
            clinical = batch["clinical_features"].to(self.device)
            times    = batch["event_times"]
            types    = batch["event_types"]

            cif = self._compute_cif(donor, recip, clinical)
            # Risk score: CIF of the target event at the median time bin
            median_bin = cif.shape[2] // 2
            risk = cif[:, self.cindex_event - 1, median_bin].cpu().numpy()
            observed = (types == self.cindex_event).numpy()

            all_times.append(times.numpy().astype(float))
            all_risks.append(risk)
            all_observed.append(observed)

        val_loss = total_loss / max(n_batches, 1)

        times_arr    = np.concatenate(all_times)
        risks_arr    = np.concatenate(all_risks)
        observed_arr = np.concatenate(all_observed).astype(bool)

        if observed_arr.sum() < 2:
            val_cindex = 0.5  # not enough uncensored subjects to compute
        else:
            val_cindex = concordance_index(times_arr, risks_arr, observed_arr)
            if np.isnan(val_cindex):
                val_cindex = 0.5

        return val_loss, val_cindex

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        donor    = batch["donor_embeddings"].to(self.device)
        recip    = batch["recipient_embeddings"].to(self.device)
        clinical = batch["clinical_features"].to(self.device)
        times    = batch["event_times"].to(self.device)
        types    = batch["event_types"].to(self.device)

        logits = self.model(donor, recip, clinical)
        return deephit_loss(logits, times, types, alpha=self.alpha, sigma=self.sigma)

    def _compute_cif(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """Return CIF tensor (batch, num_events, time_bins) from the model."""
        import torch.nn.functional as F

        out = self.model(donor, recip, clinical)
        batch, num_events, time_bins = out.shape
        if self.survival_type == "deephit":
            joint = F.softmax(out.view(batch, -1), dim=-1).view(batch, num_events, time_bins)
            return torch.cumsum(joint, dim=2)
        else:
            from capa.model.survival import hazards_to_cif
            return hazards_to_cif(out)

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        ckpt = CheckpointState(
            epoch=epoch,
            best_cindex=self._best_cindex,
            history=dict(self._history),
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
        )
        path = self.runs_dir / filename
        torch.save(ckpt, path)
        logger.debug("Checkpoint saved → %s", path)

    def _write_log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_cindex: float,
        lr: float,
        elapsed: float,
    ) -> None:
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_cindex": round(val_cindex, 6),
            "lr": lr,
            "elapsed_s": round(elapsed, 3),
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
