"""Training loop with early stopping and LR scheduling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from capa.model.losses import deephit_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Train a CAPA model with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
        The CAPA model to train.
    train_loader : DataLoader[Any]
        DataLoader for training data.
    val_loader : DataLoader[Any]
        DataLoader for validation data.
    learning_rate : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularisation coefficient.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience (epochs without val improvement).
    alpha : float
        DeepHit loss ranking weight.
    sigma : float
        DeepHit ranking loss bandwidth.
    runs_dir : Path
        Directory to save model checkpoints.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        alpha: float = 0.5,
        sigma: float = 0.1,
        runs_dir: Path = Path("runs"),
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.alpha = alpha
        self.sigma = sigma
        self.runs_dir = runs_dir
        self.device = torch.device(device)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=patience // 2)

        self.model.to(self.device)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, list[float]]:
        """Run the training loop.

        Returns
        -------
        dict[str, list[float]]
            History dict with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                epoch,
                self.max_epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint("best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in self.val_loader:
            loss = self._compute_loss(batch)
            total += loss.item()
        return total / max(len(self.val_loader), 1)

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        donor = batch["donor_embeddings"].to(self.device)
        recipient = batch["recipient_embeddings"].to(self.device)
        clinical = batch["clinical_features"].to(self.device)
        event_times = batch["event_times"].to(self.device)
        event_types = batch["event_types"].to(self.device)

        logits = self.model(donor, recipient, clinical)
        return deephit_loss(logits, event_times, event_types, self.alpha, self.sigma)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.runs_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info("Saved checkpoint → %s", path)
