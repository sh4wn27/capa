"""Extract and visualise cross-attention weights from the interaction network."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_attention_weights(
    model: nn.Module,
    donor_embeddings: torch.Tensor,
    recipient_embeddings: torch.Tensor,
) -> list[npt.NDArray[np.float32]]:
    """Extract cross-attention weight matrices from the interaction network.

    Registers forward hooks on all ``MultiheadAttention`` modules in the
    interaction network and runs a single forward pass to collect weights.

    Parameters
    ----------
    model : nn.Module
        Trained CAPA model with a ``.interaction`` attribute.
    donor_embeddings : torch.Tensor
        Shape ``(1, n_loci, embedding_dim)`` — single sample.
    recipient_embeddings : torch.Tensor
        Shape ``(1, n_loci, embedding_dim)`` — single sample.

    Returns
    -------
    list[npt.NDArray[np.float32]]
        List of attention weight arrays, one per cross-attention layer.
        Each array has shape ``(n_loci, n_loci)``.
    """
    weights: list[npt.NDArray[np.float32]] = []
    hooks: list[Any] = []

    def _hook(_module: nn.Module, _inp: Any, output: Any) -> None:
        # MultiheadAttention returns (output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            w = output[1].detach().cpu().squeeze(0).numpy().astype(np.float32)
            weights.append(w)

    for layer in model.interaction.donor_to_recip:  # type: ignore[union-attr]
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        model(donor_embeddings, recipient_embeddings, torch.zeros(1, 1))  # dummy clinical

    for h in hooks:
        h.remove()

    return weights


def plot_attention_heatmap(
    weights: npt.NDArray[np.float32],
    donor_labels: list[str],
    recipient_labels: list[str],
    title: str = "Donor → Recipient Cross-Attention",
) -> plt.Figure:
    """Plot a cross-attention weight matrix as a heatmap.

    Parameters
    ----------
    weights : npt.NDArray[np.float32]
        Attention weight matrix of shape ``(n_loci, n_loci)``.
    donor_labels : list[str]
        Labels for donor allele loci (rows).
    recipient_labels : list[str]
        Labels for recipient allele loci (columns).
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(len(recipient_labels) + 1, len(donor_labels) + 1))
    im = ax.imshow(weights, cmap="Blues", vmin=0.0, vmax=weights.max())
    ax.set_xticks(np.arange(len(recipient_labels)))
    ax.set_yticks(np.arange(len(donor_labels)))
    ax.set_xticklabels(recipient_labels, rotation=45, ha="right")
    ax.set_yticklabels(donor_labels)
    ax.set_xlabel("Recipient loci")
    ax.set_ylabel("Donor loci")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig
