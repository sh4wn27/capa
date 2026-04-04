"""Cross-attention weight extraction and publication-quality visualisation.

Overview
--------
The CAPA interaction network stores attention weights after every forward pass
in :attr:`~capa.model.interaction.CrossAttentionInteraction.last_attn_weights`.
This module provides three layers of functionality built on top of those raw
tensors:

1. **Extraction** — :func:`extract_attention_weights` runs a forward pass and
   returns structured :class:`AttentionWeightSet` objects (one per sample in
   the batch), with helpers for selecting layers and finding top pairs.

2. **Aggregation** — :func:`aggregate_population_weights` accumulates weights
   across many patients into a :class:`PopulationWeights` object with per-cell
   mean and standard deviation.

3. **Visualisation** — four plot functions generate publication-quality figures:

   * :func:`plot_attention_heatmap` — single patient, one direction.
   * :func:`plot_both_directions` — single patient, D→R and R→D side-by-side.
   * :func:`plot_population_heatmap` — population mean ± std, one direction.
   * :func:`plot_population_both_directions` — population, both directions.

   All functions accept an optional ``ax`` argument for embedding in larger
   figures.  :func:`save_figure` writes PDF (vector, 300 DPI) and PNG.

Typical single-patient usage::

    weights = extract_attention_weights(
        model, donor_emb, recip_emb, clinical, loci=loci
    )
    fig = plot_both_directions(weights[0], top_k=3)
    save_figure(fig, out_dir / "patient_001")

Typical population-level usage::

    from torch.utils.data import DataLoader
    all_weights = collect_population_weights(model, dataloader, loci)
    pop = aggregate_population_weights(all_weights)
    fig = plot_population_both_directions(pop, top_k=3)
    save_figure(fig, out_dir / "population")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Figure styling (applied once; callers may override)
# ---------------------------------------------------------------------------

_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,   # screen preview; save_figure uses 300
    "savefig.dpi": 300,
}

# Direction colour maps and labels
_D2R_CMAP = "Blues"
_R2D_CMAP = "Oranges"
_D2R_LABEL = "Donor → Recipient"
_R2D_LABEL = "Recipient → Donor"

# Top-pair highlight styling
_TOP_EDGE_COLOUR = "#d62728"    # matplotlib tab:red
_TOP_LINEWIDTH   = 2.5
_TOP_STAR_COLOUR = "#d62728"
_STAR_MARKERS    = ["★", "✦", "●"]   # rank 1, 2, 3


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TopPair:
    """One donor–recipient locus pair identified as high-attention.

    Attributes
    ----------
    rank : int
        1 = highest attention in the matrix.
    row : int
        Donor (row) locus index.
    col : int
        Recipient (column) locus index.
    donor_locus : str
    recipient_locus : str
    weight : float
    """

    rank: int
    row: int
    col: int
    donor_locus: str
    recipient_locus: str
    weight: float

    def __repr__(self) -> str:
        return (
            f"TopPair(rank={self.rank}, "
            f"{self.donor_locus}→{self.recipient_locus}, "
            f"w={self.weight:.4f})"
        )


@dataclass
class AttentionWeightSet:
    """Attention weights extracted from one forward pass for a single subject.

    Attributes
    ----------
    donor_to_recip : list[F32]
        Per-layer attention, each ``(n_loci_donor, n_loci_recip)``.
    recip_to_donor : list[F32]
        Per-layer attention, each ``(n_loci_recip, n_loci_donor)``.
    loci : list[str]
        HLA locus names in order (e.g. ``["A","B","C","DRB1","DQB1"]``).
    patient_id : str
        Optional identifier used in figure titles.
    """

    donor_to_recip: list[F32]
    recip_to_donor: list[F32]
    loci: list[str]
    patient_id: str = ""

    # ------------------------------------------------------------------
    # Layer access
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of cross-attention layers."""
        return len(self.donor_to_recip)

    def get_layer(self, layer: int = -1) -> tuple[F32, F32]:
        """Return ``(d2r, r2d)`` for one layer index (supports negative)."""
        return self.donor_to_recip[layer], self.recip_to_donor[layer]

    def mean_across_layers(self) -> tuple[F32, F32]:
        """Return mean attention matrices averaged across all layers."""
        d2r = np.mean(np.stack(self.donor_to_recip, axis=0), axis=0)
        r2d = np.mean(np.stack(self.recip_to_donor, axis=0), axis=0)
        return d2r.astype(np.float32), r2d.astype(np.float32)

    # ------------------------------------------------------------------
    # Top-pair identification
    # ------------------------------------------------------------------

    def top_k_pairs(
        self,
        weights: F32,
        k: int = 3,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
    ) -> list[TopPair]:
        """Return the *k* highest-attention locus pairs from *weights*.

        Parameters
        ----------
        weights : F32, shape (n_rows, n_cols)
            Attention weight matrix.
        k : int
            Number of top pairs to return.
        row_labels : list[str] | None
            Row label names (donor loci).  Defaults to ``self.loci``.
        col_labels : list[str] | None
            Column label names (recipient loci).  Defaults to ``self.loci``.

        Returns
        -------
        list[TopPair]
            Sorted descending by weight; at most *k* entries.
        """
        rows = row_labels or self.loci
        cols = col_labels or self.loci
        flat_idx = np.argsort(weights.ravel())[::-1][:k]
        pairs = []
        for rank, flat in enumerate(flat_idx, start=1):
            r, c = divmod(int(flat), weights.shape[1])
            pairs.append(TopPair(
                rank=rank,
                row=r, col=c,
                donor_locus=rows[r] if r < len(rows) else str(r),
                recipient_locus=cols[c] if c < len(cols) else str(c),
                weight=float(weights[r, c]),
            ))
        return pairs

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation."""
        d2r_last, r2d_last = self.get_layer(-1)
        d2r_mean, r2d_mean = self.mean_across_layers()
        return {
            "patient_id": self.patient_id,
            "loci": self.loci,
            "n_layers": self.n_layers,
            "last_layer_d2r": d2r_last.tolist(),
            "last_layer_r2d": r2d_last.tolist(),
            "mean_d2r": d2r_mean.tolist(),
            "mean_r2d": r2d_mean.tolist(),
        }


@dataclass
class PopulationWeights:
    """Mean and std of attention weights across a population of subjects.

    Attributes
    ----------
    mean_d2r : F32, shape (n_loci, n_loci)
    std_d2r  : F32, shape (n_loci, n_loci)
    mean_r2d : F32, shape (n_loci, n_loci)
    std_r2d  : F32, shape (n_loci, n_loci)
    loci : list[str]
    n_subjects : int
    """

    mean_d2r: F32
    std_d2r: F32
    mean_r2d: F32
    std_r2d: F32
    loci: list[str]
    n_subjects: int

    def top_k_pairs(
        self,
        direction: str = "d2r",
        k: int = 3,
    ) -> list[TopPair]:
        """Return top-*k* pairs by population *mean* weight.

        Parameters
        ----------
        direction : {"d2r", "r2d"}
        k : int
        """
        weights = self.mean_d2r if direction == "d2r" else self.mean_r2d
        flat_idx = np.argsort(weights.ravel())[::-1][:k]
        pairs = []
        for rank, flat in enumerate(flat_idx, start=1):
            r, c = divmod(int(flat), weights.shape[1])
            pairs.append(TopPair(
                rank=rank,
                row=r, col=c,
                donor_locus=self.loci[r] if r < len(self.loci) else str(r),
                recipient_locus=self.loci[c] if c < len(self.loci) else str(c),
                weight=float(weights[r, c]),
            ))
        return pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "loci": self.loci,
            "mean_d2r": self.mean_d2r.tolist(),
            "std_d2r": self.std_d2r.tolist(),
            "mean_r2d": self.mean_r2d.tolist(),
            "std_r2d": self.std_r2d.tolist(),
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    donor_embeddings: torch.Tensor,
    recipient_embeddings: torch.Tensor,
    clinical_features: torch.Tensor,
    *,
    loci: list[str] | None = None,
    device: str | torch.device | None = None,
) -> list[AttentionWeightSet]:
    """Run a forward pass and return per-sample attention weight sets.

    Parameters
    ----------
    model : nn.Module
        Trained :class:`~capa.model.capa_model.CAPAModel` (must expose a
        ``.interaction.last_attn_weights`` attribute).
    donor_embeddings : Tensor
        Shape ``(batch, n_loci, embedding_dim)``.
    recipient_embeddings : Tensor
        Same shape as *donor_embeddings*.
    clinical_features : Tensor
        Shape ``(batch, clinical_dim)``.
    loci : list[str] | None
        HLA locus names.  Inferred from the batch's second dimension if not
        provided (labelled ``"locus_0"``, ``"locus_1"``, …).
    device : str | torch.device | None
        Inference device.  Defaults to the device of the model's parameters.

    Returns
    -------
    list[AttentionWeightSet]
        One entry per subject in the batch.

    Raises
    ------
    ValueError
        If the model does not expose attention weights (e.g. uses
        :class:`~capa.model.interaction.DiffMLPInteraction`).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    donor_embeddings    = donor_embeddings.to(device)
    recipient_embeddings = recipient_embeddings.to(device)
    clinical_features   = clinical_features.to(device)

    batch_size = donor_embeddings.shape[0]
    n_loci     = donor_embeddings.shape[1]

    if loci is None:
        loci = [f"locus_{i}" for i in range(n_loci)]

    was_training = model.training
    model.eval()

    # Forward pass — weights stored in model.interaction.last_attn_weights
    model(donor_embeddings, recipient_embeddings, clinical_features)

    # Retrieve via public API
    attn = getattr(model, "get_attention_weights", lambda: None)()
    if attn is None:
        # Fall back to direct attribute (handles CAPAOneHotModel etc.)
        interaction = getattr(model, "interaction", None)
        if interaction is not None:
            attn = getattr(interaction, "last_attn_weights", None)

    if was_training:
        model.train()

    if attn is None:
        raise ValueError(
            "Model does not expose attention weights.  "
            "Ensure the model uses CrossAttentionInteraction and that "
            "a forward pass has been performed."
        )

    # attn.donor_to_recip: list[Tensor(batch, n_q, n_kv)]
    results: list[AttentionWeightSet] = []
    for b in range(batch_size):
        d2r_layers = [
            w[b].cpu().numpy().astype(np.float32)
            for w in attn.donor_to_recip
        ]
        r2d_layers = [
            w[b].cpu().numpy().astype(np.float32)
            for w in attn.recip_to_donor
        ]
        results.append(AttentionWeightSet(
            donor_to_recip=d2r_layers,
            recip_to_donor=r2d_layers,
            loci=list(loci),
        ))

    return results


def collect_population_weights(
    model: nn.Module,
    data_iterator: Any,
    loci: list[str],
    *,
    layer: int = -1,
    max_subjects: int = 500,
    device: str | torch.device | None = None,
) -> list[AttentionWeightSet]:
    """Collect per-subject attention weights over an entire dataset.

    Parameters
    ----------
    model : nn.Module
        Trained CAPA model.
    data_iterator : iterable of dict
        Each batch must contain ``donor_embeddings``, ``recipient_embeddings``,
        and ``clinical_features`` tensors (standard DataLoader format).
    loci : list[str]
        HLA locus labels.
    layer : int
        Which layer's weights to use for the per-subject sets.  -1 = last.
    max_subjects : int
        Stop after collecting this many subjects (for large datasets).
    device : str | torch.device | None

    Returns
    -------
    list[AttentionWeightSet]
        One per subject, up to *max_subjects*.
    """
    all_sets: list[AttentionWeightSet] = []
    n_collected = 0

    for batch in data_iterator:
        donor    = batch["donor_embeddings"]
        recip    = batch["recipient_embeddings"]
        clinical = batch["clinical_features"]

        weight_sets = extract_attention_weights(
            model, donor, recip, clinical,
            loci=loci, device=device,
        )
        all_sets.extend(weight_sets)
        n_collected += len(weight_sets)

        if n_collected >= max_subjects:
            break

    logger.info("Collected attention weights from %d subjects", len(all_sets))
    return all_sets[:max_subjects]


def aggregate_population_weights(
    weight_sets: list[AttentionWeightSet],
    layer: int = -1,
) -> PopulationWeights:
    """Compute mean and std of attention weights across subjects.

    Parameters
    ----------
    weight_sets : list[AttentionWeightSet]
        Per-subject weight sets from :func:`collect_population_weights`.
    layer : int
        Which layer to aggregate.  -1 = last layer.

    Returns
    -------
    PopulationWeights
    """
    if not weight_sets:
        raise ValueError("weight_sets must be non-empty")

    loci = weight_sets[0].loci

    d2r_stack = np.stack([ws.donor_to_recip[layer]    for ws in weight_sets], axis=0)
    r2d_stack = np.stack([ws.recip_to_donor[layer]    for ws in weight_sets], axis=0)

    return PopulationWeights(
        mean_d2r=d2r_stack.mean(axis=0).astype(np.float32),
        std_d2r=d2r_stack.std(axis=0).astype(np.float32),
        mean_r2d=r2d_stack.mean(axis=0).astype(np.float32),
        std_r2d=r2d_stack.std(axis=0).astype(np.float32),
        loci=loci,
        n_subjects=len(weight_sets),
    )


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply publication-quality rcParams without mutating the caller's state."""
    matplotlib.rcParams.update(_STYLE)


def _annotate_cells(
    ax: plt.Axes,
    weights: F32,
    fmt: str = ".3f",
    fontsize: int = 8,
    threshold: float | None = None,
) -> None:
    """Write weight values inside each heatmap cell.

    Parameters
    ----------
    ax : plt.Axes
    weights : F32, shape (n_rows, n_cols)
    fmt : str
        Python format spec for the value, e.g. ``".3f"``.
    fontsize : int
    threshold : float | None
        If given, cells above this value use white text; others use dark grey.
    """
    vmax = weights.max()
    thresh = threshold if threshold is not None else vmax * 0.6
    for r in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            val = weights[r, c]
            colour = "white" if val >= thresh else "#333333"
            ax.text(
                c, r, f"{val:{fmt}}",
                ha="center", va="center",
                fontsize=fontsize, color=colour,
                fontweight="normal",
            )


def _highlight_top_pairs(
    ax: plt.Axes,
    top_pairs: list[TopPair],
    annotate: bool = True,
    fontsize: int = 8,
) -> None:
    """Draw a red box around each top-pair cell and optionally annotate rank.

    Parameters
    ----------
    ax : plt.Axes
    top_pairs : list[TopPair]
    annotate : bool
        If True, draw a small rank label (★, ✦, ●) in the upper-right corner.
    fontsize : int
    """
    for tp in top_pairs:
        # Thick red rectangle (patch positioned in data coords)
        rect = mpatches.FancyBboxPatch(
            (tp.col - 0.48, tp.row - 0.48),
            0.96, 0.96,
            boxstyle="square,pad=0",
            linewidth=_TOP_LINEWIDTH,
            edgecolor=_TOP_EDGE_COLOUR,
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

        if annotate and tp.rank <= len(_STAR_MARKERS):
            marker = _STAR_MARKERS[tp.rank - 1]
            ax.text(
                tp.col + 0.40, tp.row - 0.40,
                marker,
                ha="right", va="top",
                fontsize=fontsize + 1,
                color=_TOP_EDGE_COLOUR,
                zorder=6,
            )


def _setup_heatmap_axes(
    ax: plt.Axes,
    weights: F32,
    row_labels: list[str],
    col_labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
    row_axis_label: str,
    col_axis_label: str,
) -> Any:
    """Draw the heatmap image, axis labels and tick marks; return the image."""
    im = ax.imshow(
        weights,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel(col_axis_label, fontsize=11)
    ax.set_ylabel(row_axis_label, fontsize=11)

    # Light grid lines between cells
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, zorder=3)
    ax.tick_params(which="minor", length=0)

    return im


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    weights: F32,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str = "Cross-Attention",
    direction_label: str = _D2R_LABEL,
    cmap: str = _D2R_CMAP,
    top_k: int = 3,
    patient_id: str | None = None,
    loci: list[str] | None = None,
    annotate_values: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot one cross-attention weight matrix as a publication-quality heatmap.

    Parameters
    ----------
    weights : F32, shape (n_rows, n_cols)
        Attention weight matrix.
    row_labels : list[str]
        Labels for the rows (donor loci).
    col_labels : list[str]
        Labels for the columns (recipient loci).
    title : str
        Primary title text.
    direction_label : str
        Sub-title / axis description (e.g. "Donor → Recipient").
    cmap : str
        Matplotlib colormap name.
    top_k : int
        Number of top-attention pairs to highlight with a red box.
        Set to 0 to disable.
    patient_id : str | None
        If provided, appended to the title.
    loci : list[str] | None
        Locus names used for :class:`~TopPair` creation (defaults to
        *row_labels*).
    annotate_values : bool
        If True, write weight values inside each cell.
    figsize : tuple[float, float] | None
        Override figure size ``(width_inches, height_inches)``.
    ax : plt.Axes | None
        Existing axes to draw on.  A new figure is created if ``None``.

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    n_rows, n_cols = len(row_labels), len(col_labels)
    if figsize is None:
        w = max(4.0, n_cols * 1.1 + 1.5)
        h = max(3.5, n_rows * 1.0 + 1.2)
        figsize = (w, h)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    assert ax is not None

    vmax = float(weights.max()) if weights.max() > 0 else 1.0
    im = _setup_heatmap_axes(
        ax, weights,
        row_labels=row_labels,
        col_labels=col_labels,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        row_axis_label="Donor locus",
        col_axis_label="Recipient locus",
    )

    if annotate_values:
        _annotate_cells(ax, weights, fmt=".3f", fontsize=8, threshold=vmax * 0.6)

    # Top-k highlighting
    if top_k > 0:
        _loci = loci or row_labels
        tmp_set = AttentionWeightSet(
            donor_to_recip=[weights], recip_to_donor=[weights], loci=_loci
        )
        top_pairs = tmp_set.top_k_pairs(
            weights, k=min(top_k, n_rows * n_cols),
            row_labels=row_labels, col_labels=col_labels,
        )
        _highlight_top_pairs(ax, top_pairs, annotate=True, fontsize=8)

    # Colorbar
    if own_fig:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Attention weight", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    full_title = title
    if patient_id:
        full_title = f"{title}  |  {patient_id}"
    ax.set_title(f"{full_title}\n{direction_label}", fontsize=12, pad=8)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_both_directions(
    weight_set: AttentionWeightSet,
    *,
    layer: int = -1,
    top_k: int = 3,
    title_prefix: str = "",
    annotate_values: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Side-by-side heatmaps for both attention directions (D→R and R→D).

    Parameters
    ----------
    weight_set : AttentionWeightSet
        Per-subject attention weights.
    layer : int
        Which layer to visualise.  -1 = last.
    top_k : int
        Number of top pairs to highlight per direction.
    title_prefix : str
        Prepended to the figure suptitle.
    annotate_values : bool
    figsize : tuple[float, float] | None
        Defaults to ``(2 * n_loci + 3, n_loci + 2.5)``.

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    loci = weight_set.loci
    n = len(loci)
    d2r, r2d = weight_set.get_layer(layer)

    if figsize is None:
        w = max(9.0, n * 2.2 + 3.0)
        h = max(4.0, n * 1.0 + 2.5)
        figsize = (w, h)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Shared colour scale across both directions for fair comparison
    vmax = max(float(d2r.max()), float(r2d.max()), 1e-9)

    for ax_i, (weights, cmap, dir_label) in enumerate([
        (d2r, _D2R_CMAP, _D2R_LABEL),
        (r2d, _R2D_CMAP, _R2D_LABEL),
    ]):
        ax = axes[ax_i]
        row_lbl = loci
        col_lbl = loci
        im = _setup_heatmap_axes(
            ax, weights,
            row_labels=row_lbl,
            col_labels=col_lbl,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            row_axis_label="Donor locus" if ax_i == 0 else "Recipient locus",
            col_axis_label="Recipient locus" if ax_i == 0 else "Donor locus",
        )
        if annotate_values:
            _annotate_cells(ax, weights, fmt=".3f", fontsize=8, threshold=vmax * 0.6)

        if top_k > 0:
            tmp_set = AttentionWeightSet(
                donor_to_recip=[weights], recip_to_donor=[weights], loci=loci
            )
            top_pairs = tmp_set.top_k_pairs(
                weights, k=min(top_k, n * n),
                row_labels=row_lbl, col_labels=col_lbl,
            )
            _highlight_top_pairs(ax, top_pairs, annotate=True, fontsize=8)

        ax.set_title(dir_label, fontsize=11)

        # Individual colorbars
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Attention weight", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    pid = weight_set.patient_id
    layer_str = f"layer {weight_set.n_layers + layer if layer < 0 else layer + 1}"
    sup = f"{title_prefix}Cross-Attention Weights ({layer_str})"
    if pid:
        sup = f"{sup}  |  Patient: {pid}"
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_population_heatmap(
    pop: PopulationWeights,
    *,
    direction: str = "d2r",
    title: str = "Population Cross-Attention",
    top_k: int = 3,
    annotate_values: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of population-mean attention weights with std annotations.

    Parameters
    ----------
    pop : PopulationWeights
    direction : {"d2r", "r2d"}
    title : str
    top_k : int
    annotate_values : bool
        If True, annotate each cell with ``mean ± std``.
    figsize : tuple | None
    ax : plt.Axes | None

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    loci = pop.loci
    n = len(loci)

    if direction == "d2r":
        mean_w = pop.mean_d2r
        std_w  = pop.std_d2r
        cmap   = _D2R_CMAP
        dir_label = _D2R_LABEL
    else:
        mean_w = pop.mean_r2d
        std_w  = pop.std_r2d
        cmap   = _R2D_CMAP
        dir_label = _R2D_LABEL

    if figsize is None:
        w = max(4.5, n * 1.2 + 2.0)
        h = max(4.0, n * 1.1 + 1.5)
        figsize = (w, h)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    assert ax is not None

    vmax = float(mean_w.max()) if mean_w.max() > 0 else 1.0
    im = _setup_heatmap_axes(
        ax, mean_w,
        row_labels=loci,
        col_labels=loci,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        row_axis_label="Donor locus",
        col_axis_label="Recipient locus",
    )

    # Annotate each cell: "μ\n±σ"
    if annotate_values:
        for r in range(n):
            for c in range(n):
                mu  = mean_w[r, c]
                sig = std_w[r, c]
                colour = "white" if mu >= vmax * 0.6 else "#333333"
                ax.text(
                    c, r - 0.1,
                    f"{mu:.3f}",
                    ha="center", va="center",
                    fontsize=7.5, color=colour, fontweight="bold",
                )
                ax.text(
                    c, r + 0.25,
                    f"±{sig:.3f}",
                    ha="center", va="center",
                    fontsize=6.5, color=colour,
                )

    # Top-k highlighting
    if top_k > 0:
        top_pairs = pop.top_k_pairs(direction=direction, k=min(top_k, n * n))
        _highlight_top_pairs(ax, top_pairs, annotate=True, fontsize=8)

    if own_fig:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Mean attention weight", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"{title}  (n={pop.n_subjects})\n{dir_label}",
        fontsize=12, pad=8,
    )

    if own_fig:
        fig.tight_layout()
    return fig


def plot_population_both_directions(
    pop: PopulationWeights,
    *,
    top_k: int = 3,
    title_prefix: str = "",
    annotate_values: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Side-by-side population-mean heatmaps for both directions.

    Parameters
    ----------
    pop : PopulationWeights
    top_k : int
    title_prefix : str
    annotate_values : bool
    figsize : tuple | None

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    n = len(pop.loci)
    if figsize is None:
        w = max(9.0, n * 2.3 + 3.5)
        h = max(4.0, n * 1.15 + 2.5)
        figsize = (w, h)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax_i, direction in enumerate(["d2r", "r2d"]):
        plot_population_heatmap(
            pop,
            direction=direction,
            top_k=top_k,
            annotate_values=annotate_values,
            ax=axes[ax_i],
        )

    sup = f"{title_prefix}Population Cross-Attention (n={pop.n_subjects})"
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    path: Path | str,
    *,
    dpi: int = 300,
    formats: list[str] | None = None,
    bbox_inches: str = "tight",
) -> list[Path]:
    """Save a figure to one or more formats.

    Parameters
    ----------
    fig : plt.Figure
    path : Path | str
        Base path *without* extension, e.g. ``runs/patient_001``.
        The extension is appended per format.
    dpi : int
        Dots per inch for raster formats (PNG, TIFF).  PDF is vector-only
        so DPI only affects embedded bitmap elements.
    formats : list[str] | None
        File extensions to write.  Defaults to ``["pdf", "png"]``.
    bbox_inches : str
        Passed to ``savefig``.  ``"tight"`` trims whitespace.

    Returns
    -------
    list[Path]
        Absolute paths of the saved files.
    """
    if formats is None:
        formats = ["pdf", "png"]

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches=bbox_inches,
            format=fmt,
        )
        logger.debug("Saved figure → %s", out)
        saved.append(out)

    return saved


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------

def generate_patient_map(
    model: nn.Module,
    donor_embeddings: torch.Tensor,
    recipient_embeddings: torch.Tensor,
    clinical_features: torch.Tensor,
    loci: list[str],
    *,
    patient_id: str = "patient",
    out_dir: Path | str | None = None,
    layer: int = -1,
    top_k: int = 3,
    dpi: int = 300,
    device: str | torch.device | None = None,
) -> dict[str, plt.Figure]:
    """Extract attention weights and generate all figures for one patient.

    Produces three figures:
    * ``"d2r"`` — donor→recipient heatmap.
    * ``"r2d"`` — recipient→donor heatmap.
    * ``"both"`` — side-by-side composite figure.

    Parameters
    ----------
    model : nn.Module
        Trained CAPA model.
    donor_embeddings : Tensor
        Shape ``(1, n_loci, embedding_dim)`` — single patient.
    recipient_embeddings : Tensor
        Same shape.
    clinical_features : Tensor
        Shape ``(1, clinical_dim)``.
    loci : list[str]
        HLA locus names.
    patient_id : str
        Used in figure titles and output file names.
    out_dir : Path | str | None
        If given, figures are saved as ``{out_dir}/{patient_id}_{key}.{fmt}``.
    layer : int
        Which attention layer to visualise (default: last).
    top_k : int
        Number of top pairs to highlight.
    dpi : int
        Save resolution.
    device : str | torch.device | None

    Returns
    -------
    dict[str, plt.Figure]
        Keys: ``"d2r"``, ``"r2d"``, ``"both"``.
    """
    # Extract
    weight_sets = extract_attention_weights(
        model, donor_embeddings, recipient_embeddings, clinical_features,
        loci=loci, device=device,
    )
    ws = weight_sets[0]
    ws.patient_id = patient_id

    d2r_weights, r2d_weights = ws.get_layer(layer)

    # Individual direction figures
    fig_d2r = plot_attention_heatmap(
        d2r_weights, loci, loci,
        title="Cross-Attention",
        direction_label=_D2R_LABEL,
        cmap=_D2R_CMAP,
        top_k=top_k,
        patient_id=patient_id,
        loci=loci,
    )
    fig_r2d = plot_attention_heatmap(
        r2d_weights, loci, loci,
        title="Cross-Attention",
        direction_label=_R2D_LABEL,
        cmap=_R2D_CMAP,
        top_k=top_k,
        patient_id=patient_id,
        loci=loci,
    )

    # Composite figure
    fig_both = plot_both_directions(
        ws, layer=layer, top_k=top_k,
        title_prefix="",
    )

    figures = {"d2r": fig_d2r, "r2d": fig_r2d, "both": fig_both}

    # Save
    if out_dir is not None:
        out_dir = Path(out_dir)
        for key, fig in figures.items():
            save_figure(
                fig,
                out_dir / f"{patient_id}_{key}",
                dpi=dpi,
            )
        logger.info("Patient figures saved to %s", out_dir)

    return figures


def generate_population_map(
    model: nn.Module,
    data_iterator: Any,
    loci: list[str],
    *,
    out_dir: Path | str | None = None,
    layer: int = -1,
    top_k: int = 3,
    max_subjects: int = 500,
    dpi: int = 300,
    device: str | torch.device | None = None,
) -> dict[str, plt.Figure]:
    """Generate population-level attention maps averaged across all subjects.

    Produces two figures:
    * ``"population_d2r"`` — donor→recipient population mean.
    * ``"population_r2d"`` — recipient→donor population mean.
    * ``"population_both"`` — side-by-side composite.

    Parameters
    ----------
    model : nn.Module
    data_iterator : iterable of dict
        Standard-format DataLoader batches.
    loci : list[str]
    out_dir : Path | str | None
    layer : int
    top_k : int
    max_subjects : int
    dpi : int
    device : str | torch.device | None

    Returns
    -------
    dict[str, plt.Figure]
    """
    weight_sets = collect_population_weights(
        model, data_iterator, loci,
        layer=layer, max_subjects=max_subjects, device=device,
    )

    if not weight_sets:
        raise ValueError("No attention weights collected — empty data_iterator?")

    pop = aggregate_population_weights(weight_sets, layer=layer)

    fig_d2r  = plot_population_heatmap(pop, direction="d2r", top_k=top_k)
    fig_r2d  = plot_population_heatmap(pop, direction="r2d", top_k=top_k)
    fig_both = plot_population_both_directions(pop, top_k=top_k)

    figures = {
        "population_d2r":  fig_d2r,
        "population_r2d":  fig_r2d,
        "population_both": fig_both,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        for key, fig in figures.items():
            save_figure(fig, out_dir / key, dpi=dpi)
        logger.info("Population figures saved to %s", out_dir)

    return figures
