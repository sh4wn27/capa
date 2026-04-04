"""SHAP-based explanations for clinical covariates in CAPA.

Overview
--------
This module provides three layers of functionality:

1. **Model wrappers** — build callable predict functions and differentiable
   PyTorch wrappers that hold HLA mismatch features fixed and expose only
   the clinical covariate pathway for attribution:

   * :func:`build_clinical_predict_fn` — callable for KernelExplainer;
     works with all 8 clinical features (4 continuous + 4 categorical).
   * :class:`ClinicalDeepWrapper` — differentiable ``nn.Module`` for
     DeepExplainer; restricted to the 4 continuous features (integer
     category indices are non-differentiable).

2. **Computation** — run SHAP and package results:

   * :func:`compute_shap_values` — model-agnostic KernelExplainer.
   * :func:`compute_shap_values_deep` — gradient-based DeepExplainer.
   * :func:`build_explanation` — package raw SHAP arrays into a
     :class:`SHAPExplanation` dataclass with feature names and values.

3. **Visualisation** — three publication-quality plot functions:

   * :func:`plot_beeswarm` — population summary; one dot per patient,
     coloured by feature value, features sorted by mean |SHAP|.
   * :func:`plot_waterfall` — individual patient decomposition; horizontal
     bars from expected value to final prediction, labelled with feature
     values.
   * :func:`plot_feature_importance` — mean |SHAP| bar chart for inclusion
     in papers.

   :func:`save_figure` writes PDF (vector) and PNG (300 DPI) to disk.

Feature matrix layout
---------------------
All high-level functions consume an ``(n_samples, 8)`` float32 numpy array:

+-------+-----------------------+------------------+
| col   | feature               | units / encoding |
+=======+=======================+==================+
| 0     | age_recipient         | raw years        |
| 1     | age_donor             | raw years        |
| 2     | cd34_dose             | ×10⁶/kg raw      |
| 3     | sex_mismatch          | 0 / 1            |
| 4     | disease               | int index        |
| 5     | conditioning          | int index        |
| 6     | donor_type            | int index        |
| 7     | stem_cell_source      | int index        |
+-------+-----------------------+------------------+

Use :func:`clinical_dict_to_row` / :func:`clinical_dicts_to_matrix` to
convert raw feature dicts to this matrix.

Typical KernelExplainer usage::

    from capa.interpret.shap_explain import (
        build_clinical_predict_fn,
        clinical_dicts_to_matrix,
        compute_shap_values,
        build_explanation,
        generate_shap_report,
    )

    predict_fn = build_clinical_predict_fn(model, donor_emb, recip_emb, event_idx=0)
    X_bg   = clinical_dicts_to_matrix(background_records)
    X_test = clinical_dicts_to_matrix(test_records)
    shap_vals = compute_shap_values(predict_fn, X_bg, X_test)
    expl = build_explanation(shap_vals, predict_fn(X_bg).mean(), X_test,
                             event_name="GvHD")
    fig_bee = plot_beeswarm(expl)
    fig_wf  = plot_waterfall(expl, sample_idx=0)
    save_figure(fig_bee, out_dir / "shap_beeswarm")

Typical DeepExplainer usage (continuous features only)::

    from capa.interpret.shap_explain import ClinicalDeepWrapper, compute_shap_values_deep

    wrapper, bg_tensor = ClinicalDeepWrapper.from_model(
        model, donor_emb, recip_emb, background_matrix, event_idx=0
    )
    shap_vals_cont, exp_val = compute_shap_values_deep(wrapper, bg_tensor, test_cont_tensor)
    expl = build_explanation(
        shap_vals_cont, exp_val, test_matrix[:, :4],
        feature_names=CLINICAL_FEATURE_NAMES[:4], event_name="GvHD",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

F32 = npt.NDArray[np.float32]

# ---------------------------------------------------------------------------
# Clinical feature specification
# ---------------------------------------------------------------------------

#: Ordered list of clinical feature names corresponding to the 8 matrix columns.
CLINICAL_FEATURE_NAMES: list[str] = [
    "Age (recipient, yr)",
    "Age (donor, yr)",
    "CD34 dose (×10⁶/kg)",
    "Sex mismatch",
    "Disease",
    "Conditioning",
    "Donor type",
    "Stem cell source",
]

# Normalisation denominators fed to ClinicalEncoder (same as prepare_inputs).
# Cols 0-3 only; cols 4-7 are integer indices, no scaling needed.
_CONT_SCALE: npt.NDArray[np.float32] = np.array(
    [100.0, 100.0, 10.0, 1.0], dtype=np.float32
)

# Human-readable category labels (kept in sync with capa.model.capa_model).
_DISEASE_CATS: list[str] = [
    "unknown", "ALL", "AML", "CML", "MDS", "NHL", "HD", "AA", "MM", "other",
]
_CONDITIONING_CATS: list[str] = ["unknown", "MAC", "RIC", "NMA"]
_DONOR_TYPE_CATS: list[str] = ["unknown", "MSD", "MUD", "MMUD", "haplo", "cord"]
_STEM_CELL_CATS: list[str] = ["unknown", "BM", "PBSC", "cord"]

# Maps from feature-dict key → (column index, category list or None)
_CONT_KEYS: list[tuple[str, int]] = [
    ("age_recipient", 0),
    ("age_donor",     1),
    ("cd34_dose",     2),
    ("sex_mismatch",  3),
]
_CAT_KEYS: list[tuple[str, int, list[str]]] = [
    ("disease",          4, _DISEASE_CATS),
    ("conditioning",     5, _CONDITIONING_CATS),
    ("donor_type",       6, _DONOR_TYPE_CATS),
    ("stem_cell_source", 7, _STEM_CELL_CATS),
]

# Build reverse look-up for string → index conversion
_CAT_REVERSE: dict[str, dict[str, int]] = {
    key: {cat: i for i, cat in enumerate(cats)}
    for key, _, cats in _CAT_KEYS
}

# Colourmap for continuous feature values in beeswarm (red=high, blue=low)
_FEAT_CMAP = "RdBu_r"

# ---------------------------------------------------------------------------
# Figure styling (mirrors attention_maps.py)
# ---------------------------------------------------------------------------

_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

# Colours for waterfall bars
_POS_COLOUR = "#d62728"   # matplotlib tab:red — positive SHAP
_NEG_COLOUR = "#1f77b4"   # matplotlib tab:blue — negative SHAP
_BASE_COLOUR = "#7f7f7f"  # grey — base value connector


def _apply_style() -> None:
    """Apply publication-quality rcParams."""
    matplotlib.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SHAPExplanation:
    """All SHAP attribution data for a single competing event.

    Attributes
    ----------
    shap_values : F32, shape (n_samples, n_features)
        SHAP values; ``shap_values[i, j]`` is the contribution of feature
        *j* to the risk score of sample *i*.
    expected_value : float
        Model output averaged over the background distribution (base value
        for waterfall plots).
    feature_names : list[str]
        Human-readable name for each column in *shap_values*.
    feature_values : F32, shape (n_samples, n_features)
        Raw (un-normalised) feature values used for dot colouring in the
        beeswarm plot.
    event_name : str
        Label for the competing event (e.g. ``"GvHD"``).
    predictions : F32 | None, shape (n_samples,)
        Model outputs for the explained samples.  If ``None``, each sample's
        prediction is approximated as ``expected_value + shap_values[i].sum()``.
    """

    shap_values: F32
    expected_value: float
    feature_names: list[str]
    feature_values: F32
    event_name: str = "event"
    predictions: F32 | None = None

    @property
    def n_samples(self) -> int:
        """Number of explained samples."""
        return int(self.shap_values.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.shap_values.shape[1])

    def mean_abs_shap(self) -> F32:
        """Mean absolute SHAP value per feature, shape ``(n_features,)``."""
        return np.abs(self.shap_values).mean(axis=0).astype(np.float32)

    def prediction_for(self, sample_idx: int) -> float:
        """Return (or approximate) the model output for sample *sample_idx*."""
        if self.predictions is not None:
            return float(self.predictions[sample_idx])
        return float(self.expected_value + self.shap_values[sample_idx].sum())


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def clinical_dict_to_row(clinical: dict[str, Any]) -> F32:
    """Convert a raw clinical feature dict to an 8-element feature vector.

    Missing or unrecognised values are imputed with 0 (continuous) or 0
    (categorical index 0 = "unknown").

    Parameters
    ----------
    clinical : dict[str, Any]
        Keys: ``age_recipient``, ``age_donor``, ``cd34_dose``,
        ``sex_mismatch``, ``disease``, ``conditioning``,
        ``donor_type``, ``stem_cell_source``.  Any subset is accepted.

    Returns
    -------
    F32
        Shape ``(8,)`` float32 vector.  Columns 4-7 are category indices
        stored as float32 (cast to int64 before model inference).
    """
    row = np.zeros(8, dtype=np.float32)

    for key, col in _CONT_KEYS:
        val = clinical.get(key)
        try:
            row[col] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            row[col] = 0.0

    for key, col, _ in _CAT_KEYS:
        val = clinical.get(key)
        if val is None:
            row[col] = 0.0
        else:
            row[col] = float(_CAT_REVERSE[key].get(str(val), 0))

    return row


def clinical_dicts_to_matrix(records: list[dict[str, Any]]) -> F32:
    """Convert a list of clinical dicts to a feature matrix.

    Parameters
    ----------
    records : list[dict[str, Any]]
        One dict per patient.

    Returns
    -------
    F32
        Shape ``(n_records, 8)`` feature matrix.
    """
    return np.stack([clinical_dict_to_row(r) for r in records], axis=0)


def _format_feature_value(col: int, value: float) -> str:
    """Format a raw feature value for waterfall bar labels."""
    if col == 0 or col == 1:
        return f"{value:.1f} yr"
    if col == 2:
        return f"{value:.2f} ×10⁶/kg"
    if col == 3:
        return "yes" if value > 0.5 else "no"
    if col == 4:
        idx = int(round(value))
        return _DISEASE_CATS[idx] if 0 <= idx < len(_DISEASE_CATS) else str(idx)
    if col == 5:
        idx = int(round(value))
        return _CONDITIONING_CATS[idx] if 0 <= idx < len(_CONDITIONING_CATS) else str(idx)
    if col == 6:
        idx = int(round(value))
        return _DONOR_TYPE_CATS[idx] if 0 <= idx < len(_DONOR_TYPE_CATS) else str(idx)
    if col == 7:
        idx = int(round(value))
        return _STEM_CELL_CATS[idx] if 0 <= idx < len(_STEM_CELL_CATS) else str(idx)
    return f"{value:.3g}"


# ---------------------------------------------------------------------------
# Internal tensor helpers
# ---------------------------------------------------------------------------

def _matrix_to_tensors(
    X: F32,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Convert an (n, 8) clinical matrix to ``(cont, cat_indices)`` tensors.

    Parameters
    ----------
    X : F32, shape (n, 8)
    device : torch.device

    Returns
    -------
    cont : Tensor, shape (n, 4), float32
        Continuous features normalised for the ClinicalEncoder.
    cat_indices : Tensor, shape (n, 4), int64
        Integer category indices.
    """
    cont_raw = X[:, :4].astype(np.float32)
    cont_norm = cont_raw / _CONT_SCALE[np.newaxis, :]
    cont = torch.tensor(cont_norm, dtype=torch.float32, device=device)
    cat = torch.tensor(X[:, 4:].astype(np.int64), dtype=torch.long, device=device)
    return cont, cat


def _resolve_device(model: nn.Module, hint: str | torch.device | None) -> torch.device:
    if hint is not None:
        return torch.device(hint)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

def build_clinical_predict_fn(
    model: nn.Module,
    donor_embeddings: Tensor,
    recipient_embeddings: Tensor,
    *,
    event_idx: int = 0,
    device: str | torch.device | None = None,
) -> Callable[[F32], F32]:
    """Build a KernelExplainer-compatible predict function over clinical features.

    HLA mismatch features are held fixed at the given
    *donor_embeddings* / *recipient_embeddings*.  The returned callable maps
    an ``(n_samples, 8)`` clinical feature matrix to a ``(n_samples,)`` array
    of risk scores (CIF at the final time bin) for the specified event.

    Parameters
    ----------
    model : nn.Module
        Trained CAPAModel.  Must expose ``interaction``, ``clinical_encoder``
        and ``survival_head`` attributes.
    donor_embeddings : Tensor
        Shape ``(1, n_loci, embedding_dim)``.
    recipient_embeddings : Tensor
        Same shape.
    event_idx : int
        Index of the competing event to explain (0 = GvHD, 1 = relapse, …).
    device : str | torch.device | None
        Inference device.  Inferred from model parameters if ``None``.

    Returns
    -------
    Callable[[F32], F32]
        ``predict_fn(X) → scores`` where X is ``(n, 8)`` float32 and
        scores is ``(n,)`` float32.

    Raises
    ------
    AttributeError
        If *model* does not expose the required sub-modules.
    """
    for attr in ("interaction", "clinical_encoder", "survival_head"):
        if not hasattr(model, attr):
            raise AttributeError(
                f"model must expose '{attr}'; got {type(model).__name__}. "
                "Pass a CAPAModel or implement a custom predict_fn."
            )

    dev = _resolve_device(model, device)
    model.eval()

    # Pre-compute fixed interaction features — never changes during SHAP runs
    with torch.no_grad():
        interaction_feats: Tensor = model.interaction(
            donor_embeddings.to(dev),
            recipient_embeddings.to(dev),
        )  # (1, interaction_dim)

    def predict_fn(X: F32) -> F32:
        n = len(X)
        cont, cat = _matrix_to_tensors(X, dev)
        with torch.no_grad():
            clin = model.clinical_encoder(cont, cat)            # (n, clinical_dim)
            intr = interaction_feats.expand(n, -1)              # (n, interaction_dim)
            combined = torch.cat([intr, clin], dim=-1)

            if hasattr(model.survival_head, "cif"):
                cif = model.survival_head.cif(combined)         # (n, K, T)
            else:
                logits = model.survival_head(combined)
                nb, ne, nt = logits.shape
                joint = F.softmax(logits.view(nb, -1), dim=-1).view(nb, ne, nt)
                cif = torch.cumsum(joint, dim=2)

        return cif[:, event_idx, -1].cpu().numpy().astype(np.float32)

    return predict_fn


class ClinicalDeepWrapper(nn.Module):
    """Differentiable wrapper for DeepExplainer over continuous clinical features.

    DeepExplainer requires a fully differentiable model with a single
    floating-point input tensor.  This module:

    * Accepts a ``(batch, 4)`` float32 tensor of **normalised** continuous
      features ``[age_recipient/100, age_donor/100, cd34_dose/10,
      sex_mismatch]``.
    * Holds categorical covariates fixed at the background median indices.
    * Holds interaction features fixed (pre-computed from donor/recipient
      embeddings).
    * Returns a ``(batch, 1)`` risk score tensor for the specified event.

    Prefer constructing via :meth:`from_model`.

    Parameters
    ----------
    clinical_encoder : nn.Module
        The ``ClinicalEncoder`` from a CAPAModel.
    survival_head : nn.Module
        The survival head from a CAPAModel.
    interaction_feats : Tensor, shape (1, interaction_dim)
        Pre-computed, fixed interaction representation.
    median_cat_indices : Tensor, shape (1, 4), int64
        Background median category indices (fixed).
    event_idx : int
        Which event's terminal CIF to return.
    """

    def __init__(
        self,
        clinical_encoder: nn.Module,
        survival_head: nn.Module,
        interaction_feats: Tensor,
        median_cat_indices: Tensor,
        event_idx: int = 0,
    ) -> None:
        super().__init__()
        self.clinical_encoder = clinical_encoder
        self.survival_head = survival_head
        self.register_buffer("interaction_feats", interaction_feats)
        self.register_buffer("median_cat_indices", median_cat_indices)
        self.event_idx = event_idx

    def forward(self, cont_norm: Tensor) -> Tensor:
        """Compute risk score from normalised continuous clinical features.

        Parameters
        ----------
        cont_norm : Tensor, shape (batch, 4)
            Normalised continuous features.

        Returns
        -------
        Tensor, shape (batch, 1)
            CIF at the final time bin for the specified event.
        """
        n = cont_norm.shape[0]
        cat = self.median_cat_indices.expand(n, -1)  # type: ignore[attr-defined]
        clin = self.clinical_encoder(cont_norm, cat)
        intr = self.interaction_feats.expand(n, -1)  # type: ignore[attr-defined]
        combined = torch.cat([intr, clin], dim=-1)

        if hasattr(self.survival_head, "cif"):
            cif = self.survival_head.cif(combined)
        else:
            logits = self.survival_head(combined)
            nb, ne, nt = logits.shape
            joint = F.softmax(logits.view(nb, -1), dim=-1).view(nb, ne, nt)
            cif = torch.cumsum(joint, dim=2)

        return cif[:, self.event_idx, -1].unsqueeze(1)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        donor_embeddings: Tensor,
        recipient_embeddings: Tensor,
        background_matrix: F32,
        *,
        event_idx: int = 0,
        device: str | torch.device | None = None,
    ) -> tuple["ClinicalDeepWrapper", Tensor]:
        """Construct a wrapper and background tensor from a CAPAModel.

        Parameters
        ----------
        model : nn.Module
            Trained CAPAModel.
        donor_embeddings : Tensor
            Shape ``(1, n_loci, embedding_dim)``.
        recipient_embeddings : Tensor
            Same shape.
        background_matrix : F32, shape (n_bg, 8)
            Background clinical feature matrix.  Categorical medians are
            derived from this set.
        event_idx : int
        device : str | torch.device | None

        Returns
        -------
        wrapper : ClinicalDeepWrapper
        background_tensor : Tensor, shape (n_bg, 4)
            Normalised continuous features of the background set, ready to
            pass to :func:`compute_shap_values_deep`.
        """
        for attr in ("interaction", "clinical_encoder", "survival_head"):
            if not hasattr(model, attr):
                raise AttributeError(f"model must expose '{attr}'")

        dev = _resolve_device(model, device)
        model.eval()

        with torch.no_grad():
            interaction_feats = model.interaction(
                donor_embeddings.to(dev),
                recipient_embeddings.to(dev),
            )

        # Median categorical indices across background
        cat_bg = background_matrix[:, 4:].astype(np.int64)
        median_cats = np.median(cat_bg, axis=0).astype(np.int64)
        median_tensor = torch.tensor(
            median_cats[np.newaxis, :], dtype=torch.long, device=dev
        )

        # Background continuous tensor (normalised)
        cont_bg_norm = background_matrix[:, :4].astype(np.float32) / _CONT_SCALE[np.newaxis, :]
        bg_tensor = torch.tensor(cont_bg_norm, dtype=torch.float32, device=dev)

        wrapper = cls(
            clinical_encoder=model.clinical_encoder,
            survival_head=model.survival_head,
            interaction_feats=interaction_feats,
            median_cat_indices=median_tensor,
            event_idx=event_idx,
        ).to(dev)

        return wrapper, bg_tensor


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    predict_fn: Any,
    background_data: F32,
    explain_data: F32,
) -> F32:
    """Compute SHAP values using KernelExplainer (model-agnostic).

    Parameters
    ----------
    predict_fn : callable
        ``f(X: ndarray) -> ndarray`` mapping ``(n, n_features)`` to
        ``(n,)`` risk scores.  Typically built with
        :func:`build_clinical_predict_fn`.
    background_data : F32, shape (n_bg, n_features)
        Reference distribution summarising the training set.  50–200
        samples are usually sufficient.
    explain_data : F32, shape (n_explain, n_features)
        Samples to explain.

    Returns
    -------
    F32, shape (n_explain, n_features)
        SHAP values; entry ``[i, j]`` is the contribution of feature *j*
        to sample *i*'s deviation from the base value.

    Raises
    ------
    ImportError
        If ``shap`` is not installed (``uv add shap``).
    """
    try:
        import shap  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("shap is required: uv add shap") from exc

    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values: F32 = explainer.shap_values(explain_data)
    return shap_values


def compute_shap_values_deep(
    wrapper: "ClinicalDeepWrapper",
    background_tensor: Tensor,
    explain_tensor: Tensor,
) -> tuple[F32, float]:
    """Compute SHAP values using DeepExplainer (continuous features only).

    Parameters
    ----------
    wrapper : ClinicalDeepWrapper
        Differentiable model wrapper built with
        :meth:`ClinicalDeepWrapper.from_model`.
    background_tensor : Tensor, shape (n_bg, 4)
        Normalised continuous background features.
    explain_tensor : Tensor, shape (n_explain, 4)
        Normalised continuous features to explain.

    Returns
    -------
    shap_values : F32, shape (n_explain, 4)
        SHAP values for the 4 continuous features.
    expected_value : float
        Base value (mean model output on the background set).

    Raises
    ------
    ImportError
        If ``shap`` is not installed.
    """
    try:
        import shap  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("shap is required: uv add shap") from exc

    wrapper.eval()
    explainer = shap.DeepExplainer(wrapper, background_tensor)
    raw = explainer.shap_values(explain_tensor)

    # DeepExplainer may return list (one per output) or array
    if isinstance(raw, list):
        shap_arr = np.array(raw[0], dtype=np.float32)
    else:
        shap_arr = np.array(raw, dtype=np.float32)

    # Squeeze output dimension if present (wrapper returns (n, 1))
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[:, :, 0]

    expected_value = float(
        wrapper(background_tensor).detach().cpu().numpy().mean()
    )
    logger.info(
        "DeepExplainer: %d samples explained, base value = %.4f",
        len(explain_tensor),
        expected_value,
    )
    return shap_arr.astype(np.float32), expected_value


def build_explanation(
    shap_values: F32,
    expected_value: float,
    feature_matrix: F32,
    *,
    feature_names: list[str] | None = None,
    event_name: str = "event",
    predictions: F32 | None = None,
) -> SHAPExplanation:
    """Package raw SHAP arrays into a :class:`SHAPExplanation`.

    Parameters
    ----------
    shap_values : F32, shape (n_samples, n_features)
    expected_value : float
        Base value from the explainer.
    feature_matrix : F32, shape (n_samples, n_features)
        Raw (un-normalised) feature values used for coloring in plots.
    feature_names : list[str] | None
        Defaults to :data:`CLINICAL_FEATURE_NAMES` (trimmed to *n_features*).
    event_name : str
    predictions : F32 | None, shape (n_samples,)
        Actual model outputs; if ``None`` they are approximated from SHAP.

    Returns
    -------
    SHAPExplanation
    """
    n_feat = shap_values.shape[1]
    if feature_names is None:
        feature_names = CLINICAL_FEATURE_NAMES[:n_feat]

    return SHAPExplanation(
        shap_values=shap_values.astype(np.float32),
        expected_value=float(expected_value),
        feature_names=list(feature_names),
        feature_values=feature_matrix[:, :n_feat].astype(np.float32),
        event_name=event_name,
        predictions=(predictions.astype(np.float32) if predictions is not None else None),
    )


# ---------------------------------------------------------------------------
# Internal plot helpers
# ---------------------------------------------------------------------------

def _beeswarm_y_positions(
    shap_vals: F32,
    strip_width: float = 0.72,
    n_bins: int = 50,
) -> F32:
    """Compute vertical jitter for one feature strip in the beeswarm plot.

    Points are binned along the x-axis (SHAP value axis) and distributed
    evenly in the y-direction within each bin to avoid overlap.

    Parameters
    ----------
    shap_vals : F32, shape (n_samples,)
    strip_width : float
        Maximum total y-extent of the strip (in axes data units).
    n_bins : int
        Number of bins along the SHAP value axis.

    Returns
    -------
    F32, shape (n_samples,)
        Y jitter offsets centred around 0.
    """
    n = len(shap_vals)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    x_min, x_max = float(shap_vals.min()), float(shap_vals.max())
    if x_min == x_max:
        return np.zeros(n, dtype=np.float32)

    edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(shap_vals, edges) - 1, 0, n_bins - 1)

    y = np.zeros(n, dtype=np.float32)
    for b_id in range(n_bins):
        members = np.where(bin_ids == b_id)[0]
        count = len(members)
        if count == 0:
            continue
        if count == 1:
            y[members[0]] = 0.0
        else:
            half = strip_width / 2.0 * (count - 1) / count
            y[members] = np.linspace(-half, half, count)
    return y


def _normalise_for_colour(values: F32) -> F32:
    """Normalise a 1-D array to ``[0, 1]`` for colormap application."""
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full_like(values, 0.5)
    return ((values - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_beeswarm(
    expl: SHAPExplanation,
    *,
    max_display: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dot_size: float = 16.0,
    strip_width: float = 0.72,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Summary beeswarm plot: one dot per sample, coloured by feature value.

    Features are sorted by mean absolute SHAP value (most important at the
    top).  Dots are jittered vertically within each feature strip to avoid
    overlap.

    Parameters
    ----------
    expl : SHAPExplanation
    max_display : int
        Maximum number of features to show.  Less-important features are
        omitted (not aggregated).
    title : str | None
        Figure title.  Defaults to ``"SHAP Summary — {event_name}"``.
    figsize : tuple[float, float] | None
    dot_size : float
        Scatter marker size in points.
    strip_width : float
        Maximum y-extent of each feature's dot strip.
    ax : plt.Axes | None
        Existing axes; a new figure is created if ``None``.

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    # Sort features by mean |SHAP| descending, keep top-max_display
    order = np.argsort(expl.mean_abs_shap())[::-1][:max_display]
    # Reverse so most important is at the top (highest y)
    order = order[::-1]

    n_show = len(order)
    if figsize is None:
        figsize = (8.0, max(4.0, n_show * 0.65 + 1.5))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    assert ax is not None

    cmap = cm.get_cmap(_FEAT_CMAP)

    for y_pos, feat_i in enumerate(order):
        sv = expl.shap_values[:, feat_i]
        fv = expl.feature_values[:, feat_i]
        colours = cmap(_normalise_for_colour(fv))

        jitter = _beeswarm_y_positions(sv, strip_width=strip_width)

        ax.scatter(
            sv, y_pos + jitter,
            c=colours,
            s=dot_size,
            linewidths=0.15,
            edgecolors="white",
            zorder=3,
            rasterized=True,  # keeps PDF vector for axes, dots as bitmap
        )

    # Y-axis tick labels
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(
        [expl.feature_names[i] for i in order],
        fontsize=10,
    )
    ax.set_ylim(-0.7, n_show - 0.3)

    # X-axis and zero line
    ax.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlabel("SHAP value (impact on risk score)", fontsize=11)

    # Horizontal grid
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Colorbar for feature values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    if own_fig:
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
        cbar.set_label("Feature value\n(low → high)", fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

    # Title
    _title = title or f"SHAP Summary \u2014 {expl.event_name}"
    ax.set_title(_title, fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    if own_fig:
        fig.tight_layout()

    logger.info("Beeswarm plot created for event '%s' (%d samples)", expl.event_name, expl.n_samples)
    return fig


def plot_waterfall(
    expl: SHAPExplanation,
    *,
    sample_idx: int = 0,
    max_display: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Waterfall plot: individual patient SHAP decomposition.

    Shows how each clinical feature contributes to pushing the predicted
    risk above or below the expected value (base value).  The features are
    sorted by descending absolute SHAP magnitude for the chosen sample.

    Parameters
    ----------
    expl : SHAPExplanation
    sample_idx : int
        Which sample to explain.
    max_display : int
        Maximum number of feature bars.  Remaining contributions are
        aggregated into a single "other features" bar.
    title : str | None
        Defaults to ``"SHAP Waterfall — {event_name} | sample {sample_idx}"``.
    figsize : tuple[float, float] | None
    ax : plt.Axes | None

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    sv = expl.shap_values[sample_idx]          # (n_features,)
    fv = expl.feature_values[sample_idx]       # (n_features,)
    base = expl.expected_value
    pred = expl.prediction_for(sample_idx)

    # Sort by |SHAP| descending
    order = np.argsort(np.abs(sv))[::-1]

    # Split into displayed and aggregated-"other"
    show_idx = order[:max_display]
    other_idx = order[max_display:]
    has_other = len(other_idx) > 0

    n_bars = len(show_idx) + (1 if has_other else 0)
    if figsize is None:
        figsize = (9.0, max(4.0, n_bars * 0.55 + 2.5))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    assert ax is not None

    # Build bar list (bottom to top in chart)
    bar_labels: list[str] = []
    bar_shap: list[float] = []

    # "Other features" bar aggregated first (shown at bottom)
    if has_other:
        other_total = float(sv[other_idx].sum())
        bar_labels.append(f"Other {len(other_idx)} features")
        bar_shap.append(other_total)

    # Individual feature bars (least important first → at bottom of display)
    for fi in show_idx[::-1]:
        name = expl.feature_names[fi]
        val_str = _format_feature_value(fi, float(fv[fi]))
        bar_labels.append(f"{name} = {val_str}")
        bar_shap.append(float(sv[fi]))

    # Compute running cumulative start positions (from base value)
    running = base
    starts: list[float] = []
    for s in bar_shap:
        starts.append(running)
        running += s

    # Draw horizontal bars
    for row, (start, width, label) in enumerate(zip(starts, bar_shap, bar_labels)):
        colour = _POS_COLOUR if width >= 0 else _NEG_COLOUR
        ax.barh(
            row, width, left=start,
            color=colour, alpha=0.85,
            height=0.65,
            linewidth=0.0,
        )
        # Numeric annotation at bar edge
        edge = start + width
        ha = "left" if width >= 0 else "right"
        offset = (pred - base) * 0.01
        ax.text(
            edge + (offset if width >= 0 else -offset),
            row,
            f"{width:+.3f}",
            ha=ha, va="center",
            fontsize=8.5, color="black",
        )

    # Vertical lines at base value and prediction
    ax.axvline(base, color=_BASE_COLOUR, linewidth=1.2, linestyle="--", zorder=4)
    ax.axvline(pred, color="black",       linewidth=1.5, linestyle="-",  zorder=4)

    # Annotations for base and prediction
    ylim_top = n_bars - 0.3
    ax.annotate(
        f"E[f(x)] = {base:.3f}",
        xy=(base, ylim_top), xytext=(base, ylim_top + 0.35),
        ha="center", fontsize=9, color=_BASE_COLOUR,
        arrowprops={"arrowstyle": "-", "color": _BASE_COLOUR, "lw": 0.8},
    )
    ax.annotate(
        f"f(x) = {pred:.3f}",
        xy=(pred, ylim_top), xytext=(pred, ylim_top + 0.35),
        ha="center", fontsize=9, color="black",
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8},
    )

    # Y-axis labels
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(bar_labels, fontsize=9.5)
    ax.set_ylim(-0.6, ylim_top + 0.8)

    ax.set_xlabel("Model output (risk score)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    _title = title or (
        f"SHAP Waterfall \u2014 {expl.event_name} | sample {sample_idx}"
    )
    ax.set_title(_title, fontsize=12, pad=10)

    if own_fig:
        fig.tight_layout()

    logger.info(
        "Waterfall plot for sample %d: base=%.4f, prediction=%.4f",
        sample_idx, base, pred,
    )
    return fig


def plot_feature_importance(
    expl: SHAPExplanation,
    *,
    max_display: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    colour: str = "#4878CF",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart of mean absolute SHAP values per feature.

    Suitable for inclusion in methods sections; shows which clinical
    covariates drive the model's predictions on average.

    Parameters
    ----------
    expl : SHAPExplanation
    max_display : int
        Maximum number of features to show (most important first).
    title : str | None
        Defaults to ``"Feature Importance — {event_name}"``.
    figsize : tuple[float, float] | None
    colour : str
        Bar colour (hex or named matplotlib colour).
    ax : plt.Axes | None

    Returns
    -------
    plt.Figure
    """
    _apply_style()

    mean_abs = expl.mean_abs_shap()
    order = np.argsort(mean_abs)[::-1][:max_display]
    order = order[::-1]  # least important first (bottom of chart)

    n_show = len(order)
    if figsize is None:
        figsize = (7.0, max(3.5, n_show * 0.5 + 1.5))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    assert ax is not None

    values = mean_abs[order]
    labels = [expl.feature_names[i] for i in order]

    ax.barh(range(n_show), values, color=colour, alpha=0.85, height=0.65)

    # Numeric annotation
    for i, v in enumerate(values):
        ax.text(v + values.max() * 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    ax.set_yticks(range(n_show))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_xlim(0, values.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    _title = title or f"Feature Importance \u2014 {expl.event_name}"
    ax.set_title(_title, fontsize=12, pad=10)

    if own_fig:
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
    """Save a figure to PDF and PNG (or custom formats).

    Parameters
    ----------
    fig : plt.Figure
    path : Path | str
        Base path **without** extension.
    dpi : int
        DPI for raster formats.
    formats : list[str] | None
        Defaults to ``["pdf", "png"]``.
    bbox_inches : str
        Passed to ``savefig``; ``"tight"`` trims whitespace.

    Returns
    -------
    list[Path]
        Absolute paths of all written files.
    """
    if formats is None:
        formats = ["pdf", "png"]

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        logger.debug("Saved figure → %s", out)
        saved.append(out)

    return saved


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def generate_shap_report(
    model: nn.Module,
    donor_embeddings: Tensor,
    recipient_embeddings: Tensor,
    background_matrix: F32,
    explain_matrix: F32,
    *,
    event_idx: int = 0,
    event_name: str = "event",
    out_dir: Path | str | None = None,
    waterfall_indices: list[int] | None = None,
    max_display: int = 10,
    device: str | torch.device | None = None,
    dpi: int = 300,
) -> dict[str, plt.Figure]:
    """Compute SHAP values and generate all report figures in one call.

    Runs KernelExplainer over all 8 clinical features, then produces:

    * ``"beeswarm"`` — population summary beeswarm.
    * ``"feature_importance"`` — mean |SHAP| bar chart.
    * ``"waterfall_{i}"`` — one waterfall per sample in *waterfall_indices*.

    Parameters
    ----------
    model : nn.Module
        Trained CAPAModel.
    donor_embeddings : Tensor
        Shape ``(1, n_loci, embedding_dim)``.
    recipient_embeddings : Tensor
        Same shape.
    background_matrix : F32, shape (n_bg, 8)
        Background clinical feature matrix (50–200 samples recommended).
    explain_matrix : F32, shape (n_explain, 8)
        Samples to explain.
    event_idx : int
        Which competing event to explain (0 = GvHD, 1 = relapse, …).
    event_name : str
        Human-readable label for the event.
    out_dir : Path | str | None
        If provided, all figures are saved under this directory.
    waterfall_indices : list[int] | None
        Indices in *explain_matrix* for waterfall plots.  Defaults to
        ``[0]`` (first patient).
    max_display : int
        Maximum features shown in each plot.
    device : str | torch.device | None
    dpi : int
        Save resolution.

    Returns
    -------
    dict[str, plt.Figure]
        Keys: ``"beeswarm"``, ``"feature_importance"``,
        ``"waterfall_0"``, ``"waterfall_1"``, …
    """
    if waterfall_indices is None:
        waterfall_indices = [0]

    # Build predict function and compute SHAP values
    predict_fn = build_clinical_predict_fn(
        model, donor_embeddings, recipient_embeddings,
        event_idx=event_idx, device=device,
    )

    logger.info(
        "Running KernelExplainer: %d background, %d explain samples",
        len(background_matrix), len(explain_matrix),
    )
    shap_vals = compute_shap_values(predict_fn, background_matrix, explain_matrix)

    # Base value: mean prediction on background
    base_val = float(predict_fn(background_matrix).mean())
    predictions = predict_fn(explain_matrix)

    expl = build_explanation(
        shap_vals, base_val, explain_matrix,
        event_name=event_name,
        predictions=predictions,
    )

    figures: dict[str, plt.Figure] = {}

    figures["beeswarm"] = plot_beeswarm(expl, max_display=max_display)
    figures["feature_importance"] = plot_feature_importance(expl, max_display=max_display)

    for idx in waterfall_indices:
        if idx < 0 or idx >= expl.n_samples:
            logger.warning("waterfall_indices: %d out of range (%d samples), skipping", idx, expl.n_samples)
            continue
        figures[f"waterfall_{idx}"] = plot_waterfall(
            expl, sample_idx=idx, max_display=max_display
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        for key, fig in figures.items():
            save_figure(fig, out_dir / f"shap_{key}", dpi=dpi)
        logger.info("SHAP report saved to %s", out_dir)

    return figures
