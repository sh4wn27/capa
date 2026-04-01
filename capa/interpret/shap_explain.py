"""SHAP-based explanations for clinical covariates."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def compute_shap_values(
    predict_fn: Any,
    background_data: npt.NDArray[np.float32],
    explain_data: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute SHAP values for clinical covariates using KernelExplainer.

    Parameters
    ----------
    predict_fn : callable
        A function ``f(X) -> np.ndarray`` that takes a 2-D array of clinical
        features and returns predicted risk scores.
    background_data : npt.NDArray[np.float32]
        Background dataset for SHAP reference distribution, shape ``(n_bg, n_features)``.
    explain_data : npt.NDArray[np.float32]
        Data points to explain, shape ``(n_explain, n_features)``.

    Returns
    -------
    npt.NDArray[np.float32]
        SHAP values of shape ``(n_explain, n_features)``.
    """
    try:
        import shap  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("shap is required: uv add shap") from exc

    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values: npt.NDArray[np.float32] = explainer.shap_values(explain_data)
    return shap_values
