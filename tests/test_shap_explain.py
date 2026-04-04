"""Tests for capa/interpret/shap_explain.py.

Coverage targets
----------------
Constants / feature spec
  - CLINICAL_FEATURE_NAMES has 8 entries

clinical_dict_to_row
  - shape (8,) float32
  - correct continuous values
  - correct categorical index lookup
  - missing keys default to 0
  - unrecognised categorical string → index 0
  - bad float value → 0

clinical_dicts_to_matrix
  - shape (n, 8) float32
  - batches of clinical_dict_to_row

_format_feature_value (internal)
  - all 8 column branches

_matrix_to_tensors (internal)
  - cont shape (n, 4), dtype float32
  - cat shape (n, 4), dtype int64
  - continuous values are normalised

SHAPExplanation
  - n_samples / n_features properties
  - mean_abs_shap shape and values
  - prediction_for when predictions provided
  - prediction_for approximated from SHAP when predictions=None

build_explanation
  - default feature_names from CLINICAL_FEATURE_NAMES
  - custom feature_names preserved
  - predictions optional

build_clinical_predict_fn
  - raises AttributeError when model missing sub-modules
  - returns callable
  - output shape (n_samples,)
  - output is float32
  - runs without gradient

ClinicalDeepWrapper
  - from_model constructs wrapper and background tensor
  - forward returns (n, 1) tensor
  - from_model raises AttributeError on bad model

compute_shap_values (KernelExplainer)
  - raises ImportError when shap is missing
  - returns correct shape given mock KernelExplainer
  - calls KernelExplainer with correct args

compute_shap_values_deep
  - raises ImportError when shap is missing
  - returns (shap_values, expected_value) with correct shapes

plot_beeswarm
  - returns Figure
  - max_display respected (fewer features shown)
  - with existing ax

plot_waterfall
  - returns Figure with bars
  - max_display aggregates "other features"
  - sample_idx parameter

plot_feature_importance
  - returns Figure
  - max_display respected

save_figure
  - writes PDF and PNG
  - custom formats
  - creates parent directory

generate_shap_report
  - returns dict with correct keys
  - waterfall_indices honoured
  - out_of_range waterfall_index is skipped (warning)
  - saves files when out_dir given
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capa.interpret.shap_explain import (
    CLINICAL_FEATURE_NAMES,
    ClinicalDeepWrapper,
    SHAPExplanation,
    _CONT_SCALE,
    _beeswarm_y_positions,
    _format_feature_value,
    _matrix_to_tensors,
    _normalise_for_colour,
    _resolve_device,
    build_clinical_predict_fn,
    build_explanation,
    clinical_dict_to_row,
    clinical_dicts_to_matrix,
    compute_shap_values,
    compute_shap_values_deep,
    generate_shap_report,
    plot_beeswarm,
    plot_feature_importance,
    plot_waterfall,
    save_figure,
)
from capa.model.interaction import AttentionWeights


# ---------------------------------------------------------------------------
# Tiny model fixture that mimics CAPAModel structure
# ---------------------------------------------------------------------------

class _TinyInteraction(nn.Module):
    OUT_DIM = 4

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, self.OUT_DIM, bias=False)
        self.last_attn_weights: AttentionWeights | None = None

    def forward(self, donor: torch.Tensor, recip: torch.Tensor) -> torch.Tensor:
        B = donor.shape[0]
        attn = torch.rand(B, 3, 3)
        self.last_attn_weights = AttentionWeights(
            donor_to_recip=[attn], recip_to_donor=[attn]
        )
        return self.linear(torch.ones(B, 1))


class _TinyClinicalEncoder(nn.Module):
    CLIN_DIM = 6

    def __init__(self) -> None:
        super().__init__()
        # Encode 4 cont features + 4 cat embeddings (1-d each)
        self.cat_emb = nn.ModuleList([nn.Embedding(10, 1) for _ in range(4)])
        self.mlp = nn.Linear(4 + 4, self.CLIN_DIM)

    def forward(self, cont: torch.Tensor, cat_idx: torch.Tensor) -> torch.Tensor:
        embeds = [self.cat_emb[i](cat_idx[:, i]) for i in range(4)]
        cat_vec = torch.cat(embeds, dim=-1)
        x = torch.cat([cont, cat_vec], dim=-1)
        return self.mlp(x)


class _TinySurvivalHead(nn.Module):
    NUM_EVENTS = 2
    TIME_BINS = 5

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, self.NUM_EVENTS * self.TIME_BINS)
        self.num_events = self.NUM_EVENTS
        self.time_bins = self.TIME_BINS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out.view(-1, self.NUM_EVENTS, self.TIME_BINS)

    def cif(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        B, K, T = logits.shape
        joint = F.softmax(logits.view(B, -1), dim=-1).view(B, K, T)
        return torch.cumsum(joint, dim=2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interaction = _TinyInteraction()
        self.clinical_encoder = _TinyClinicalEncoder()
        combined_dim = _TinyInteraction.OUT_DIM + _TinyClinicalEncoder.CLIN_DIM
        self.survival_head = _TinySurvivalHead(combined_dim)

    def get_attention_weights(self) -> AttentionWeights | None:
        return self.interaction.last_attn_weights

    def forward(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        intr = self.interaction(donor, recip)
        combined = torch.cat([intr, clinical], dim=-1)
        return self.survival_head(combined)


class _NoSubModulesModel(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1)


N_LOCI = 3
EMB_DIM = 8
N_FEAT = 8


@pytest.fixture()
def tiny_model() -> _TinyModel:
    m = _TinyModel()
    m.eval()
    return m


@pytest.fixture()
def donor_emb() -> torch.Tensor:
    return torch.randn(1, N_LOCI, EMB_DIM)


@pytest.fixture()
def recip_emb() -> torch.Tensor:
    return torch.randn(1, N_LOCI, EMB_DIM)


@pytest.fixture()
def clinical_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    X = np.zeros((10, 8), dtype=np.float32)
    X[:, 0] = rng.uniform(1, 18, 10)    # age_recipient
    X[:, 1] = rng.uniform(20, 50, 10)   # age_donor
    X[:, 2] = rng.uniform(1, 10, 10)    # cd34_dose
    X[:, 3] = rng.integers(0, 2, 10)    # sex_mismatch
    X[:, 4] = rng.integers(0, 9, 10)    # disease
    X[:, 5] = rng.integers(0, 3, 10)    # conditioning
    X[:, 6] = rng.integers(0, 5, 10)    # donor_type
    X[:, 7] = rng.integers(0, 3, 10)    # stem_cell_source
    return X


@pytest.fixture()
def shap_expl(clinical_matrix: np.ndarray) -> SHAPExplanation:
    rng = np.random.default_rng(1)
    shap_vals = rng.standard_normal((10, 8)).astype(np.float32)
    return SHAPExplanation(
        shap_values=shap_vals,
        expected_value=0.35,
        feature_names=CLINICAL_FEATURE_NAMES,
        feature_values=clinical_matrix.astype(np.float32),
        event_name="GvHD",
        predictions=(0.35 + shap_vals.sum(axis=1)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_clinical_feature_names_length() -> None:
    assert len(CLINICAL_FEATURE_NAMES) == 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_hint_not_none_returns_hint(self) -> None:
        model = nn.Linear(2, 2)
        dev = _resolve_device(model, "cpu")
        assert dev == torch.device("cpu")

    def test_infer_from_model(self) -> None:
        model = nn.Linear(2, 2)
        dev = _resolve_device(model, None)
        assert isinstance(dev, torch.device)

    def test_no_params_falls_back_to_cpu(self) -> None:
        class _Empty(nn.Module):
            def forward(self) -> None:  # type: ignore[override]
                pass

        dev = _resolve_device(_Empty(), None)
        assert dev == torch.device("cpu")


class TestBeeswarmYPositions:
    def test_single_point_returns_zero(self) -> None:
        vals = np.array([0.5], dtype=np.float32)
        y = _beeswarm_y_positions(vals)
        assert y.shape == (1,)
        assert y[0] == 0.0

    def test_constant_values_returns_zeros(self) -> None:
        vals = np.full(5, 0.3, dtype=np.float32)
        y = _beeswarm_y_positions(vals)
        np.testing.assert_array_equal(y, 0.0)

    def test_normal_case_shape(self) -> None:
        rng = np.random.default_rng(0)
        vals = rng.standard_normal(20).astype(np.float32)
        y = _beeswarm_y_positions(vals)
        assert y.shape == (20,)


class TestNormaliseForColour:
    def test_constant_input_returns_half(self) -> None:
        vals = np.full(5, 3.0, dtype=np.float32)
        out = _normalise_for_colour(vals)
        np.testing.assert_allclose(out, 0.5)

    def test_normal_range_is_zero_to_one(self) -> None:
        vals = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        out = _normalise_for_colour(vals)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# clinical_dict_to_row
# ---------------------------------------------------------------------------


class TestClinicalDictToRow:
    def test_shape_and_dtype(self) -> None:
        row = clinical_dict_to_row({})
        assert row.shape == (8,)
        assert row.dtype == np.float32

    def test_continuous_values(self) -> None:
        row = clinical_dict_to_row({"age_recipient": 12.5, "cd34_dose": 5.2})
        assert row[0] == pytest.approx(12.5)
        assert row[2] == pytest.approx(5.2)

    def test_sex_mismatch_float(self) -> None:
        row = clinical_dict_to_row({"sex_mismatch": 1})
        assert row[3] == pytest.approx(1.0)

    def test_categorical_index_lookup(self) -> None:
        row = clinical_dict_to_row({"disease": "ALL"})
        assert row[4] == pytest.approx(1.0)  # ALL is index 1

    def test_unknown_categorical_defaults_to_zero(self) -> None:
        row = clinical_dict_to_row({"disease": "INVALID_DISEASE"})
        assert row[4] == pytest.approx(0.0)

    def test_missing_keys_default_to_zero(self) -> None:
        row = clinical_dict_to_row({})
        assert np.all(row == 0.0)

    def test_bad_float_value_defaults_to_zero(self) -> None:
        row = clinical_dict_to_row({"age_recipient": "not_a_number"})
        assert row[0] == pytest.approx(0.0)

    def test_all_categoricals(self) -> None:
        row = clinical_dict_to_row({
            "disease": "AML",           # idx 2
            "conditioning": "MAC",      # idx 1
            "donor_type": "MUD",        # idx 2
            "stem_cell_source": "BM",   # idx 1
        })
        assert row[4] == pytest.approx(2.0)
        assert row[5] == pytest.approx(1.0)
        assert row[6] == pytest.approx(2.0)
        assert row[7] == pytest.approx(1.0)


class TestClinicalDictsToMatrix:
    def test_shape(self) -> None:
        recs = [{"age_recipient": float(i)} for i in range(5)]
        M = clinical_dicts_to_matrix(recs)
        assert M.shape == (5, 8)
        assert M.dtype == np.float32

    def test_values_match_single(self) -> None:
        d = {"age_recipient": 8.0, "disease": "ALL"}
        M = clinical_dicts_to_matrix([d])
        np.testing.assert_array_equal(M[0], clinical_dict_to_row(d))


# ---------------------------------------------------------------------------
# _format_feature_value
# ---------------------------------------------------------------------------


class TestFormatFeatureValue:
    def test_age_recipient(self) -> None:
        assert "yr" in _format_feature_value(0, 12.5)

    def test_age_donor(self) -> None:
        assert "yr" in _format_feature_value(1, 35.0)

    def test_cd34_dose(self) -> None:
        assert "×10" in _format_feature_value(2, 5.2)

    def test_sex_mismatch_yes(self) -> None:
        assert _format_feature_value(3, 1.0) == "yes"

    def test_sex_mismatch_no(self) -> None:
        assert _format_feature_value(3, 0.0) == "no"

    def test_disease(self) -> None:
        result = _format_feature_value(4, 1.0)  # index 1 = ALL
        assert result == "ALL"

    def test_conditioning(self) -> None:
        result = _format_feature_value(5, 1.0)  # index 1 = MAC
        assert result == "MAC"

    def test_donor_type(self) -> None:
        result = _format_feature_value(6, 2.0)  # index 2 = MUD
        assert result == "MUD"

    def test_stem_cell_source(self) -> None:
        result = _format_feature_value(7, 1.0)  # index 1 = BM
        assert result == "BM"

    def test_unknown_column(self) -> None:
        # column outside 0-7 falls back to generic format
        result = _format_feature_value(99, 3.14159)
        assert "3.14" in result


# ---------------------------------------------------------------------------
# _matrix_to_tensors
# ---------------------------------------------------------------------------


class TestMatrixToTensors:
    def test_shapes(self, clinical_matrix: np.ndarray) -> None:
        cont, cat = _matrix_to_tensors(clinical_matrix, torch.device("cpu"))
        assert cont.shape == (10, 4)
        assert cat.shape == (10, 4)

    def test_cont_dtype(self, clinical_matrix: np.ndarray) -> None:
        cont, _ = _matrix_to_tensors(clinical_matrix, torch.device("cpu"))
        assert cont.dtype == torch.float32

    def test_cat_dtype(self, clinical_matrix: np.ndarray) -> None:
        _, cat = _matrix_to_tensors(clinical_matrix, torch.device("cpu"))
        assert cat.dtype == torch.long

    def test_continuous_normalised(self, clinical_matrix: np.ndarray) -> None:
        cont, _ = _matrix_to_tensors(clinical_matrix, torch.device("cpu"))
        cont_np = cont.numpy()
        # age_recipient / 100 should be small
        assert cont_np[:, 0].max() < 1.0
        # Verify against expected normalisation
        expected = clinical_matrix[:, :4] / _CONT_SCALE[np.newaxis, :]
        np.testing.assert_allclose(cont_np, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# SHAPExplanation
# ---------------------------------------------------------------------------


class TestSHAPExplanation:
    def test_n_samples(self, shap_expl: SHAPExplanation) -> None:
        assert shap_expl.n_samples == 10

    def test_n_features(self, shap_expl: SHAPExplanation) -> None:
        assert shap_expl.n_features == 8

    def test_mean_abs_shap_shape(self, shap_expl: SHAPExplanation) -> None:
        m = shap_expl.mean_abs_shap()
        assert m.shape == (8,)
        assert m.dtype == np.float32

    def test_mean_abs_shap_values(self, shap_expl: SHAPExplanation) -> None:
        m = shap_expl.mean_abs_shap()
        expected = np.abs(shap_expl.shap_values).mean(axis=0)
        np.testing.assert_allclose(m, expected, atol=1e-6)

    def test_prediction_for_provided(self, shap_expl: SHAPExplanation) -> None:
        assert shap_expl.predictions is not None
        p = shap_expl.prediction_for(0)
        assert p == pytest.approx(float(shap_expl.predictions[0]))

    def test_prediction_for_approximated(self, shap_expl: SHAPExplanation) -> None:
        expl_no_preds = SHAPExplanation(
            shap_values=shap_expl.shap_values,
            expected_value=shap_expl.expected_value,
            feature_names=shap_expl.feature_names,
            feature_values=shap_expl.feature_values,
        )
        expected = float(shap_expl.expected_value + shap_expl.shap_values[2].sum())
        assert expl_no_preds.prediction_for(2) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# build_explanation
# ---------------------------------------------------------------------------


class TestBuildExplanation:
    def test_default_feature_names(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 8), dtype=np.float32)
        expl = build_explanation(shap_vals, 0.3, clinical_matrix)
        assert expl.feature_names == CLINICAL_FEATURE_NAMES[:8]

    def test_custom_feature_names(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 4), dtype=np.float32)
        names = ["A", "B", "C", "D"]
        expl = build_explanation(shap_vals, 0.3, clinical_matrix, feature_names=names)
        assert expl.feature_names == names

    def test_predictions_stored(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 8), dtype=np.float32)
        preds = np.ones(10, dtype=np.float32)
        expl = build_explanation(shap_vals, 0.3, clinical_matrix, predictions=preds)
        np.testing.assert_array_equal(expl.predictions, preds)

    def test_predictions_optional(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 8), dtype=np.float32)
        expl = build_explanation(shap_vals, 0.3, clinical_matrix)
        assert expl.predictions is None

    def test_event_name(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 8), dtype=np.float32)
        expl = build_explanation(shap_vals, 0.3, clinical_matrix, event_name="TRM")
        assert expl.event_name == "TRM"

    def test_float32_output(self, clinical_matrix: np.ndarray) -> None:
        shap_vals = np.zeros((10, 8), dtype=np.float64)
        expl = build_explanation(shap_vals, 0.3, clinical_matrix)
        assert expl.shap_values.dtype == np.float32


# ---------------------------------------------------------------------------
# build_clinical_predict_fn
# ---------------------------------------------------------------------------


class TestBuildClinicalPredictFn:
    def test_raises_on_missing_interaction(
        self, clinical_matrix: np.ndarray
    ) -> None:
        model = _NoSubModulesModel()
        with pytest.raises(AttributeError, match="interaction"):
            build_clinical_predict_fn(
                model,
                torch.randn(1, N_LOCI, EMB_DIM),
                torch.randn(1, N_LOCI, EMB_DIM),
            )

    def test_raises_on_missing_clinical_encoder(
        self, clinical_matrix: np.ndarray
    ) -> None:
        class _NoEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction = _TinyInteraction()

            def forward(self, *a: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1)

        with pytest.raises(AttributeError, match="clinical_encoder"):
            build_clinical_predict_fn(
                _NoEncoder(),
                torch.randn(1, N_LOCI, EMB_DIM),
                torch.randn(1, N_LOCI, EMB_DIM),
            )

    def test_returns_callable(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
    ) -> None:
        fn = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb)
        assert callable(fn)

    def test_output_shape(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        fn = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb, event_idx=0)
        scores = fn(clinical_matrix)
        assert scores.shape == (10,)

    def test_output_dtype(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        fn = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb)
        scores = fn(clinical_matrix)
        assert scores.dtype == np.float32

    def test_output_in_unit_interval(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        fn = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb)
        scores = fn(clinical_matrix)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_no_gradient_required(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """Predict function must run inside torch.no_grad()."""
        fn = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb)
        # Should not raise even when autograd is disabled
        with torch.no_grad():
            scores = fn(clinical_matrix)
        assert scores.shape == (10,)

    def test_different_event_idx(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        fn0 = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb, event_idx=0)
        fn1 = build_clinical_predict_fn(tiny_model, donor_emb, recip_emb, event_idx=1)
        s0 = fn0(clinical_matrix)
        s1 = fn1(clinical_matrix)
        # Different events should generally give different scores
        assert not np.allclose(s0, s1)

    def test_explicit_device_hint(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """Exercises _resolve_device with a non-None hint (line 363)."""
        fn = build_clinical_predict_fn(
            tiny_model, donor_emb, recip_emb, device="cpu"
        )
        scores = fn(clinical_matrix)
        assert scores.shape == (10,)

    def test_fallback_cif_without_cif_method(
        self,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """Exercises the else-branch when survival_head has no .cif method."""

        class _NoCifHead(nn.Module):
            NUM_EVENTS = 2
            TIME_BINS = 5

            def __init__(self, in_dim: int) -> None:
                super().__init__()
                self.fc = nn.Linear(in_dim, self.NUM_EVENTS * self.TIME_BINS)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x).view(-1, self.NUM_EVENTS, self.TIME_BINS)

        class _ModelNoCif(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction = _TinyInteraction()
                self.clinical_encoder = _TinyClinicalEncoder()
                combined = _TinyInteraction.OUT_DIM + _TinyClinicalEncoder.CLIN_DIM
                self.survival_head = _NoCifHead(combined)

        model = _ModelNoCif()
        fn = build_clinical_predict_fn(model, donor_emb, recip_emb)
        scores = fn(clinical_matrix)
        assert scores.shape == (10,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


# ---------------------------------------------------------------------------
# ClinicalDeepWrapper
# ---------------------------------------------------------------------------


class TestClinicalDeepWrapper:
    def test_from_model_returns_wrapper_and_tensor(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        assert isinstance(wrapper, ClinicalDeepWrapper)
        assert isinstance(bg, torch.Tensor)
        assert bg.shape == (10, 4)

    def test_forward_shape(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        cont_norm = bg[:3]
        out = wrapper(cont_norm)
        assert out.shape == (3, 1)

    def test_forward_values_in_unit_interval(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        out = wrapper(bg).detach().numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_from_model_raises_on_missing_attr(
        self,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        with pytest.raises(AttributeError):
            ClinicalDeepWrapper.from_model(
                _NoSubModulesModel(), donor_emb, recip_emb, clinical_matrix
            )

    def test_forward_without_cif_method(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """Exercises the else-branch in ClinicalDeepWrapper.forward."""

        class _NoCifHead(nn.Module):
            NUM_EVENTS = 2
            TIME_BINS = 5

            def __init__(self, in_dim: int) -> None:
                super().__init__()
                self.fc = nn.Linear(in_dim, self.NUM_EVENTS * self.TIME_BINS)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x).view(-1, self.NUM_EVENTS, self.TIME_BINS)

        class _ModelNoCif(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction = _TinyInteraction()
                self.clinical_encoder = _TinyClinicalEncoder()
                combined = _TinyInteraction.OUT_DIM + _TinyClinicalEncoder.CLIN_DIM
                self.survival_head = _NoCifHead(combined)

        model = _ModelNoCif()
        wrapper, bg = ClinicalDeepWrapper.from_model(model, donor_emb, recip_emb, clinical_matrix)
        out = wrapper(bg[:2])
        assert out.shape == (2, 1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# compute_shap_values (KernelExplainer)
# ---------------------------------------------------------------------------


class TestComputeShapValues:
    def test_raises_import_error_when_shap_missing(self) -> None:
        with patch.dict(sys.modules, {"shap": None}):
            with pytest.raises(ImportError, match="shap"):
                compute_shap_values(
                    lambda x: np.zeros(len(x), dtype=np.float32),
                    np.zeros((5, 8), dtype=np.float32),
                    np.zeros((2, 8), dtype=np.float32),
                )

    def test_returns_correct_shape(self) -> None:
        n_bg, n_exp, n_feat = 5, 3, 8
        expected = np.zeros((n_exp, n_feat), dtype=np.float32)
        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = expected
        mock_shap = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_expl

        with patch.dict(sys.modules, {"shap": mock_shap}):
            bg = np.zeros((n_bg, n_feat), dtype=np.float32)
            exp = np.zeros((n_exp, n_feat), dtype=np.float32)
            result = compute_shap_values(lambda x: x[:, 0], bg, exp)

        assert result.shape == (n_exp, n_feat)

    def test_passes_correct_args_to_explainer(self) -> None:
        predict_fn = lambda x: np.zeros(len(x), dtype=np.float32)  # noqa: E731
        bg = np.ones((4, 8), dtype=np.float32)
        exp_data = np.ones((2, 8), dtype=np.float32) * 2

        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = np.zeros((2, 8), dtype=np.float32)
        mock_shap = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_expl

        with patch.dict(sys.modules, {"shap": mock_shap}):
            compute_shap_values(predict_fn, bg, exp_data)

        mock_shap.KernelExplainer.assert_called_once_with(predict_fn, bg)
        mock_expl.shap_values.assert_called_once_with(exp_data)


# ---------------------------------------------------------------------------
# compute_shap_values_deep
# ---------------------------------------------------------------------------


class TestComputeShapValuesDeep:
    def test_raises_import_error_when_shap_missing(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        with patch.dict(sys.modules, {"shap": None}):
            with pytest.raises(ImportError, match="shap"):
                compute_shap_values_deep(wrapper, bg, bg[:2])

    def test_returns_correct_shapes(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        n_explain = 3
        exp_tensor = bg[:n_explain]

        shap_raw = np.zeros((n_explain, 4), dtype=np.float32)
        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = shap_raw
        mock_shap = MagicMock()
        mock_shap.DeepExplainer.return_value = mock_expl

        with patch.dict(sys.modules, {"shap": mock_shap}):
            sv, ev = compute_shap_values_deep(wrapper, bg, exp_tensor)

        assert sv.shape == (n_explain, 4)
        assert isinstance(ev, float)

    def test_handles_list_output(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """DeepExplainer may return a list; ensure it is handled."""
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        exp_tensor = bg[:2]
        shap_raw = [np.zeros((2, 4), dtype=np.float32)]

        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = shap_raw
        mock_shap = MagicMock()
        mock_shap.DeepExplainer.return_value = mock_expl

        with patch.dict(sys.modules, {"shap": mock_shap}):
            sv, _ = compute_shap_values_deep(wrapper, bg, exp_tensor)

        assert sv.shape == (2, 4)

    def test_handles_3d_array_output(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """Exercises the 3D squeeze branch (shap_arr[:, :, 0])."""
        wrapper, bg = ClinicalDeepWrapper.from_model(
            tiny_model, donor_emb, recip_emb, clinical_matrix
        )
        exp_tensor = bg[:2]
        # DeepExplainer returning (n, n_feat, 1) shaped array
        shap_raw = np.zeros((2, 4, 1), dtype=np.float32)

        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = shap_raw
        mock_shap = MagicMock()
        mock_shap.DeepExplainer.return_value = mock_expl

        with patch.dict(sys.modules, {"shap": mock_shap}):
            sv, _ = compute_shap_values_deep(wrapper, bg, exp_tensor)

        assert sv.shape == (2, 4)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestPlotBeeswarm:
    def test_returns_figure(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_beeswarm(shap_expl)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_max_display_respected(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_beeswarm(shap_expl, max_display=3)
        ax = fig.axes[0]
        # Y-ticks should have at most 3 labels
        assert len(ax.get_yticks()) <= 3
        plt.close("all")

    def test_existing_ax(self, shap_expl: SHAPExplanation) -> None:
        fig_ext, ax = plt.subplots()
        fig_ret = plot_beeswarm(shap_expl, ax=ax)
        assert fig_ret is fig_ext
        plt.close("all")

    def test_custom_title(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_beeswarm(shap_expl, title="Custom Title")
        ax = fig.axes[0]
        assert "Custom Title" in ax.get_title()
        plt.close("all")


class TestPlotWaterfall:
    def test_returns_figure(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_waterfall(shap_expl, sample_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_max_display_aggregates_other(self, shap_expl: SHAPExplanation) -> None:
        # With max_display=3, 5 features become "other features" bar
        fig = plot_waterfall(shap_expl, sample_idx=0, max_display=3)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("other" in lbl.lower() for lbl in labels)
        plt.close("all")

    def test_sample_idx(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_waterfall(shap_expl, sample_idx=5)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_existing_ax(self, shap_expl: SHAPExplanation) -> None:
        fig_ext, ax = plt.subplots()
        fig_ret = plot_waterfall(shap_expl, ax=ax)
        assert fig_ret is fig_ext
        plt.close("all")

    def test_no_other_when_enough_display(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_waterfall(shap_expl, sample_idx=0, max_display=20)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert not any("other" in lbl.lower() for lbl in labels)
        plt.close("all")


class TestPlotFeatureImportance:
    def test_returns_figure(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_feature_importance(shap_expl)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_max_display_respected(self, shap_expl: SHAPExplanation) -> None:
        fig = plot_feature_importance(shap_expl, max_display=4)
        ax = fig.axes[0]
        assert len(ax.get_yticks()) <= 4
        plt.close("all")

    def test_existing_ax(self, shap_expl: SHAPExplanation) -> None:
        fig_ext, ax = plt.subplots()
        fig_ret = plot_feature_importance(shap_expl, ax=ax)
        assert fig_ret is fig_ext
        plt.close("all")


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    def test_writes_pdf_and_png(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        paths = save_figure(fig, tmp_path / "out")
        plt.close("all")
        exts = {p.suffix for p in paths}
        assert ".pdf" in exts
        assert ".png" in exts
        assert all(p.exists() for p in paths)

    def test_custom_formats(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        paths = save_figure(fig, tmp_path / "out", formats=["png"])
        plt.close("all")
        assert len(paths) == 1
        assert paths[0].suffix == ".png"

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        out = tmp_path / "a" / "b" / "fig"
        save_figure(fig, out, formats=["png"])
        plt.close("all")
        assert out.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# generate_shap_report
# ---------------------------------------------------------------------------


class TestGenerateShapReport:
    """End-to-end report generation using a mock KernelExplainer."""

    def _mock_shap(self, n_explain: int, n_feat: int) -> MagicMock:
        mock_expl = MagicMock()
        mock_expl.shap_values.return_value = np.zeros(
            (n_explain, n_feat), dtype=np.float32
        )
        mock_shap = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_expl
        return mock_shap

    def test_returns_correct_keys(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        mock_shap = self._mock_shap(10, 8)
        with patch.dict(sys.modules, {"shap": mock_shap}):
            figs = generate_shap_report(
                tiny_model, donor_emb, recip_emb,
                clinical_matrix[:5], clinical_matrix,
                event_name="GvHD",
                waterfall_indices=[0, 1],
            )
        plt.close("all")
        assert "beeswarm" in figs
        assert "feature_importance" in figs
        assert "waterfall_0" in figs
        assert "waterfall_1" in figs

    def test_out_of_range_waterfall_skipped(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        mock_shap = self._mock_shap(10, 8)
        with patch.dict(sys.modules, {"shap": mock_shap}):
            figs = generate_shap_report(
                tiny_model, donor_emb, recip_emb,
                clinical_matrix[:5], clinical_matrix,
                waterfall_indices=[0, 999],  # 999 is out of range
            )
        plt.close("all")
        assert "waterfall_0" in figs
        assert "waterfall_999" not in figs

    def test_saves_files_when_out_dir_given(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
        tmp_path: Path,
    ) -> None:
        mock_shap = self._mock_shap(10, 8)
        with patch.dict(sys.modules, {"shap": mock_shap}):
            generate_shap_report(
                tiny_model, donor_emb, recip_emb,
                clinical_matrix[:5], clinical_matrix,
                waterfall_indices=[0],
                out_dir=tmp_path,
            )
        plt.close("all")
        # 3 figures × 2 formats = 6 files minimum
        assert len(list(tmp_path.iterdir())) >= 6

    def test_default_waterfall_indices_is_zero(
        self,
        tiny_model: _TinyModel,
        donor_emb: torch.Tensor,
        recip_emb: torch.Tensor,
        clinical_matrix: np.ndarray,
    ) -> None:
        """When waterfall_indices=None, defaults to [0] — exercises line 1245."""
        mock_shap = self._mock_shap(10, 8)
        with patch.dict(sys.modules, {"shap": mock_shap}):
            figs = generate_shap_report(
                tiny_model, donor_emb, recip_emb,
                clinical_matrix[:5], clinical_matrix,
                # waterfall_indices intentionally omitted
            )
        plt.close("all")
        assert "waterfall_0" in figs


# ---------------------------------------------------------------------------
# Additional edge-case coverage for internal helpers
# ---------------------------------------------------------------------------


class TestBeeswarmConstantFeatureColour:
    """Exercises _normalise_for_colour with constant feature values via beeswarm."""

    def test_beeswarm_with_constant_feature(self) -> None:
        # Create an explanation where one feature has constant value (triggers
        # the `hi == lo` branch in _normalise_for_colour)
        shap_vals = np.random.default_rng(5).standard_normal((8, 8)).astype(np.float32)
        feat_vals = np.ones((8, 8), dtype=np.float32)  # all identical → constant
        expl = SHAPExplanation(
            shap_values=shap_vals,
            expected_value=0.3,
            feature_names=CLINICAL_FEATURE_NAMES,
            feature_values=feat_vals,
        )
        fig = plot_beeswarm(expl)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestBeeswarmSingleSample:
    """Exercises _beeswarm_y_positions with n=1 (early-return branch)."""

    def test_beeswarm_single_sample(self) -> None:
        shap_vals = np.array([[0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.08, -0.04]],
                             dtype=np.float32)
        feat_vals = np.zeros((1, 8), dtype=np.float32)
        expl = SHAPExplanation(
            shap_values=shap_vals,
            expected_value=0.4,
            feature_names=CLINICAL_FEATURE_NAMES,
            feature_values=feat_vals,
        )
        fig = plot_beeswarm(expl)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
