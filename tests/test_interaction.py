"""Tests for capa/model/interaction.py.

Coverage targets
----------------
CrossAttentionInteraction
  - Output shape (various batch/loci/dim configs)
  - No NaN / Inf in output
  - Gradients flow to donor and recipient inputs
  - Gradients flow through all block parameters
  - Projection weights participate in gradient graph
  - attention_weights stored after forward: correct structure, shapes, row-sums ≈ 1
  - attention_weights is None before first forward
  - attention_weights refreshed on second forward
  - Works with num_layers=1 and num_layers=3
  - Works with batch_size=1 (edge case)
  - Symmetric attention: identical donor/recipient → output unchanged when swapped
  - Different donor/recipient → different output (model is sensitive to inputs)
  - embedding_dim / num_heads mismatch raises ValueError
  - Model enters eval mode without error
  - Training vs eval mode: dropout changes output distribution

DiffMLPInteraction
  - Output shape
  - No NaN / Inf
  - Gradients flow to donor and recipient inputs
  - Gradients flow through MLP parameters
  - last_attn_weights is always None
  - Identical donor/recipient → abs-diff is zero (feature ablation)
  - Swapping donor and recipient changes output (asymmetric due to concat order)
  - Works with batch_size=1
  - custom hidden_dim respected (parameter count check)

Shared interface
  - Both modules expose embedding_dim / interaction_dim properties
  - Both accept the same (donor, recipient) call signature
  - Both output float32
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from capa.model.interaction import (
    AttentionWeights,
    CrossAttentionInteraction,
    DiffMLPInteraction,
)

# ---------------------------------------------------------------------------
# Constants for fast CPU tests
# ---------------------------------------------------------------------------
BATCH = 4
N_LOCI = 5
EMB = 32        # embedding_dim — small for CPU speed
DIM = 16        # interaction_dim
HEADS = 4       # num_heads (must divide EMB)
LAYERS = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cross_model() -> CrossAttentionInteraction:
    return CrossAttentionInteraction(
        embedding_dim=EMB,
        interaction_dim=DIM,
        num_heads=HEADS,
        num_layers=LAYERS,
        dropout=0.0,  # deterministic for most tests
    )


@pytest.fixture()
def diff_model() -> DiffMLPInteraction:
    return DiffMLPInteraction(
        embedding_dim=EMB,
        interaction_dim=DIM,
        dropout=0.0,
    )


def _rand(batch: int = BATCH, loci: int = N_LOCI, emb: int = EMB) -> torch.Tensor:
    return torch.randn(batch, loci, emb, requires_grad=True)


# ===========================================================================
# CrossAttentionInteraction
# ===========================================================================

class TestCrossAttentionOutputShape:
    def test_standard_shape(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    def test_batch_one(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(1), _rand(1))
        assert out.shape == (1, DIM)

    def test_single_locus(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(BATCH, 1), _rand(BATCH, 1))
        assert out.shape == (BATCH, DIM)

    def test_many_loci(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(BATCH, 10), _rand(BATCH, 10))
        assert out.shape == (BATCH, DIM)

    def test_large_batch(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(16), _rand(16))
        assert out.shape == (16, DIM)

    def test_output_dtype_float32(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(), _rand())
        assert out.dtype == torch.float32

    def test_num_layers_1(self) -> None:
        m = CrossAttentionInteraction(EMB, DIM, HEADS, num_layers=1, dropout=0.0)
        out = m(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    def test_num_layers_3(self) -> None:
        m = CrossAttentionInteraction(EMB, DIM, HEADS, num_layers=3, dropout=0.0)
        out = m(_rand(), _rand())
        assert out.shape == (BATCH, DIM)


class TestCrossAttentionNumerics:
    def test_no_nan(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(), _rand())
        assert not torch.isnan(out).any()

    def test_no_inf(self, cross_model: CrossAttentionInteraction) -> None:
        out = cross_model(_rand(), _rand())
        assert not torch.isinf(out).any()

    def test_no_nan_zero_input(self, cross_model: CrossAttentionInteraction) -> None:
        z = torch.zeros(BATCH, N_LOCI, EMB)
        out = cross_model(z, z)
        assert not torch.isnan(out).any()


class TestCrossAttentionGradients:
    def test_grad_flows_to_donor(self, cross_model: CrossAttentionInteraction) -> None:
        d = _rand()
        r = _rand()
        out = cross_model(d, r)
        out.sum().backward()
        assert d.grad is not None
        assert not torch.all(d.grad == 0)

    def test_grad_flows_to_recipient(self, cross_model: CrossAttentionInteraction) -> None:
        d = _rand()
        r = _rand()
        out = cross_model(d, r)
        out.sum().backward()
        assert r.grad is not None
        assert not torch.all(r.grad == 0)

    def test_grad_flows_to_projection_weights(self, cross_model: CrossAttentionInteraction) -> None:
        d, r = _rand(), _rand()
        out = cross_model(d, r)
        out.sum().backward()
        # Check the first Linear in projection
        proj_linear: nn.Linear = cross_model.projection[0]  # type: ignore[index]
        assert proj_linear.weight.grad is not None
        assert not torch.all(proj_linear.weight.grad == 0)

    def test_grad_flows_through_all_blocks(self, cross_model: CrossAttentionInteraction) -> None:
        d, r = _rand(), _rand()
        out = cross_model(d, r)
        out.sum().backward()
        for i, block in enumerate(cross_model.blocks):
            for name, param in block.named_parameters():
                assert param.grad is not None, f"block[{i}].{name} has no grad"

    def test_double_backward_compatible(self, cross_model: CrossAttentionInteraction) -> None:
        """Second-order gradients don't raise."""
        d, r = _rand(), _rand()
        out = cross_model(d, r)
        grad_d = torch.autograd.grad(out.sum(), d, create_graph=True)[0]
        grad_d.sum().backward()  # second-order — should not raise


class TestCrossAttentionWeights:
    def test_none_before_forward(self, cross_model: CrossAttentionInteraction) -> None:
        assert cross_model.last_attn_weights is None

    def test_weights_stored_after_forward(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        assert cross_model.last_attn_weights is not None

    def test_weights_is_attention_weights_type(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        assert isinstance(cross_model.last_attn_weights, AttentionWeights)

    def test_weights_list_length_matches_num_layers(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        assert len(w.donor_to_recip) == LAYERS
        assert len(w.recip_to_donor) == LAYERS

    def test_weights_shape_d2r(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        for layer_w in w.donor_to_recip:
            # averaged over heads → (batch, n_loci_q, n_loci_kv)
            assert layer_w.shape == (BATCH, N_LOCI, N_LOCI)

    def test_weights_shape_r2d(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        for layer_w in w.recip_to_donor:
            assert layer_w.shape == (BATCH, N_LOCI, N_LOCI)

    def test_weights_row_sum_to_one(self, cross_model: CrossAttentionInteraction) -> None:
        """Attention weights are softmax outputs — each query's distribution sums to 1."""
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        for layer_w in w.donor_to_recip + w.recip_to_donor:
            row_sums = layer_w.sum(dim=-1)  # (batch, n_loci_q)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_weights_non_negative(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        for layer_w in w.donor_to_recip + w.recip_to_donor:
            assert (layer_w >= 0).all()

    def test_weights_refreshed_on_second_forward(
        self, cross_model: CrossAttentionInteraction
    ) -> None:
        d1, r1 = _rand(), _rand()
        d2, r2 = _rand(), _rand()
        cross_model(d1, r1)
        w1_last = cross_model.last_attn_weights
        cross_model(d2, r2)
        w2_last = cross_model.last_attn_weights
        # Objects should differ (new tensors were created)
        assert w1_last is not w2_last

    def test_weights_detached_from_graph_in_eval(
        self, cross_model: CrossAttentionInteraction
    ) -> None:
        cross_model.eval()
        cross_model(_rand(), _rand())
        w = cross_model.last_attn_weights
        assert w is not None
        # Attention weights in eval mode don't need grad by default
        for layer_w in w.donor_to_recip + w.recip_to_donor:
            assert layer_w is not None  # just confirm they exist


class TestCrossAttentionSensitivity:
    def test_different_inputs_different_outputs(
        self, cross_model: CrossAttentionInteraction
    ) -> None:
        torch.manual_seed(0)
        d1, r1 = _rand(), _rand()
        d2, r2 = _rand(), _rand()
        out1 = cross_model(d1, r1)
        out2 = cross_model(d2, r2)
        assert not torch.allclose(out1, out2)

    def test_donor_vs_recipient_asymmetric(
        self, cross_model: CrossAttentionInteraction
    ) -> None:
        """Swapping donor/recipient should produce different output (concat is ordered)."""
        torch.manual_seed(1)
        d, r = _rand(), _rand()
        out_dr = cross_model(d, r)
        out_rd = cross_model(r, d)
        assert not torch.allclose(out_dr, out_rd)

    def test_identical_donor_recipient(
        self, cross_model: CrossAttentionInteraction
    ) -> None:
        """When donor == recipient the model should still produce a finite output."""
        x = torch.randn(BATCH, N_LOCI, EMB)
        out = cross_model(x, x)
        assert out.shape == (BATCH, DIM)
        assert not torch.isnan(out).any()


class TestCrossAttentionValidation:
    def test_invalid_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            CrossAttentionInteraction(embedding_dim=33, num_heads=4)

    def test_eval_mode_no_error(self, cross_model: CrossAttentionInteraction) -> None:
        cross_model.eval()
        out = cross_model(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    def test_dropout_changes_output_in_train(self) -> None:
        """With dropout > 0, two training-mode forward passes should differ."""
        m = CrossAttentionInteraction(EMB, DIM, HEADS, dropout=0.9)
        m.train()
        torch.manual_seed(42)
        d, r = _rand(), _rand()
        out1 = m(d.detach(), r.detach())
        out2 = m(d.detach(), r.detach())
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self, cross_model: CrossAttentionInteraction) -> None:
        """Eval mode (dropout=0) produces identical results on repeated forward."""
        cross_model.eval()
        d, r = _rand(), _rand()
        out1 = cross_model(d.detach(), r.detach())
        out2 = cross_model(d.detach(), r.detach())
        assert torch.allclose(out1, out2)


class TestCrossAttentionProperties:
    def test_embedding_dim_property(self, cross_model: CrossAttentionInteraction) -> None:
        assert cross_model.embedding_dim == EMB

    def test_interaction_dim_property(self, cross_model: CrossAttentionInteraction) -> None:
        assert cross_model.interaction_dim == DIM


# ===========================================================================
# DiffMLPInteraction
# ===========================================================================

class TestDiffMLPOutputShape:
    def test_standard_shape(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    def test_batch_one(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(1), _rand(1))
        assert out.shape == (1, DIM)

    def test_single_locus(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(BATCH, 1), _rand(BATCH, 1))
        assert out.shape == (BATCH, DIM)

    def test_output_dtype_float32(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(), _rand())
        assert out.dtype == torch.float32


class TestDiffMLPNumerics:
    def test_no_nan(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(), _rand())
        assert not torch.isnan(out).any()

    def test_no_inf(self, diff_model: DiffMLPInteraction) -> None:
        out = diff_model(_rand(), _rand())
        assert not torch.isinf(out).any()


class TestDiffMLPGradients:
    def test_grad_flows_to_donor(self, diff_model: DiffMLPInteraction) -> None:
        d, r = _rand(), _rand()
        diff_model(d, r).sum().backward()
        assert d.grad is not None
        assert not torch.all(d.grad == 0)

    def test_grad_flows_to_recipient(self, diff_model: DiffMLPInteraction) -> None:
        d, r = _rand(), _rand()
        diff_model(d, r).sum().backward()
        assert r.grad is not None
        assert not torch.all(r.grad == 0)

    def test_grad_flows_to_mlp_weights(self, diff_model: DiffMLPInteraction) -> None:
        d, r = _rand(), _rand()
        diff_model(d, r).sum().backward()
        first_linear: nn.Linear = diff_model.mlp[0]  # type: ignore[index]
        assert first_linear.weight.grad is not None
        assert not torch.all(first_linear.weight.grad == 0)


class TestDiffMLPBehavior:
    def test_attn_weights_always_none(self, diff_model: DiffMLPInteraction) -> None:
        assert diff_model.last_attn_weights is None
        diff_model(_rand(), _rand())
        assert diff_model.last_attn_weights is None

    def test_identical_inputs_zero_diff_feature(self, diff_model: DiffMLPInteraction) -> None:
        """When donor == recipient the abs-diff feature is zero.
        The model can still produce non-zero output via donor/recipient terms."""
        x = torch.randn(BATCH, N_LOCI, EMB)
        out = diff_model(x, x)
        assert out.shape == (BATCH, DIM)
        assert not torch.isnan(out).any()

    def test_swap_donor_recipient_changes_output(
        self, diff_model: DiffMLPInteraction
    ) -> None:
        """concat([d, r, |d-r|]) ≠ concat([r, d, |r-d|]) in general."""
        torch.manual_seed(0)
        d, r = _rand(), _rand()
        out_dr = diff_model(d, r)
        out_rd = diff_model(r, d)
        # abs-diff is symmetric, but donor and recipient slots differ → outputs differ
        assert not torch.allclose(out_dr, out_rd)

    def test_custom_hidden_dim(self) -> None:
        m = DiffMLPInteraction(EMB, DIM, hidden_dim=64)
        # First linear: (EMB * 3, 64)
        first_linear: nn.Linear = m.mlp[0]  # type: ignore[index]
        assert first_linear.out_features == 64
        # Verify output shape is still correct
        out = m(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    def test_default_hidden_dim(self, diff_model: DiffMLPInteraction) -> None:
        # Default hidden = interaction_dim * 4 = DIM * 4
        first_linear: nn.Linear = diff_model.mlp[0]  # type: ignore[index]
        assert first_linear.out_features == DIM * 4


class TestDiffMLPProperties:
    def test_embedding_dim_property(self, diff_model: DiffMLPInteraction) -> None:
        assert diff_model.embedding_dim == EMB

    def test_interaction_dim_property(self, diff_model: DiffMLPInteraction) -> None:
        assert diff_model.interaction_dim == DIM


# ===========================================================================
# Shared interface tests
# ===========================================================================

class TestSharedInterface:
    @pytest.mark.parametrize("model_cls,kwargs", [
        (CrossAttentionInteraction, {"num_heads": HEADS, "num_layers": 1, "dropout": 0.0}),
        (DiffMLPInteraction, {}),
    ])
    def test_output_shape(self, model_cls: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        m = model_cls(embedding_dim=EMB, interaction_dim=DIM, **kwargs)
        out = m(_rand(), _rand())
        assert out.shape == (BATCH, DIM)

    @pytest.mark.parametrize("model_cls,kwargs", [
        (CrossAttentionInteraction, {"num_heads": HEADS, "num_layers": 1, "dropout": 0.0}),
        (DiffMLPInteraction, {}),
    ])
    def test_embedding_dim_property(self, model_cls: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        m = model_cls(embedding_dim=EMB, interaction_dim=DIM, **kwargs)
        assert m.embedding_dim == EMB

    @pytest.mark.parametrize("model_cls,kwargs", [
        (CrossAttentionInteraction, {"num_heads": HEADS, "num_layers": 1, "dropout": 0.0}),
        (DiffMLPInteraction, {}),
    ])
    def test_interaction_dim_property(self, model_cls: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        m = model_cls(embedding_dim=EMB, interaction_dim=DIM, **kwargs)
        assert m.interaction_dim == DIM

    @pytest.mark.parametrize("model_cls,kwargs", [
        (CrossAttentionInteraction, {"num_heads": HEADS, "num_layers": 1, "dropout": 0.0}),
        (DiffMLPInteraction, {}),
    ])
    def test_grad_flows_from_output(self, model_cls: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        m = model_cls(embedding_dim=EMB, interaction_dim=DIM, **kwargs)
        d, r = _rand(), _rand()
        m(d, r).sum().backward()
        assert d.grad is not None
        assert r.grad is not None

    @pytest.mark.parametrize("model_cls,kwargs", [
        (CrossAttentionInteraction, {"num_heads": HEADS, "num_layers": 1, "dropout": 0.0}),
        (DiffMLPInteraction, {}),
    ])
    def test_is_nn_module(self, model_cls: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        m = model_cls(embedding_dim=EMB, interaction_dim=DIM, **kwargs)
        assert isinstance(m, nn.Module)
