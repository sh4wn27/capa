"""Tests for capa/interpret/attention_maps.py.

Coverage targets
----------------
TopPair
  - __repr__ format
  - Fields stored correctly

AttentionWeightSet
  - n_layers property
  - get_layer (positive and negative index)
  - mean_across_layers shape and values
  - top_k_pairs: correct rank, row, col, weight ordering
  - top_k_pairs with k > n cells (clamped)
  - top_k_pairs with custom row/col labels
  - to_dict keys and shapes

PopulationWeights
  - top_k_pairs d2r and r2d
  - to_dict keys

aggregate_population_weights
  - shape and dtype of outputs
  - n_subjects matches input
  - raises on empty list
  - layer=-1 vs layer=0

collect_population_weights
  - accumulates correct number of subjects
  - respects max_subjects cap

extract_attention_weights
  - returns one AttentionWeightSet per batch element
  - infers loci names when not provided
  - raises ValueError when model has no attention weights

plot_attention_heatmap
  - returns a Figure
  - works with top_k=0 and top_k=3
  - works with annotate_values=False
  - works with existing ax

plot_both_directions
  - returns a Figure with 2 axes

plot_population_heatmap
  - returns a Figure for d2r and r2d
  - works with existing ax

plot_population_both_directions
  - returns a Figure

save_figure
  - writes PDF and PNG files to disk
  - accepts custom formats list

generate_patient_map
  - returns dict with keys "d2r", "r2d", "both"
  - saves files when out_dir is provided

generate_population_map
  - returns dict with population keys
  - raises ValueError on empty iterator
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt

from capa.interpret.attention_maps import (
    AttentionWeightSet,
    PopulationWeights,
    TopPair,
    aggregate_population_weights,
    collect_population_weights,
    extract_attention_weights,
    generate_patient_map,
    generate_population_map,
    plot_attention_heatmap,
    plot_both_directions,
    plot_population_both_directions,
    plot_population_heatmap,
    save_figure,
)
from capa.model.interaction import AttentionWeights, CrossAttentionInteraction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LOCI = ["A", "B", "DRB1"]
N = len(LOCI)


def _rand_weights(n: int = N) -> np.ndarray:
    w = np.random.default_rng(0).random((n, n)).astype(np.float32)
    # Row-normalise so rows sum to 1 (mimics real softmax output)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture()
def single_layer_set() -> AttentionWeightSet:
    d2r = _rand_weights()
    r2d = _rand_weights()
    return AttentionWeightSet(
        donor_to_recip=[d2r],
        recip_to_donor=[r2d],
        loci=LOCI,
        patient_id="p001",
    )


@pytest.fixture()
def multi_layer_set() -> AttentionWeightSet:
    rng = np.random.default_rng(42)
    d2r_layers = [rng.random((N, N)).astype(np.float32) for _ in range(3)]
    r2d_layers = [rng.random((N, N)).astype(np.float32) for _ in range(3)]
    return AttentionWeightSet(
        donor_to_recip=d2r_layers,
        recip_to_donor=r2d_layers,
        loci=LOCI,
    )


@pytest.fixture()
def population_weights() -> PopulationWeights:
    rng = np.random.default_rng(7)
    return PopulationWeights(
        mean_d2r=rng.random((N, N)).astype(np.float32),
        std_d2r=rng.random((N, N)).astype(np.float32) * 0.1,
        mean_r2d=rng.random((N, N)).astype(np.float32),
        std_r2d=rng.random((N, N)).astype(np.float32) * 0.1,
        loci=LOCI,
        n_subjects=10,
    )


# Minimal model that stores AttentionWeights compatible with extract_attention_weights
class _MockInteraction(nn.Module):
    def __init__(self, n_loci: int, out_dim: int = 8) -> None:
        super().__init__()
        self.last_attn_weights: AttentionWeights | None = None
        self._n_loci = n_loci
        self._out_dim = out_dim
        self.linear = nn.Linear(1, out_dim)

    def forward(self, donor: torch.Tensor, recip: torch.Tensor) -> torch.Tensor:
        B, L, _ = donor.shape
        rng = torch.rand(B, L, L)
        rng_r = torch.rand(B, L, L)
        self.last_attn_weights = AttentionWeights(
            donor_to_recip=[rng],
            recip_to_donor=[rng_r],
        )
        return self.linear(torch.ones(B, 1))


class _MockModel(nn.Module):
    def __init__(self, n_loci: int = N) -> None:
        super().__init__()
        self.interaction = _MockInteraction(n_loci)

    def get_attention_weights(self) -> AttentionWeights | None:
        return self.interaction.last_attn_weights

    def forward(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        return self.interaction(donor, recip)


class _NoAttnModel(nn.Module):
    """A model that exposes no attention weights."""

    def forward(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(donor.shape[0], 4)


class _InteractionFallbackModel(nn.Module):
    """Model without get_attention_weights but with interaction.last_attn_weights.

    Exercises the fallback branch in extract_attention_weights (line 364).
    """

    def __init__(self, n_loci: int = N) -> None:
        super().__init__()
        self._n_loci = n_loci
        self.interaction = _MockInteraction(n_loci)

    # Deliberately does NOT define get_attention_weights

    def forward(
        self,
        donor: torch.Tensor,
        recip: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        return self.interaction(donor, recip)


# ---------------------------------------------------------------------------
# TopPair
# ---------------------------------------------------------------------------


class TestTopPair:
    def test_repr(self) -> None:
        tp = TopPair(rank=1, row=0, col=1, donor_locus="A", recipient_locus="B", weight=0.5)
        r = repr(tp)
        assert "rank=1" in r
        assert "A→B" in r
        assert "w=0.5" in r

    def test_fields(self) -> None:
        tp = TopPair(rank=2, row=1, col=2, donor_locus="B", recipient_locus="DRB1", weight=0.25)
        assert tp.rank == 2
        assert tp.row == 1
        assert tp.col == 2
        assert tp.weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# AttentionWeightSet
# ---------------------------------------------------------------------------


class TestAttentionWeightSet:
    def test_n_layers(self, multi_layer_set: AttentionWeightSet) -> None:
        assert multi_layer_set.n_layers == 3

    def test_get_layer_last(self, multi_layer_set: AttentionWeightSet) -> None:
        d2r, r2d = multi_layer_set.get_layer(-1)
        np.testing.assert_array_equal(d2r, multi_layer_set.donor_to_recip[-1])

    def test_get_layer_first(self, multi_layer_set: AttentionWeightSet) -> None:
        d2r, r2d = multi_layer_set.get_layer(0)
        np.testing.assert_array_equal(d2r, multi_layer_set.donor_to_recip[0])

    def test_mean_across_layers_shape(self, multi_layer_set: AttentionWeightSet) -> None:
        d2r_mean, r2d_mean = multi_layer_set.mean_across_layers()
        assert d2r_mean.shape == (N, N)
        assert r2d_mean.shape == (N, N)
        assert d2r_mean.dtype == np.float32

    def test_mean_across_layers_values(self, multi_layer_set: AttentionWeightSet) -> None:
        d2r_mean, _ = multi_layer_set.mean_across_layers()
        expected = np.mean(
            np.stack(multi_layer_set.donor_to_recip, axis=0), axis=0
        ).astype(np.float32)
        np.testing.assert_allclose(d2r_mean, expected, atol=1e-6)

    def test_top_k_pairs_ordering(self, single_layer_set: AttentionWeightSet) -> None:
        weights = single_layer_set.donor_to_recip[0]
        pairs = single_layer_set.top_k_pairs(weights, k=3)
        assert len(pairs) == 3
        assert pairs[0].rank == 1
        assert pairs[0].weight >= pairs[1].weight >= pairs[2].weight

    def test_top_k_pairs_custom_labels(self, single_layer_set: AttentionWeightSet) -> None:
        weights = single_layer_set.donor_to_recip[0]
        row_labels = ["D_A", "D_B", "D_DRB1"]
        col_labels = ["R_A", "R_B", "R_DRB1"]
        pairs = single_layer_set.top_k_pairs(weights, k=2, row_labels=row_labels, col_labels=col_labels)
        for p in pairs:
            assert p.donor_locus.startswith("D_")
            assert p.recipient_locus.startswith("R_")

    def test_top_k_clamped(self, single_layer_set: AttentionWeightSet) -> None:
        weights = single_layer_set.donor_to_recip[0]
        # Request more pairs than cells — should return N*N at most
        pairs = single_layer_set.top_k_pairs(weights, k=N * N + 10)
        assert len(pairs) == N * N

    def test_to_dict_keys(self, single_layer_set: AttentionWeightSet) -> None:
        d = single_layer_set.to_dict()
        for key in ("patient_id", "loci", "n_layers", "last_layer_d2r", "last_layer_r2d",
                    "mean_d2r", "mean_r2d"):
            assert key in d

    def test_to_dict_shapes(self, single_layer_set: AttentionWeightSet) -> None:
        d = single_layer_set.to_dict()
        assert len(d["last_layer_d2r"]) == N
        assert len(d["last_layer_d2r"][0]) == N


# ---------------------------------------------------------------------------
# PopulationWeights
# ---------------------------------------------------------------------------


class TestPopulationWeights:
    def test_top_k_d2r(self, population_weights: PopulationWeights) -> None:
        pairs = population_weights.top_k_pairs(direction="d2r", k=2)
        assert len(pairs) == 2
        assert pairs[0].weight >= pairs[1].weight

    def test_top_k_r2d(self, population_weights: PopulationWeights) -> None:
        pairs = population_weights.top_k_pairs(direction="r2d", k=2)
        assert len(pairs) == 2
        # Values come from mean_r2d
        assert all(p.weight >= 0 for p in pairs)

    def test_to_dict_keys(self, population_weights: PopulationWeights) -> None:
        d = population_weights.to_dict()
        for key in ("n_subjects", "loci", "mean_d2r", "std_d2r", "mean_r2d", "std_r2d"):
            assert key in d


# ---------------------------------------------------------------------------
# aggregate_population_weights
# ---------------------------------------------------------------------------


class TestAggregatePopulationWeights:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_population_weights([])

    def test_output_shapes(self, single_layer_set: AttentionWeightSet) -> None:
        sets = [single_layer_set, single_layer_set]
        pop = aggregate_population_weights(sets)
        assert pop.mean_d2r.shape == (N, N)
        assert pop.std_d2r.shape == (N, N)
        assert pop.n_subjects == 2

    def test_dtype_float32(self, single_layer_set: AttentionWeightSet) -> None:
        pop = aggregate_population_weights([single_layer_set])
        assert pop.mean_d2r.dtype == np.float32

    def test_std_zero_for_single_subject(self, single_layer_set: AttentionWeightSet) -> None:
        pop = aggregate_population_weights([single_layer_set])
        np.testing.assert_allclose(pop.std_d2r, 0.0, atol=1e-6)

    def test_mean_is_average(self, single_layer_set: AttentionWeightSet) -> None:
        rng = np.random.default_rng(1)
        w1 = rng.random((N, N)).astype(np.float32)
        w2 = rng.random((N, N)).astype(np.float32)
        s1 = AttentionWeightSet(donor_to_recip=[w1], recip_to_donor=[w1], loci=LOCI)
        s2 = AttentionWeightSet(donor_to_recip=[w2], recip_to_donor=[w2], loci=LOCI)
        pop = aggregate_population_weights([s1, s2])
        np.testing.assert_allclose(pop.mean_d2r, (w1 + w2) / 2, atol=1e-6)

    def test_layer_zero(self, multi_layer_set: AttentionWeightSet) -> None:
        pop = aggregate_population_weights([multi_layer_set, multi_layer_set], layer=0)
        expected_mean = multi_layer_set.donor_to_recip[0]
        np.testing.assert_allclose(pop.mean_d2r, expected_mean, atol=1e-6)


# ---------------------------------------------------------------------------
# collect_population_weights
# ---------------------------------------------------------------------------


class TestCollectPopulationWeights:
    def _make_batch(self) -> dict[str, torch.Tensor]:
        return {
            "donor_embeddings": torch.randn(2, N, 8),
            "recipient_embeddings": torch.randn(2, N, 8),
            "clinical_features": torch.randn(2, 4),
        }

    def test_collects_all_subjects(self) -> None:
        model = _MockModel(n_loci=N)
        batches = [self._make_batch(), self._make_batch()]  # 2 batches × 2 subjects = 4
        sets = collect_population_weights(model, batches, LOCI)
        assert len(sets) == 4

    def test_respects_max_subjects(self) -> None:
        model = _MockModel(n_loci=N)
        batches = [self._make_batch() for _ in range(10)]  # up to 20 subjects
        sets = collect_population_weights(model, batches, LOCI, max_subjects=3)
        assert len(sets) == 3


# ---------------------------------------------------------------------------
# extract_attention_weights
# ---------------------------------------------------------------------------


class TestExtractAttentionWeights:
    def test_returns_one_per_sample(self) -> None:
        model = _MockModel()
        donor = torch.randn(3, N, 8)
        recip = torch.randn(3, N, 8)
        clinical = torch.randn(3, 4)
        sets = extract_attention_weights(model, donor, recip, clinical, loci=LOCI)
        assert len(sets) == 3

    def test_inferred_loci_names(self) -> None:
        model = _MockModel()
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        sets = extract_attention_weights(model, donor, recip, clinical)
        assert sets[0].loci == [f"locus_{i}" for i in range(N)]

    def test_custom_loci_preserved(self) -> None:
        model = _MockModel()
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        sets = extract_attention_weights(model, donor, recip, clinical, loci=LOCI)
        assert sets[0].loci == LOCI

    def test_raises_for_no_attention_model(self) -> None:
        model = _NoAttnModel()
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        with pytest.raises(ValueError, match="attention weights"):
            extract_attention_weights(model, donor, recip, clinical, loci=LOCI)

    def test_fallback_via_interaction_attribute(self) -> None:
        """Exercises the fallback path when get_attention_weights is absent."""
        model = _InteractionFallbackModel(n_loci=N)
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        sets = extract_attention_weights(model, donor, recip, clinical, loci=LOCI)
        assert len(sets) == 1
        d2r, _ = sets[0].get_layer(-1)
        assert d2r.shape == (N, N)

    def test_output_shapes(self) -> None:
        model = _MockModel()
        B = 2
        donor = torch.randn(B, N, 8)
        recip = torch.randn(B, N, 8)
        clinical = torch.randn(B, 4)
        sets = extract_attention_weights(model, donor, recip, clinical, loci=LOCI)
        for ws in sets:
            d2r, r2d = ws.get_layer(-1)
            assert d2r.shape == (N, N)
            assert r2d.shape == (N, N)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


@pytest.fixture()
def weights_2d() -> np.ndarray:
    w = np.random.default_rng(3).random((N, N)).astype(np.float32)
    return w / w.sum(axis=1, keepdims=True)


class TestPlotAttentionHeatmap:
    def test_returns_figure(self, weights_2d: np.ndarray) -> None:
        fig = plot_attention_heatmap(weights_2d, LOCI, LOCI)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_top_k_zero(self, weights_2d: np.ndarray) -> None:
        fig = plot_attention_heatmap(weights_2d, LOCI, LOCI, top_k=0)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_no_annotate(self, weights_2d: np.ndarray) -> None:
        fig = plot_attention_heatmap(weights_2d, LOCI, LOCI, annotate_values=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_existing_ax(self, weights_2d: np.ndarray) -> None:
        fig_ext, ax = plt.subplots()
        fig_ret = plot_attention_heatmap(weights_2d, LOCI, LOCI, ax=ax)
        assert fig_ret is fig_ext
        plt.close("all")

    def test_with_patient_id(self, weights_2d: np.ndarray) -> None:
        fig = plot_attention_heatmap(weights_2d, LOCI, LOCI, patient_id="p001")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotBothDirections:
    def test_returns_figure(self, single_layer_set: AttentionWeightSet) -> None:
        fig = plot_both_directions(single_layer_set)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        plt.close("all")

    def test_multi_layer(self, multi_layer_set: AttentionWeightSet) -> None:
        fig = plot_both_directions(multi_layer_set, layer=0)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_no_annotate(self, single_layer_set: AttentionWeightSet) -> None:
        fig = plot_both_directions(single_layer_set, annotate_values=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotPopulationHeatmap:
    def test_returns_figure_d2r(self, population_weights: PopulationWeights) -> None:
        fig = plot_population_heatmap(population_weights, direction="d2r")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_returns_figure_r2d(self, population_weights: PopulationWeights) -> None:
        fig = plot_population_heatmap(population_weights, direction="r2d")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_existing_ax(self, population_weights: PopulationWeights) -> None:
        fig_ext, ax = plt.subplots()
        fig_ret = plot_population_heatmap(population_weights, ax=ax)
        assert fig_ret is fig_ext
        plt.close("all")

    def test_no_annotate(self, population_weights: PopulationWeights) -> None:
        fig = plot_population_heatmap(population_weights, annotate_values=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotPopulationBothDirections:
    def test_returns_figure(self, population_weights: PopulationWeights) -> None:
        fig = plot_population_both_directions(population_weights)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        plt.close("all")


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    def test_saves_pdf_and_png(self, tmp_path: pytest.FixtureRequest) -> None:
        fig, _ = plt.subplots()
        paths = save_figure(fig, tmp_path / "test_fig")
        plt.close("all")
        assert len(paths) == 2
        exts = {p.suffix for p in paths}
        assert ".pdf" in exts
        assert ".png" in exts
        for p in paths:
            assert p.exists()

    def test_custom_formats(self, tmp_path: pytest.FixtureRequest) -> None:
        fig, _ = plt.subplots()
        paths = save_figure(fig, tmp_path / "custom", formats=["png"])
        plt.close("all")
        assert len(paths) == 1
        assert paths[0].suffix == ".png"
        assert paths[0].exists()

    def test_creates_parent_dir(self, tmp_path: pytest.FixtureRequest) -> None:
        fig, _ = plt.subplots()
        out = tmp_path / "subdir" / "nested" / "fig"
        save_figure(fig, out, formats=["png"])
        plt.close("all")
        assert out.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# generate_patient_map
# ---------------------------------------------------------------------------


class TestGeneratePatientMap:
    def test_returns_correct_keys(self) -> None:
        model = _MockModel()
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        figures = generate_patient_map(model, donor, recip, clinical, LOCI)
        plt.close("all")
        assert set(figures.keys()) == {"d2r", "r2d", "both"}
        for fig in figures.values():
            assert isinstance(fig, plt.Figure)

    def test_saves_files_when_out_dir_given(self, tmp_path: pytest.FixtureRequest) -> None:
        model = _MockModel()
        donor = torch.randn(1, N, 8)
        recip = torch.randn(1, N, 8)
        clinical = torch.randn(1, 4)
        generate_patient_map(
            model, donor, recip, clinical, LOCI,
            patient_id="p001", out_dir=tmp_path,
        )
        plt.close("all")
        # Expect 3 keys × 2 formats (pdf + png) = 6 files
        files = list(tmp_path.iterdir())
        assert len(files) >= 6


# ---------------------------------------------------------------------------
# generate_population_map
# ---------------------------------------------------------------------------


class TestGeneratePopulationMap:
    def _make_batch(self) -> dict[str, torch.Tensor]:
        return {
            "donor_embeddings": torch.randn(2, N, 8),
            "recipient_embeddings": torch.randn(2, N, 8),
            "clinical_features": torch.randn(2, 4),
        }

    def test_returns_correct_keys(self) -> None:
        model = _MockModel()
        batches = [self._make_batch()]
        figures = generate_population_map(model, batches, LOCI)
        plt.close("all")
        assert set(figures.keys()) == {"population_d2r", "population_r2d", "population_both"}

    def test_raises_on_empty_iterator(self) -> None:
        model = _MockModel()
        with pytest.raises(ValueError, match="No attention weights"):
            generate_population_map(model, [], LOCI)

    def test_saves_files_when_out_dir_given(self, tmp_path: pytest.FixtureRequest) -> None:
        model = _MockModel()
        batches = [self._make_batch()]
        generate_population_map(model, batches, LOCI, out_dir=tmp_path)
        plt.close("all")
        files = list(tmp_path.iterdir())
        assert len(files) >= 6
