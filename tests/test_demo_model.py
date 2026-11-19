"""Tests for the demo-model pieces: partial-mismatch cohorts and CV label wiring.

Uses synthetic embeddings for the simulation pool alleles only; no real cache,
no patient data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.run_directional_simulation import generate_directional_cohort  # noqa: E402
from scripts.run_haplo_simulation import _POOLS, LOCI  # noqa: E402
from scripts.run_hierarchy_sensitivity import make_labels  # noqa: E402
from scripts.train_demo_model import MODEL_CONFIG  # noqa: E402

from capa.model.capa_model import CAPAModel  # noqa: E402


@pytest.fixture(scope="module")
def pool_embeddings() -> dict[str, np.ndarray]:
    """Random 16-dim embeddings for every allele the simulation can draw."""
    rng = np.random.default_rng(0)
    return {a: rng.standard_normal(16) for loc in LOCI for a in _POOLS[loc][0]}


class TestPartialMismatch:
    def test_default_mismatches_every_locus(self, pool_embeddings: dict) -> None:
        df, _ = generate_directional_cohort(60, pool_embeddings, np.random.default_rng(1))
        assert (df["n_mismatches"] == len(LOCI)).all()
        for loc in LOCI:
            assert (df[f"donor_{loc}_1"] != df[f"recip_{loc}_1"]).all()

    def test_default_equals_explicit_one(self, pool_embeddings: dict) -> None:
        a, _ = generate_directional_cohort(60, pool_embeddings, np.random.default_rng(1))
        b, _ = generate_directional_cohort(
            60, pool_embeddings, np.random.default_rng(1), p_mismatch=1.0,
        )
        pd.testing.assert_frame_equal(a, b)

    def test_half_mismatch_has_matched_loci(self, pool_embeddings: dict) -> None:
        df, _ = generate_directional_cohort(
            400, pool_embeddings, np.random.default_rng(2), p_mismatch=0.5,
        )
        rate = df[[f"bin_mm_{loc}" for loc in LOCI]].to_numpy().mean()
        assert 0.4 < rate < 0.6
        for loc in LOCI:
            matched = df[f"bin_mm_{loc}"] == 0
            assert (df.loc[matched, f"donor_{loc}_1"] == df.loc[matched, f"recip_{loc}_1"]).all()
            assert (df.loc[matched, f"dist_{loc}"] == 0.0).all()
        assert (df["n_mismatches"] == df[[f"bin_mm_{loc}" for loc in LOCI]].sum(axis=1)).all()


class TestDemoArchitecture:
    def test_primary_model_parameter_count(self) -> None:
        model = CAPAModel(**MODEL_CONFIG)  # type: ignore[arg-type]
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 469_580

    def test_matched_pair_differs_from_mismatched(self) -> None:
        torch.manual_seed(0)
        model = CAPAModel(**{**MODEL_CONFIG, "dropout": 0.0}).eval()  # type: ignore[arg-type]
        clin = model.clinical_encoder(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))
        diff = torch.zeros(2, len(LOCI), 1280)
        diff[1, 3] = torch.randn(1280)
        cif = model.cif(diff, diff, clin)
        assert not torch.allclose(cif[0], cif[1])


class TestHierarchyLabels:
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({
            "survival_time_days": [500.0, 300.0, 800.0],
            "relapse": [1, 0, 0],
            "dead": [0, 1, 0],
            "acute_gvhd_iii_iv": [1, 1, 0],
            "days_to_acute_gvhd_iii_iv": [20.0, 15.0, np.nan],
        })

    def test_outcome_hierarchy_suppresses_gvhd(self) -> None:
        etype, etime = make_labels(self._frame(), "outcome")
        assert etype.tolist() == [2, 3, 0]
        assert etime.tolist() == [500.0, 300.0, 730.0]

    def test_earliest_uses_gvhd_onset(self) -> None:
        etype, etime = make_labels(self._frame(), "earliest")
        assert etype.tolist() == [1, 1, 0]
        assert etime.tolist() == [20.0, 15.0, 730.0]
