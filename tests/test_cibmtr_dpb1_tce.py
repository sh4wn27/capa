"""Tests for scripts/run_cibmtr_dpb1_tce.py (synthetic rows only, no CIBMTR data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import run_cibmtr_dpb1_tce as tce  # noqa: E402


def _synthetic_cohort(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random cohort with the IB19-01a columns the analysis reads."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tce": rng.integers(0, 4, n).astype(float),
        "age": rng.uniform(20, 70, n),
        "dnrage": rng.uniform(18, 50, n),
        "yeartx": rng.integers(2008, 2018, n).astype(float),
        **{c: rng.integers(0, 2, n).astype(float) for c in tce.HLA_MM},
        **{c: rng.integers(0, 3, n).astype(float) for c in tce.CATEGORICAL},
        "agvhd24": rng.integers(0, 2, n).astype(float),
        "intxagvhd24": rng.uniform(0.5, 24, n),
    })
    return tce.add_tce_indicators(df)


class TestTceIndicators:
    """Decoding of the IB19-01a ``tce`` variable."""

    def test_codes_map_to_groups(self) -> None:
        """Each code maps to one group and the right indicators."""
        df = tce.add_tce_indicators(pd.DataFrame({"tce": [0.0, 1.0, 2.0, 3.0]}))
        assert df["tce_group"].tolist() == [
            "gvh_nonpermissive", "hvg_nonpermissive", "match", "permissive",
        ]
        assert df["tce_nonpermissive"].tolist() == [1.0, 1.0, 0.0, 0.0]
        assert df["tce_gvh_nonpermissive"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert df["tce_hvg_nonpermissive"].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert df["tce_permissive"].tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_unknown_code_raises(self) -> None:
        """An undocumented code is rejected, not silently mapped."""
        with pytest.raises(ValueError):
            tce.add_tce_indicators(pd.DataFrame({"tce": [7.0]}))


class TestDesign:
    """Design-matrix construction."""

    def test_directional_nests_symmetric(self) -> None:
        """Directional adds exactly the GvH indicator to symmetric."""
        df = _synthetic_cohort()
        sub = tce.outcome_frame(df, "aGvHD II-IV")
        clin = tce.clinical_matrix(df)
        sym = tce.design(sub, clin, "symmetric")
        dirn = tce.design(sub, clin, "directional")
        assert set(sym.columns) < set(dirn.columns)
        assert set(dirn.columns) - set(sym.columns) == {"tce_gvh_nonpermissive"}

    def test_rare_levels_dropped(self) -> None:
        """Levels below MIN_LEVEL_N are folded away."""
        df = _synthetic_cohort()
        df.loc[df.index[:3], "drsex"] = 99.0  # 3 patients < MIN_LEVEL_N
        clin = tce.clinical_matrix(df)
        assert "drsex_99" not in clin.columns
        assert (clin.filter(like="_").select_dtypes("number").sum() > 0).all()

    def test_outcome_frame_drops_missing(self) -> None:
        """Missing event codes and donor ages are excluded."""
        df = _synthetic_cohort(n=50)
        df.loc[df.index[0], "agvhd24"] = tce.MISSING_CODE
        df.loc[df.index[1], "dnrage"] = np.nan
        sub = tce.outcome_frame(df, "aGvHD II-IV")
        assert len(sub) == 48
        assert set(sub["E"].unique()) <= {0, 1}


class TestHazardRatios:
    """Centre-clustered hazard-ratio output."""

    def test_contrast_reported(self) -> None:
        """All TCE HRs, incl. the GvH/HvG contrast, are returned."""
        df = _synthetic_cohort(n=600, seed=3)
        df["pseudoccn"] = np.arange(len(df)) % 20
        sub = tce.outcome_frame(df, "aGvHD II-IV")
        hr = tce.tce_hazard_ratios(sub, tce.clinical_matrix(df), df["pseudoccn"])
        assert set(hr) == {
            "tce_permissive", "tce_gvh_nonpermissive", "tce_hvg_nonpermissive",
            "gvh_vs_hvg_nonpermissive",
        }
        for row in hr.values():
            assert row["ci_lo"] <= row["hr"] <= row["ci_hi"]
