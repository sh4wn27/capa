"""Tests for scripts/run_cibmtr_dpb1_structure.py (synthetic rows only, no CIBMTR data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import run_cibmtr_dpb1_structure as st  # noqa: E402

# (TCE, TCE3, v84_87, coretce) -> expected dpb1_group
_CASES = [
    ((1.0, -9.0, -9.0, -9.0), "match"),
    ((3.0, -9.0, -9.0, -9.0), "nonpermissive"),
    ((2.0, 0.0, -9.0, -9.0), "permissive_other"),
    ((2.0, 1.0, 0.0, 0.0), "tce3_motif_mismatch"),
    ((2.0, 1.0, 1.0, 0.0), "tce3_motif_match_noncore"),
    ((2.0, 1.0, 1.0, 1.0), "tce3_core"),
]


def _releases() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One synthetic patient per DPB1 subgroup, split into the two releases."""
    codes = np.array([c for c, _ in _CASES])
    b = pd.DataFrame(codes, columns=["TCE", "TCE3", "v84_87", "coretce"])
    b["pseudoid"] = np.arange(len(b), dtype=float)
    b["gvhdgp"] = [4.0, 5.0, 10.0, 11.0, 20.0, 6.0]
    a = pd.DataFrame({"pseudoid": b["pseudoid"][::-1].to_numpy(), "tce": 3.0})
    for col in (*st.HLA_MM, "DPB1_MM"):
        a[col] = 1.0
    return a, b


class TestMergeReleases:
    """Linking IB19-01a/b and deriving DPB1 subgroups."""

    def test_subgroups(self) -> None:
        """Each TCE/TCE3/motif/core code combination maps to its subgroup."""
        m = st.merge_releases(*_releases())
        assert m["dpb1_group"].tolist() == [g for _, g in _CASES]
        assert m["in_tce3"].tolist() == [False, False, False, True, True, True]
        assert (m[list(st.SUBGROUPS)].sum(axis=1) == (m["dpb1_group"] != "match")).all()

    def test_core_implies_motif_match(self) -> None:
        """Core TCE3 rows are always motif-matched."""
        m = st.merge_releases(*_releases())
        assert (m.loc[m["core_tce3"] == 1, "motif_match"] == 1).all()

    def test_rare_prophylaxis_folded(self) -> None:
        """Regimens 10 and 11 are folded into 20 ("other")."""
        m = st.merge_releases(*_releases())
        assert not m["gvhdgp"].isin([10.0, 11.0]).any()

    def test_unlinked_patient_raises(self) -> None:
        """A patient present in only one release is an error, not a silent drop."""
        a, b = _releases()
        with pytest.raises(ValueError):
            st.merge_releases(a.iloc[1:], b)


class TestSubcohortDesign:
    """Nested design matrices within the TCE3 subcohort."""

    def test_nesting(self) -> None:
        """Clinical < motif < core, each adding one structural column."""
        rng = np.random.default_rng(0)
        n = 200
        sub = pd.DataFrame({
            "DPB1_MM": rng.integers(1, 3, n).astype(float),
            "motif_match": rng.integers(0, 2, n).astype(float),
            "T": rng.uniform(1, 24, n),
            "E": rng.integers(0, 2, n),
        })
        sub["core_tce3"] = sub["motif_match"] * rng.integers(0, 2, n)
        clin = pd.DataFrame({"age": rng.uniform(20, 70, n)})
        cols = {fs: set(st.subcohort_design(sub, clin, fs).columns) for fs in st.FEATURE_SETS}
        assert cols["motif"] - cols["clinical"] == {"motif_match"}
        assert cols["core"] - cols["motif"] == {"core_tce3"}
        assert "dpb1_two_mm" in cols["clinical"]
