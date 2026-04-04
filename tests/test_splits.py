"""Tests for capa.data.splits — stratified competing-risks splitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capa.data.splits import (
    _ALIVE,
    _GVHD,
    _RELAPSE,
    _TRM,
    _merge_small_classes,
    load_split_indices,
    make_competing_risk_label,
    make_splits,
    save_split_indices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    *,
    n: int = 200,
    relapse_rate: float = 0.25,
    trm_rate: float = 0.20,
    gvhd_rate: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with the four competing-risk outcome columns.

    The distributions are chosen so that every class has enough members for
    StratifiedShuffleSplit to work:

    * relapse   ≈ relapse_rate * n
    * trm       ≈ trm_rate * n   (dead=1, relapse=0)
    * gvhd      ≈ gvhd_rate * n  (alive, severe GvHD, no relapse)
    * alive     ≈ remaining

    Other numeric columns are added so that the DataFrame looks like a real
    BMT frame.  They are not used by the splitting logic.
    """
    rng = np.random.default_rng(seed)
    n_relapse = int(n * relapse_rate)
    n_trm = int(n * trm_rate)
    n_gvhd = int(n * gvhd_rate)
    n_alive = n - n_relapse - n_trm - n_gvhd

    relapse = np.array([1] * n_relapse + [0] * (n - n_relapse), dtype=np.int8)
    dead = np.array(
        [0] * n_relapse  # relapse patients may be alive
        + [1] * n_trm    # TRM: dead
        + [0] * n_gvhd   # GvHD survivors: alive
        + [0] * n_alive,
        dtype=np.int8,
    )
    acute_gvhd = np.array(
        [0] * n_relapse
        + [0] * n_trm
        + [1] * n_gvhd
        + [0] * n_alive,
        dtype=np.int8,
    )
    # Shuffle rows together
    perm = rng.permutation(n)
    return pd.DataFrame(
        {
            "relapse": relapse[perm],
            "dead": dead[perm],
            "acute_gvhd_iii_iv": acute_gvhd[perm],
            "survival_time_days": rng.uniform(10, 2000, size=n),
        }
    )


# A small realistic-sized frame matching UCI BMT proportions
@pytest.fixture()
def bmt_like() -> pd.DataFrame:
    """187-row synthetic frame mimicking UCI BMT outcome proportions."""
    return _make_df(n=187, relapse_rate=0.27, trm_rate=0.22, gvhd_rate=0.08, seed=7)


# ---------------------------------------------------------------------------
# make_competing_risk_label
# ---------------------------------------------------------------------------

class TestMakeCompetingRiskLabel:
    def test_relapse_gets_relapse_label(self) -> None:
        df = pd.DataFrame({"relapse": [1], "dead": [0], "acute_gvhd_iii_iv": [0]})
        assert make_competing_risk_label(df).iloc[0] == _RELAPSE

    def test_dead_no_relapse_gets_trm(self) -> None:
        df = pd.DataFrame({"relapse": [0], "dead": [1], "acute_gvhd_iii_iv": [0]})
        assert make_competing_risk_label(df).iloc[0] == _TRM

    def test_alive_gvhd_no_relapse_gets_gvhd(self) -> None:
        df = pd.DataFrame({"relapse": [0], "dead": [0], "acute_gvhd_iii_iv": [1]})
        assert make_competing_risk_label(df).iloc[0] == _GVHD

    def test_alive_no_events_gets_alive(self) -> None:
        df = pd.DataFrame({"relapse": [0], "dead": [0], "acute_gvhd_iii_iv": [0]})
        assert make_competing_risk_label(df).iloc[0] == _ALIVE

    def test_relapse_overrides_gvhd(self) -> None:
        """Patient with both relapse and severe GvHD → relapse (highest priority)."""
        df = pd.DataFrame({"relapse": [1], "dead": [0], "acute_gvhd_iii_iv": [1]})
        assert make_competing_risk_label(df).iloc[0] == _RELAPSE

    def test_relapse_overrides_trm(self) -> None:
        """Dead patient who also relapsed → relapse (TRM would be incorrect)."""
        df = pd.DataFrame({"relapse": [1], "dead": [1], "acute_gvhd_iii_iv": [0]})
        assert make_competing_risk_label(df).iloc[0] == _RELAPSE

    def test_dead_gvhd_no_relapse_gets_trm(self) -> None:
        """Patient who died WITH severe GvHD but no relapse → TRM (priority 2 > 3)."""
        df = pd.DataFrame({"relapse": [0], "dead": [1], "acute_gvhd_iii_iv": [1]})
        assert make_competing_risk_label(df).iloc[0] == _TRM

    def test_all_four_classes_present(self, bmt_like: pd.DataFrame) -> None:
        label = make_competing_risk_label(bmt_like)
        assert set(label.unique()) >= {_ALIVE, _RELAPSE, _TRM, _GVHD}

    def test_no_nan_in_label(self, bmt_like: pd.DataFrame) -> None:
        label = make_competing_risk_label(bmt_like)
        assert label.notna().all()

    def test_same_length_as_input(self, bmt_like: pd.DataFrame) -> None:
        label = make_competing_risk_label(bmt_like)
        assert len(label) == len(bmt_like)

    def test_index_preserved(self) -> None:
        df = pd.DataFrame(
            {"relapse": [1, 0, 0], "dead": [0, 1, 0], "acute_gvhd_iii_iv": [0, 0, 1]},
            index=[10, 20, 30],
        )
        label = make_competing_risk_label(df)
        assert list(label.index) == [10, 20, 30]

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"relapse": [1], "dead": [0]})  # no acute_gvhd_iii_iv
        with pytest.raises(KeyError, match="acute_gvhd_iii_iv"):
            make_competing_risk_label(df)

    def test_small_class_merged_into_alive(self) -> None:
        """_merge_small_classes must collapse a singleton class into 'alive'."""
        # Construct a label where gvhd has exactly 1 sample
        rows = []
        for _ in range(5):
            rows.append({"relapse": 0, "dead": 0, "acute_gvhd_iii_iv": 0})  # alive
        rows.append({"relapse": 0, "dead": 0, "acute_gvhd_iii_iv": 1})  # single gvhd
        df = pd.DataFrame(rows)
        raw_label = make_competing_risk_label(df)
        assert _GVHD in raw_label.values  # present before merging
        merged = _merge_small_classes(raw_label)
        assert _GVHD not in merged.values  # collapsed into alive


# ---------------------------------------------------------------------------
# make_splits — sizes and disjointness
# ---------------------------------------------------------------------------

class TestMakeSplitsSizes:
    def test_default_fractions_60_20_20(self, bmt_like: pd.DataFrame) -> None:
        train, val, test = make_splits(bmt_like, random_seed=0)
        n = len(bmt_like)
        assert len(train) + len(val) + len(test) == n

    def test_approximate_60_20_20(self, bmt_like: pd.DataFrame) -> None:
        train, val, test = make_splits(bmt_like, random_seed=0)
        n = len(bmt_like)
        # Allow ±2 rows of rounding tolerance
        assert abs(len(train) - int(0.60 * n)) <= 2
        assert abs(len(val)   - int(0.20 * n)) <= 2
        assert abs(len(test)  - int(0.20 * n)) <= 2

    def test_no_overlap_between_splits(self, bmt_like: pd.DataFrame) -> None:
        """Original row indices must be disjoint across all three splits."""
        # We recover original indices by tracking via a synthetic index column
        df = bmt_like.copy()
        df["_row_id"] = np.arange(len(df))
        train, val, test = make_splits(df, random_seed=0)

        train_ids = set(train["_row_id"])
        val_ids = set(val["_row_id"])
        test_ids = set(test["_row_id"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_rows_accounted_for(self, bmt_like: pd.DataFrame) -> None:
        df = bmt_like.copy()
        df["_row_id"] = np.arange(len(df))
        train, val, test = make_splits(df, random_seed=0)

        all_ids = set(train["_row_id"]) | set(val["_row_id"]) | set(test["_row_id"])
        assert all_ids == set(range(len(df)))

    def test_reset_index_starts_at_zero(self, bmt_like: pd.DataFrame) -> None:
        train, val, test = make_splits(bmt_like, random_seed=0)
        assert list(train.index) == list(range(len(train)))
        assert list(val.index) == list(range(len(val)))
        assert list(test.index) == list(range(len(test)))

    def test_custom_fractions(self, bmt_like: pd.DataFrame) -> None:
        train, val, test = make_splits(
            bmt_like, val_fraction=0.15, test_fraction=0.15, random_seed=0
        )
        n = len(bmt_like)
        assert len(train) + len(val) + len(test) == n
        # Train should be ~70%
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_invalid_fractions_raise(self, bmt_like: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            make_splits(bmt_like, val_fraction=0.5, test_fraction=0.6)


# ---------------------------------------------------------------------------
# make_splits — stratification quality
# ---------------------------------------------------------------------------

class TestStratificationQuality:
    """Verify that outcome class proportions are approximately preserved."""

    def _proportions(self, df: pd.DataFrame) -> dict[str, float]:
        label = make_competing_risk_label(df)
        counts = label.value_counts(normalize=True)
        return counts.to_dict()

    def test_relapse_proportion_preserved_in_test(self, bmt_like: pd.DataFrame) -> None:
        global_prop = self._proportions(bmt_like).get(_RELAPSE, 0.0)
        _, _, test = make_splits(bmt_like, random_seed=0)
        test_prop = self._proportions(test).get(_RELAPSE, 0.0)
        # Allow ±15 percentage points — n is small so exact match is impossible
        assert abs(test_prop - global_prop) < 0.15, (
            f"Relapse proportion diverged: global={global_prop:.3f}, test={test_prop:.3f}"
        )

    def test_trm_proportion_preserved_in_test(self, bmt_like: pd.DataFrame) -> None:
        global_prop = self._proportions(bmt_like).get(_TRM, 0.0)
        _, _, test = make_splits(bmt_like, random_seed=0)
        test_prop = self._proportions(test).get(_TRM, 0.0)
        assert abs(test_prop - global_prop) < 0.15, (
            f"TRM proportion diverged: global={global_prop:.3f}, test={test_prop:.3f}"
        )

    def test_alive_proportion_preserved_in_train(self, bmt_like: pd.DataFrame) -> None:
        global_prop = self._proportions(bmt_like).get(_ALIVE, 0.0)
        train, _, _ = make_splits(bmt_like, random_seed=0)
        train_prop = self._proportions(train).get(_ALIVE, 0.0)
        assert abs(train_prop - global_prop) < 0.15

    def test_all_classes_present_in_train(self, bmt_like: pd.DataFrame) -> None:
        """Train set must contain all label classes present in the full dataset."""
        full_classes = set(make_competing_risk_label(bmt_like).unique())
        train, _, _ = make_splits(bmt_like, random_seed=0)
        train_classes = set(make_competing_risk_label(train).unique())
        # Train is ~60% of data, so all classes should survive unless tiny
        assert train_classes == full_classes

    def test_test_set_has_all_classes(self, bmt_like: pd.DataFrame) -> None:
        """Test set should represent all competing-risk outcomes."""
        full_classes = set(make_competing_risk_label(bmt_like).unique())
        _, _, test = make_splits(bmt_like, random_seed=0)
        test_classes = set(make_competing_risk_label(test).unique())
        assert test_classes == full_classes

    def test_stratification_better_than_random(self) -> None:
        """Stratified splits should have lower class-proportion variance than random."""
        df = _make_df(n=300, relapse_rate=0.25, trm_rate=0.20, gvhd_rate=0.10, seed=0)
        global_props = make_competing_risk_label(df).value_counts(normalize=True)

        stratified_errors = []
        for seed in range(10):
            _, _, test = make_splits(df, random_seed=seed)
            test_props = make_competing_risk_label(test).value_counts(normalize=True)
            for cls in global_props.index:
                stratified_errors.append(abs(test_props.get(cls, 0) - global_props[cls]))

        # All errors should be small (≤ 12 pp) with stratification
        assert max(stratified_errors) <= 0.12, (
            f"Largest stratification error: {max(stratified_errors):.3f}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_splits(self, bmt_like: pd.DataFrame) -> None:
        df = bmt_like.copy()
        df["_id"] = np.arange(len(df))

        train1, val1, test1 = make_splits(df, random_seed=99)
        train2, val2, test2 = make_splits(df, random_seed=99)

        assert list(train1["_id"]) == list(train2["_id"])
        assert list(val1["_id"]) == list(val2["_id"])
        assert list(test1["_id"]) == list(test2["_id"])

    def test_different_seed_different_splits(self, bmt_like: pd.DataFrame) -> None:
        df = bmt_like.copy()
        df["_id"] = np.arange(len(df))

        _, _, test1 = make_splits(df, random_seed=1)
        _, _, test2 = make_splits(df, random_seed=2)

        assert list(test1["_id"]) != list(test2["_id"])


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

class TestSaveLoadSplitIndices:
    def test_round_trip(self, tmp_path: Path) -> None:
        indices = {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]}
        p = tmp_path / "splits.json"
        save_split_indices(indices, p)
        loaded = load_split_indices(p)
        assert loaded == indices

    def test_meta_excluded_from_load(self, tmp_path: Path) -> None:
        indices = {"train": [0, 1], "val": [2], "test": [3]}
        p = tmp_path / "splits.json"
        save_split_indices(indices, p, metadata={"n_total": 4, "random_seed": 0})
        loaded = load_split_indices(p)
        assert "_meta" not in loaded

    def test_meta_written_to_file(self, tmp_path: Path) -> None:
        indices = {"train": [0], "val": [1], "test": [2]}
        p = tmp_path / "splits.json"
        save_split_indices(indices, p, metadata={"random_seed": 42})
        raw = json.loads(p.read_text())
        assert "_meta" in raw
        assert raw["_meta"]["random_seed"] == 42  # type: ignore[index]

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "splits.json"
        save_split_indices({"train": [0], "val": [1], "test": [2]}, p)
        assert p.exists()

    def test_make_splits_saves_json(
        self, bmt_like: pd.DataFrame, tmp_path: Path
    ) -> None:
        p = tmp_path / "splits.json"
        make_splits(bmt_like, random_seed=0, output_path=p)
        assert p.exists()

        loaded = load_split_indices(p)
        assert set(loaded.keys()) == {"train", "val", "test"}
        # All indices are non-negative integers within range
        n = len(bmt_like)
        for split_name, idxs in loaded.items():
            assert all(0 <= i < n for i in idxs), (
                f"{split_name} contains out-of-range index"
            )

    def test_saved_indices_reconstruct_splits(
        self, bmt_like: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Positional indices saved to JSON must reproduce identical DataFrames."""
        df = bmt_like.copy()
        df["_id"] = np.arange(len(df))

        p = tmp_path / "splits.json"
        train, val, test = make_splits(df, random_seed=5, output_path=p)

        loaded = load_split_indices(p)
        train_r = df.iloc[loaded["train"]].reset_index(drop=True)
        val_r = df.iloc[loaded["val"]].reset_index(drop=True)
        test_r = df.iloc[loaded["test"]].reset_index(drop=True)

        assert list(train_r["_id"]) == list(train["_id"])
        assert list(val_r["_id"]) == list(val["_id"])
        assert list(test_r["_id"]) == list(test["_id"])

    def test_saved_meta_contains_fractions(
        self, bmt_like: pd.DataFrame, tmp_path: Path
    ) -> None:
        p = tmp_path / "splits.json"
        make_splits(
            bmt_like,
            val_fraction=0.20,
            test_fraction=0.20,
            random_seed=0,
            output_path=p,
        )
        raw = json.loads(p.read_text())
        meta = raw["_meta"]
        assert meta["val_fraction"] == 0.20  # type: ignore[index]
        assert meta["test_fraction"] == 0.20  # type: ignore[index]
        assert meta["random_seed"] == 0  # type: ignore[index]
        assert meta["n_total"] == len(bmt_like)  # type: ignore[index]

    def test_no_file_written_when_output_path_none(
        self, bmt_like: pd.DataFrame, tmp_path: Path
    ) -> None:
        make_splits(bmt_like, random_seed=0, output_path=None)
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Real data (skipped when file absent)
# ---------------------------------------------------------------------------

_ARFF_PATH = Path(__file__).parent.parent / "data" / "raw" / "bone-marrow.arff"
_REQUIRES_REAL_DATA = pytest.mark.skipif(
    not _ARFF_PATH.exists(), reason="bone-marrow.arff not present"
)


@_REQUIRES_REAL_DATA
class TestRealData:
    @pytest.fixture(scope="class")
    def bmt(self) -> pd.DataFrame:
        from capa.data.loader import load_bmt
        return load_bmt(_ARFF_PATH)

    def test_split_sizes_187_patients(self, bmt: pd.DataFrame) -> None:
        train, val, test = make_splits(bmt, random_seed=42)
        assert len(train) + len(val) + len(test) == 187
        assert abs(len(train) - 112) <= 2  # 60% of 187
        assert abs(len(val)   - 37) <= 2   # 20% of 187
        assert abs(len(test)  - 38) <= 2   # 20% of 187

    def test_four_classes_present(self, bmt: pd.DataFrame) -> None:
        label = make_competing_risk_label(bmt)
        classes = set(label.unique())
        # All four should appear in a 187-patient cohort
        assert classes >= {_ALIVE, _RELAPSE, _TRM}

    def test_relapse_proportion_within_5pct(self, bmt: pd.DataFrame) -> None:
        global_rate = (bmt["relapse"] == 1).mean()
        _, _, test = make_splits(bmt, random_seed=42)
        test_rate = (test["relapse"] == 1).mean()
        assert abs(test_rate - global_rate) < 0.05

    def test_dead_proportion_within_5pct(self, bmt: pd.DataFrame) -> None:
        global_rate = (bmt["dead"] == 1).mean()
        _, _, test = make_splits(bmt, random_seed=42)
        test_rate = (test["dead"] == 1).mean()
        assert abs(test_rate - global_rate) < 0.05
