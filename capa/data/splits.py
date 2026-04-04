"""Stratified train/val/test splitting for competing-risks survival data.

Stratification label
--------------------
:func:`sklearn.model_selection.StratifiedShuffleSplit` requires a single
categorical label per sample.  For competing-risks data with three possible
events — relapse, transplant-related mortality (TRM), and severe GvHD — plus
long-term survivors, we derive a 4-class label using a **priority hierarchy**
so that every patient belongs to exactly one class:

1. ``"relapse"``  — ``relapse == 1`` (highest priority; defines the primary
   competing event in the survival model regardless of vital status).
2. ``"trm"``      — ``dead == 1`` AND ``relapse == 0`` (died without
   relapsing; almost always transplant-related).
3. ``"gvhd"``     — ``acute_gvhd_iii_iv == 1``, ``relapse == 0``,
   ``dead == 0`` (severe GvHD but survived; important subgroup to preserve).
4. ``"alive"``    — all remaining patients (censored, alive, no severe events
   by end of follow-up).

If any class contains fewer than two samples (which can happen in very small
synthetic datasets), it is merged into ``"alive"`` before splitting to avoid
a :class:`~sklearn.exceptions.ValueError` from
:class:`~sklearn.model_selection.StratifiedShuffleSplit`.

Persistence
-----------
Split indices are stored as **positional indices** (0-based, relative to the
row order of the input DataFrame).  Callers can reconstruct any partition with
``df.iloc[indices["train"]]``.  The JSON file also records a ``"_meta"`` key
with split parameters for provenance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Competing-risk class labels (used as string values in the stratification series)
_ALIVE = "alive"
_RELAPSE = "relapse"
_TRM = "trm"
_GVHD = "gvhd"

# Minimum number of samples a class must have to be kept as a distinct stratum.
# Classes below this threshold are merged into _ALIVE.
_MIN_CLASS_SIZE = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_competing_risk_label(df: pd.DataFrame) -> pd.Series:
    """Derive a 4-class competing-risk label from outcome columns.

    Every patient is assigned to exactly one class via the priority hierarchy
    described in the module docstring.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BMT DataFrame with at least the columns ``relapse``,
        ``dead``, and ``acute_gvhd_iii_iv``.

    Returns
    -------
    pd.Series
        String-valued Series of the same length as *df* with values in
        ``{"alive", "relapse", "trm", "gvhd"}``.

    Raises
    ------
    KeyError
        If any of the required columns is absent from *df*.
    """
    required = {"relapse", "dead", "acute_gvhd_iii_iv"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Required outcome columns missing from DataFrame: {sorted(missing)}")

    label = pd.Series(_ALIVE, index=df.index, dtype="object")

    # Priority 3: Alive patients with severe GvHD who did NOT relapse
    mask_gvhd = (
        (df["acute_gvhd_iii_iv"] == 1)
        & (df["relapse"] == 0)
        & (df["dead"] == 0)
    )
    label[mask_gvhd] = _GVHD

    # Priority 2: TRM — died without relapsing (overrides gvhd label if both apply)
    mask_trm = (df["dead"] == 1) & (df["relapse"] == 0)
    label[mask_trm] = _TRM

    # Priority 1: Relapse (highest priority, overrides all)
    label[df["relapse"] == 1] = _RELAPSE

    return label


def _merge_small_classes(label: pd.Series, min_size: int = _MIN_CLASS_SIZE) -> pd.Series:
    """Merge any class with fewer than *min_size* samples into ``"alive"``.

    Called internally by :func:`make_splits` so that
    :class:`~sklearn.model_selection.StratifiedShuffleSplit` never raises a
    ``ValueError`` due to a stratum that is too small to split.

    Parameters
    ----------
    label : pd.Series
        Output of :func:`make_competing_risk_label`.
    min_size : int
        Minimum samples required to keep a class as a distinct stratum.

    Returns
    -------
    pd.Series
        Possibly-modified copy of *label*.
    """
    counts = label.value_counts()
    small_classes = [cls for cls in counts.index if counts[cls] < min_size]
    if small_classes:
        logger.warning(
            "Merging minority class(es) %s (n < %d) into '%s' for stratification.",
            small_classes,
            min_size,
            _ALIVE,
        )
        label = label.replace({cls: _ALIVE for cls in small_classes})
    return label


def make_splits(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train / val / test partitions of *df*.

    Uses :class:`~sklearn.model_selection.StratifiedShuffleSplit` twice —
    once to carve out the test set and once to split the remaining data into
    train and validation.  The competing-risk label produced by
    :func:`make_competing_risk_label` is used as the stratification target.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset from :func:`capa.data.loader.load_bmt`.  Row
        order must be stable; positional indices saved to JSON refer to this
        ordering.
    val_fraction : float
        Fraction of the **total** dataset to use for validation (default 0.20).
    test_fraction : float
        Fraction of the **total** dataset to use for testing (default 0.20).
    random_seed : int
        Master random seed.  The validation split uses ``random_seed + 1`` to
        ensure the two splits are independent.
    output_path : Path | None
        If given, write split indices (and metadata) to this JSON file.
        Parent directories are created automatically.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` with reset 0-based integer indices.

    Raises
    ------
    ValueError
        If ``val_fraction + test_fraction >= 1.0``.
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0"
        )

    label = _merge_small_classes(make_competing_risk_label(df))
    logger.info(
        "Competing-risk label distribution (n=%d):\n%s",
        len(df),
        label.value_counts().to_string(),
    )

    # --- Step 1: carve out held-out test set ---
    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_seed,
    )
    (trainval_pos, test_pos), = sss_test.split(np.arange(len(df)), label)

    # --- Step 2: split train+val into train and val ---
    adjusted_val = val_fraction / (1.0 - test_fraction)
    trainval_label = label.iloc[trainval_pos]

    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=adjusted_val,
        random_state=random_seed + 1,
    )
    (train_sub_pos, val_sub_pos), = sss_val.split(
        np.arange(len(trainval_pos)), trainval_label
    )

    # Map sub-positions back to original df positional indices
    train_pos: np.ndarray = trainval_pos[train_sub_pos]
    val_pos: np.ndarray = trainval_pos[val_sub_pos]

    logger.info(
        "Split sizes — train: %d (%.0f%%), val: %d (%.0f%%), test: %d (%.0f%%)",
        len(train_pos), 100 * len(train_pos) / len(df),
        len(val_pos), 100 * len(val_pos) / len(df),
        len(test_pos), 100 * len(test_pos) / len(df),
    )
    _log_label_distribution("train", label.iloc[train_pos])
    _log_label_distribution("val", label.iloc[val_pos])
    _log_label_distribution("test", label.iloc[test_pos])

    if output_path is not None:
        save_split_indices(
            {
                "train": train_pos.tolist(),
                "val": val_pos.tolist(),
                "test": test_pos.tolist(),
            },
            output_path,
            metadata={
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "random_seed": random_seed,
                "n_total": len(df),
                "n_train": int(len(train_pos)),
                "n_val": int(len(val_pos)),
                "n_test": int(len(test_pos)),
            },
        )

    return (
        df.iloc[train_pos].reset_index(drop=True),
        df.iloc[val_pos].reset_index(drop=True),
        df.iloc[test_pos].reset_index(drop=True),
    )


def save_split_indices(
    indices: dict[str, list[int]],
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Write split indices to a JSON file for reproducibility.

    Parameters
    ----------
    indices : dict[str, list[int]]
        Mapping of split name (``"train"``, ``"val"``, ``"test"``) to a list
        of **positional** indices into the original DataFrame.
    path : Path
        Destination path.  Parent directories are created if absent.
    metadata : dict[str, object] | None
        Optional provenance information stored under the ``"_meta"`` key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if metadata is not None:
        payload["_meta"] = metadata
    payload.update(indices)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved split indices → %s", path)


def load_split_indices(path: Path) -> dict[str, list[int]]:
    """Load split indices previously written by :func:`save_split_indices`.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict[str, list[int]]
        Mapping of split name → positional index list.  The ``"_meta"`` key
        is excluded from the return value.
    """
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, object] = json.load(fh)
    return {k: v for k, v in raw.items() if k != "_meta"}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_label_distribution(split_name: str, labels: pd.Series) -> None:
    """Log the normalised label distribution for a split partition."""
    dist = labels.value_counts(normalize=True).sort_index()
    logger.info(
        "%s label proportions:\n%s",
        split_name,
        dist.to_string(),
    )
