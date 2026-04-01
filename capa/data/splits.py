"""Stratified train/val/test splitting for competing risks data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_splits(
    df: pd.DataFrame,
    *,
    event_col: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits preserving event distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    event_col : str
        Column name indicating the event type (used for stratification).
    val_fraction : float
        Fraction of total data to use for validation.
    test_fraction : float
        Fraction of total data to use for testing.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``
    """
    rng = np.random.default_rng(random_seed)
    seed = int(rng.integers(0, 2**31))

    stratify = df[event_col] if event_col in df.columns else None

    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )

    adjusted_val = val_fraction / (1.0 - test_fraction)
    stratify_tv = train_val[event_col] if stratify is not None else None

    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        random_state=seed,
        stratify=stratify_tv,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
