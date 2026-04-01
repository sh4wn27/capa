"""UCI BMT dataset loading and cleaning."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Document every column mapping from the UCI BMT dataset's messy column names.
# Reference: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children


def load_bmt(path: Path) -> pd.DataFrame:
    """Load and clean the UCI Bone Marrow Transplant dataset.

    Parameters
    ----------
    path : Path
        Path to the ``bone-marrow.csv`` file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names.
    """
    logger.info("Loading UCI BMT dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    df = _rename_columns(df)
    df = _cast_types(df)
    df = _drop_unusable(df)
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise UCI BMT column names to snake_case.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    # TODO: Complete mapping after inspecting the actual CSV headers.
    rename_map: dict[str, str] = {}
    return df.rename(columns=rename_map)


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after column renaming.

    Returns
    -------
    pd.DataFrame
        DataFrame with correct dtypes.
    """
    # TODO: Implement type casting once column names are confirmed.
    return df


def _drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows or columns that are entirely missing or unusable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after type casting.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)
    df = df.dropna(how="all")
    logger.info("Dropped %d all-NaN rows", before - len(df))
    return df
