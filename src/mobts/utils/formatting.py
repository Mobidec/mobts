"""
Initial formatting of the dataset

This module contains:
- standardizing input names
- removing rows with missing counter names
- removing counters which do not have sufficient present observations
- transforming the timestamp column to datetime type
- adding further temporal columns
"""

import pandas as pd
from ..configs.config_common import ColumnsConfig, SparsityConfig


def _standardize_input(
    df: pd.DataFrame,
    counter_col: str,
    timestamp_col: str,
    count_col: str,
    out_cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:
    """
    Standardize raw input to canonical column names

    ------
    Parameters:

    - df: Dataset
    - counter_col: Column for counter
    - timestamp_col: Column for timestamp
    - count_col: Column for counts
    - out_cols: Column config

    -----
    Returns:

    - Dataframe with standardized canonical names

    -----
    Notes:

    - This functions is fed externally when it is run. The three column names should be given by user.
    """

    # Initial check to see if input columns are well defined, and if not, raise an error
    missing_col = [c for c in (counter_col, timestamp_col, count_col) if c not in df.columns]

    if missing_col:
        raise ValueError(f'Missing required columns: {missing_col}')

    # Rename column names from input names to canonical names
    out = df.rename(columns={counter_col: out_cols.counter, timestamp_col: out_cols.timestamp, count_col: out_cols.count}).copy()

    out = out[[out_cols.counter, out_cols.timestamp, out_cols.count]]

    return out


def _drop_nan_counters(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:
    """
    Remove rows with missing counter names

    ------
    Parameters:

    - df: Dataset
    - cols: Column config

    -----
    Returns:

    - Dataframe with no missing counter name
    - Number of rows with missing counter names
    """
    out = df.copy()
    out = out[~out[cols.counter].isna()].copy()

    n_missing_rows = len(df) - len(out)

    return df, n_missing_rows


def _drop_sparse_counters(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg_spr: SparsityConfig = SparsityConfig(),
) -> pd.DataFrame:
    """
    Remove counters with sparse observations

    -----
    Parameters:

    - df: dataset
    - cols: column config
    - cfg_spr: sparsity config

    -----
    Returns:

    - Dataframe with no sparse counters
    """

    out = df.copy()

    out = out[out.groupby(cols.counter)[cols.count].transform(lambda s: s.isna().mean()) <= cfg_spr.sparse_threshold]

    return out


def _format_datetime(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:
    """
    Transform the timestamp column to datetime type

    ------
    Parameters:

    - df: Dataset
    - cols: Column config

    -----
    Returns:

    - Dataframe with timestamp column as datetime datetype
    """

    # Make sure the timestamp column is not already in datetime format, to avoid costly execution

    if pd.api.types.is_datetime64_any_dtype(df[cols.timestamp]):
        return df

    else:
        df[cols.timestamp] = pd.to_datetime(df[cols.timestamp], utc=True, format='mixed')
        out = df.sort_values([cols.counter, cols.timestamp]).reset_index(drop=True)

        return out


def _add_temporal_columns(
    df: pd.DataFrame,
    freq: str,
    cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:
    """
    Adding further temporal columns

    ------
    Parameters:

    - df: Dataset
    - cols: Column config

    -----
    Returns:

    - Dataframe with additional date, weekday, and week number columns for daily frequency
    - All above, plus hour and hour of week (how) columns for hourly frequency
    """

    df = df.copy()

    df[cols.date] = df[cols.timestamp].dt.floor('D')
    df[cols.weekday] = df[cols.timestamp].dt.dayofweek
    df[cols.week_num] = (df[cols.timestamp] - df[cols.timestamp].min()).dt.days // 7

    # only add the hour-related columns if the frequency is hourly
    if freq == 'hourly':
        df[cols.how] = df[cols.timestamp].dt.dayofweek * 24 + df[cols.timestamp].dt.hour
        df[cols.hour] = df[cols.timestamp].dt.hour

    return df


def _determine_temporal_frequency(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
) -> str:

    dt = df[cols.timestamp].sort_values().diff().dropna()

    # remove zero diffs (duplicates)
    dt = dt[dt > pd.Timedelta(0)]

    delta = dt.mode().iloc[0]

    if pd.Timedelta('30min') <= delta <= pd.Timedelta('2h'):
        return 'hourly'
    elif pd.Timedelta('12h') <= delta <= pd.Timedelta('2D'):
        return 'daily'
    else:
        return 'unknown'


def _validate_frequency(
    freq: str,
) -> str:

    if freq not in {'daily', 'hourly'}:
        raise ValueError(f'Unsupported frequency: {freq}')
    return freq
