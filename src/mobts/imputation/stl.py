"""
STL imputation, prerequisite for donor imputation

This module contains:
- setting the 'period' argument based on temporal frequency, to be used in STL functions
- determining the termporal column based on temporal frequency, on which STL will operate
- a linear interpolation function for initiating the STL function
- rolling median function to be used for calculating rolling median of STL residuals
- function for the application of the initial interpolation for STL
- application of the STL function on one counter (method with adjustment for long holes)
- application of STL on the entire network

"""



import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from ..configs.config_common import ColumnsConfig
from ..configs.config_imputation import STLConfig, OutputConfig
from ..utils.formatting import _determine_temporal_frequency, _validate_frequency


def _get_stl_period(
    freq: str, 
    stl_cfg: STLConfig = STLConfig(),
) -> int:

    """
    Determines the 'period' argument for the STL function

    ------
    Parameters:
    
    - freq: temporal frequency of the project
    - stl_cfg: config for STL

    -----
    Returns:

    - Integar for STL period. 7 for daily data, and 168 for hourly data
    """
    
    freq = _validate_frequency(freq)
    if freq == "hourly":
        return stl_cfg.stl_season_hourly
    elif freq == "daily":
        return stl_cfg.stl_season_daily
    else:
        raise ValueError(f"Unsupported frequency: {freq}")



def _get_grouping_column_for_stl(
    freq: str,
) -> str:

    """
    Determines the temporal column for the STL function to operate on

    ------
    Parameters:
    
    - freq: temporal frequency of the project
    
    -----
    Returns:

    - string indicating the temporal column. "weekday" for daily data, "how" (hour of week) for hourly data
    """
    
    freq = _validate_frequency(freq)
    if freq == "hourly":
        return "how"
    elif freq == "daily":
        return "weekday"
    else:
        raise ValueError(f"Unsupported frequency: {freq}")       


        
def _interpolate_linear(
    s: pd.Series,
) -> pd.Series:

    """
    basic interpolation

    ------
    Parameters:
    
    - s: time-serie corresponding to one single counter

    -----
    Returns:

    - the interpolated time-serie
    """
    
    return s.interpolate(method="linear", limit_area="inside")


    
def _rolling_median_week_window(
    series: pd.Series,
    freq: str,
    stl_cfg: STLConfig = STLConfig(),
) -> pd.Series:

    """
    Calculates a rolling median of time-series

    ------
    Parameters:
    
    - series: time series corresponding to one single counter
    - freq: temporal frequency of the project
    - stl_cfg: config for STL

    -----
    Returns:

    - time series of rolling medians for the time-serie
    """

    # window for calculating the rolling median
    window = stl_cfg.rolling_median_window

    # minimum non-NaN observations in the window for calculating the rolling median
    min_valid = stl_cfg.rolling_median_min_valid
    period_stl = _get_stl_period(freq=freq)

    shifts = []
    for k in range(1, window + 1):
        shifts.append(series.shift(period_stl * k))
        shifts.append(series.shift(-period_stl * k))

    # in case shifts is not available, it would return an empty float series
    if not shifts:
        return pd.Series(index=series.index, dtype=float)

    # stacks the shifted values and checks that there's no infinite values
    mat = np.vstack([s.to_numpy(dtype=float) for s in shifts]).T
    valid_counts = np.isfinite(mat).sum(axis=1)

    # calculating medians for valid observations
    out = np.full(mat.shape[0], np.nan, dtype=float)
    ok = valid_counts >= min_valid
    if ok.any():
        out[ok] = np.nanmedian(mat[ok], axis=1)

    return pd.Series(out, index=series.index)



def _initial_interpolate_for_stl(
    df: pd.DataFrame,
    cols: ColumnsConfig,
    out_cfg: OutputConfig,
) -> pd.DataFrame:

    """
    Applies the preliminary interpolation necessary for STL

    ------
    Parameters:
    
    - series: full dataset
    - cols: columns config
    - out_cfg: config for output columns' names

    -----
    Returns:

    - dataframe with interpolated time-series

    -----
    Notes:
    - this is necessary as STL cannot run on time series with NaN values, so the existing holes are filled using
      interpolation. This allows us to preserve the trend for the missing periods.
    """
    out = df.copy()
    
    out[out_cfg.col_intp] = out.groupby(cols.counter, group_keys=False)[cols.count].apply(_interpolate_linear)
    
    return out



def _stl_on_counter_hole_adjusted(
    g: pd.DataFrame,
    freq: str,
    cols: ColumnsConfig,
    stl_cfg: STLConfig,
    out_cfg: OutputConfig,
) -> pd.DataFrame:

    """
    Applies STL on one counter

    ------
    Parameters:
    
    - g: dataframe for a single counter
    - freq: temporal frequency of the project
    - cols: columns config
    - stl_cfg: config for STL    
    - out_cfg: config for output columns' names

    -----
    Returns:

    - dataframe with imputed missing values for one counter, using STL

    -----
    Notes:
    - the hole adjustment uses an average of general seasonality to fill in for the missing seasonality of the missing period
    """

    # getting the time series of observation for the counter
    y = g[out_cfg.col_intp].astype(float)

    # a double check for if interplotion hasn't filled all gaps for STL
    if y.isna().any():
        y = y.interpolate(limit_direction="both").ffill().bfill()

    # retrieves temporal elements (period and grouping column) based on temporal frequency
    period_stl = _get_stl_period(freq, stl_cfg)
    grouping_col = _get_grouping_column_for_stl(freq)

    # applies STL, and stores them on the counter dataframe
    res = STL(y, period=period_stl, robust=stl_cfg.stl_robust).fit()
    
    g["stl_trend"] = res.trend
    g["stl_season"] = res.seasonal
    g["stl_resid"] = res.resid

    # rolling median of residuals to create a base for STL imputation
    res_med = _rolling_median_week_window(g["stl_resid"], freq=freq, stl_cfg=stl_cfg)

    # STL base for imputation consisting of the trend + seasonality + [rolling] median residuals
    stl_base = (g["stl_trend"] + g["stl_season"] + res_med).clip(lower=stl_cfg.clip_lower)

    # calculates median seasonality of of the entire time series, and where time series has missing data, it replaces the projected seasonality
    # with said median seasonality
    seasonality_by_temporal_granularity = g.loc[g[out_cfg.col_intp].notna()].groupby(grouping_col)["stl_season"].median()
    seasonality_fill = seasonality_by_temporal_granularity.fillna(seasonality_by_temporal_granularity.median(skipna=True)).fillna(0.0)
    
    stl_base_seasonality_update = (g["stl_trend"] + g[grouping_col].map(seasonality_fill)).clip(lower=stl_cfg.clip_lower)

    # replaces missing values with STL estimate
    mask_missing = g[cols.count].isna()
    stl_adv = np.where(mask_missing, stl_base_seasonality_update.to_numpy(), stl_base.to_numpy())
    
    # final STL imputed column: fills original missing with advanced estimate
    g[out_cfg.col_stl_imputed] = g[cols.count].fillna(pd.Series(stl_adv, index=g.index))
    return g


def impute_stl(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
    stl_cfg: STLConfig = STLConfig(),
    out_cfg: OutputConfig = OutputConfig(),
) -> pd.DataFrame:

    """
    Applies STL on all counters

    ------
    Parameters:
    
    - df: full dataset
    - cols: columns config
    - stl_cfg: config for STL    
    - out_cfg: config for output columns' names

    -----
    Returns:

    - dataframe with imputed missing values, using STL
    """
    
    freq = _determine_temporal_frequency(df, cols=cols)
    
    out = _initial_interpolate_for_stl(df, cols=cols, out_cfg=out_cfg)
    
    out = out.groupby(cols.counter, group_keys=False).apply(
        _stl_on_counter_hole_adjusted, freq, cols, stl_cfg, out_cfg
    )

    return out