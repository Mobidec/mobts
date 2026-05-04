"""
Calculating and assigning an outlier score to each observation

This module contains:
- calculating outlier score for daily data
- calculating outlier score for hourly data
- applying outlier score to all counters in the network
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from ..configs.config_common import ColumnsConfig
from ..configs.config_preprocessing import STLConfig


def _calculate_outlier_score_counter_daily(
    df_counter_daily: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg: STLConfig = STLConfig(),
) -> pd.DataFrame:
    """
    Calculating outlier score for daily data

    ------
    Parameters:

    - df_counter_daily: daily dataset for a single counter
    - cols: Column config
    - cfg: STL config

    -----
    Returns:

    - DataFrame of single counter with aggregate daily observations, and corresponding outlier scores

    -----
    Notes:
    - STL's residuals are used for calculating an outlier score.
    - To avoid assigning high scores to observations that are prevalent, the outlier score is increased for quantile 99,
      and is decreased for observations between 25th and 75th quantiles
    """

    df = df_counter_daily.copy()

    # an initial "dumb" imputation so that STL can run with no NaN values
    df['count_filled'] = df[cols.count].interpolate(method='time').bfill().ffill()

    # running the STL on the filled counts, period is 7 because data is daily
    stl = STL(df['count_filled'], period=7, robust=cfg.robust)
    res = stl.fit()

    # extracting residuals for outlier detection
    resid = res.resid
    df['resid'] = resid

    # getting the average and std of the residuals
    average = np.mean(resid)
    std = np.std(np.abs(resid - average))

    # guarding against divide-by-zero on flat/constant series, else calculating the outlier score
    df['out_score'] = np.nan if std == 0 else (np.abs(df['resid'] - average) / std)

    # global percentile thresholds over all observations of this counter
    q25 = df[cols.count].quantile(0.25)
    q75 = df[cols.count].quantile(0.75)
    q99 = df[cols.count].quantile(0.99)

    # weight adjustment
    w = np.ones(len(df), dtype=float)

    # increase score for top 1% observations
    w[df[cols.count] >= q99] *= 2.5

    # reduce score for observations between 25th and 75th percentile
    mid_mask = (df[cols.count] >= q25) & (df[cols.count] <= q75)
    w[mid_mask] *= 0.1

    df['out_score'] = df['out_score'] * w

    return df


def _calculate_outlier_score_counter_hourly(
    df_couter_hourly: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:
    """
    Calculating outlier score for hourly data

    ------
    Parameters:

    - df_counter_hourly: hourly dataset for a single counter
    - cols: Column config

    -----
    Returns:

    - DataFrame of single counter with aggregate hourly observations, and corresponding outlier scores

    -----
    Notes:
    - Compared to daily outlier score calculation, a rather simple method has been adopted for hourly data, mainly because
      STL is computationally heavy on hourly data, with a period of 168 hours. Furthermore,
    - To avoid assigning high scores to observations that are prevalent, the outlier score is increased for quantile 99,
      and is decreased for observations between 25th and 75th quantiles
    """

    df = df_couter_hourly.copy()

    # calculating median of each hour, absolute deviation of each observation from said median, and MAD
    med = df.groupby(cols.hour)[cols.count].transform('median')
    dev = np.abs(df[cols.count] - med)
    mad = dev.groupby(df[cols.hour]).transform('median')

    # robust std estimate
    scale = 1.4826 * mad

    # getting the outlier score by dividing the difference of counts by the scale, and avoid divide by zero
    df['out_score'] = np.abs(df[cols.count] - med) / scale
    df.loc[scale == 0, 'out_score'] = np.nan

    # global percentile thresholds over all observations of this counter
    q25 = df[cols.count].quantile(0.25)
    q75 = df[cols.count].quantile(0.75)
    q99 = df[cols.count].quantile(0.99)

    # weight adjustment
    w = np.ones(len(df), dtype=float)

    # increase score for top 1% observations
    w[df[cols.count] >= q99] *= 2.5

    # reduce score for observations between 25th and 75th percentile
    mid_mask = (df[cols.count] >= q25) & (df[cols.count] <= q75)
    w[mid_mask] *= 0.1

    df['out_score'] = df['out_score'] * w

    df.drop(columns=cols.how, inplace=True)

    return df


def _calculate_outlier_score(
    df: pd.DataFrame,
    data_is_hourly: bool = True,
    change_to_daily: bool = False,
    cols: ColumnsConfig = ColumnsConfig(),
    stl_cfg: STLConfig = STLConfig(),
) -> pd.DataFrame:
    """
    Calculating outlier score for each observation based on temporal frequency

    ------
    Parameters:

    - df: complete dataset
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data
    - cols: Column config
    - stl_cfg: config for STL

    -----
    Returns:

    - DataFrame of all observations with their corresponding outlier score

    -----
    Notes:
    - The scales of scores between hourly and daily datasets are not necessarily similar, this is reflected on
      the choice of outlier threshold downstream in the code
    """

    df = df.copy().sort_values([cols.counter, cols.timestamp])

    # going through each counter, verifying temporal frequency, and applying the corresponding function
    out = []
    for counter, g in df.groupby(cols.counter):
        g = g.copy()
        g = g.set_index(cols.timestamp)

        if data_is_hourly and not change_to_daily:
            scored = _calculate_outlier_score_counter_hourly(g, cols=cols)

        else:
            scored = _calculate_outlier_score_counter_daily(g, cols=cols, cfg=stl_cfg)

        scored[cols.counter] = counter
        out.append(scored)

    # concat is applied directly at the output
    return pd.concat(out)
