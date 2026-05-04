 """
Cleaning the dataset for further operations

This module contains:
- transforming the hourly dataset to daily dataset
- removing measurement errors for data with hourly frequency per counter
- removing measurement errors for data with hourly frequency per counter
- a wrapper for removing measurement errors per counter
- function for removing measurement errors for the entire network
"""



import numpy as np
import pandas as pd

from ..configs.config_common import ColumnsConfig, SparsityConfig
from ..configs.config_preprocessing import PreprocessConfig
from ..utils.formatting import _add_temporal_columns



def _aggregate_hourly_to_daily(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
) -> pd.DataFrame:

    """
    Aggregate hourly frequency to daily frequency

    ------
    Parameters:
    
    - df: hourly dataset
    - cols: Column config

    -----
    Returns:

    - dataframe with daily frequency
    """

    daily = df.groupby(cols.counter).resample('D', on=cols.timestamp, include_groups=False)[cols.count].sum().reset_index()

    return daily



def _remove_measurement_errors_hourly(
    df_counter: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg: PreprocessConfig = PreprocessConfig(),
) -> pd.DataFrame:

    """
    Remove measurement errors for data with hourly frequency per counter
    
    ------
    Parameters:
    
    - df_counter: hourly dataset of a single counter
    - cols: Column config
    - cfg: config for preprocessing

    -----
    Returns:

    - Single counter dataframe where hourly counts suspcious of being made with measurement errors are set to NaN

    -----
    Notes:

    - The process includes removing counts in following conditions:
                    1. if a counter has recorded 0 counts for en entire day, the entire day will be set to NaN.
                    2. for an hour's count to be set to zero, it has to be outside night hours, and the rate of 0 observations for that 
                       hour needs to be lower than 'zero_rate_max'. additionally, the median observation of the hour has to be at least one
                       standard deviation higher than zero.
                    3. after the conditions mentioned above (2), only 0 records that are consequent for 'zero_run_min' hours are set to NaN.
                    4. observation islands with a duration lower than 'island_max_len' that are surrounded by zero or NaN observations lengthier
                       than 'surround_min_len' are also set to NaN.

    - Considering noise and variations of hourly data, strict conditions are set as explained above to prevent unnecessary loss of data.
    """
    
    df = df_counter.copy()

    # addition of temporal columns. important here: 'how', 'hour', and 'date'
    df = _add_temporal_columns(df, freq="hourly", cols=cols)
    
    how = df[cols.how]
    hour = df[cols.hour]
    date = df[cols.date]
    x = df[cols.count].astype(float)

    # entire date is 0 -> NaN (applies regardless of night hours) 
    day_all_zero = x.eq(0).groupby(date).transform("all")
    x = x.mask(day_all_zero & x.eq(0), np.nan)

    #  per-hour eligibility for converting zeros
    g = x.groupby(hour)
    zero_rate = g.apply(lambda s: s.eq(0).mean()).reindex(range(24), fill_value=0.0)
    med = g.median()
    sd = g.std(ddof=0)

    hour_ok = hour.map(lambda h: (zero_rate.get(h, 0.0) < cfg.zero_rate_max) and (h not in cfg.night_hours))
    stats_ok = hour.map(lambda h: (med.get(h, np.nan) - sd.get(h, np.nan)) > 0)

    candidate_zero = x.eq(0) & hour_ok & stats_ok

    # only convert zeros if they are in runs >= zero_run_min 
    run_id = (candidate_zero != candidate_zero.shift()).cumsum()
    run_len = candidate_zero.groupby(run_id).transform("sum")
    x = x.mask(candidate_zero & (run_len >= cfg.zero_run_min), np.nan)

    # remove small non-zero islands surrounded by long gaps (0 or NaN) 
    is_gap = x.isna() | x.eq(0)
    rid = (is_gap != is_gap.shift()).cumsum()
    rlen = is_gap.groupby(rid).transform("size")

    # runs of non-gaps ("islands")
    is_island = ~is_gap
    island_id = (is_island != is_island.shift()).cumsum()
    island_len = is_island.groupby(island_id).transform("sum")

    # gap lengths just before/after each timestamp
    gap_len = rlen.where(is_gap, 0)
    prev_gap = gap_len.shift().fillna(0)
    next_gap = gap_len.shift(-1).fillna(0)

    # mark points inside an island that is short and flanked by long gaps
    island_points = (
        is_island
        & (island_len <= cfg.island_max_len)
        & (prev_gap >= cfg.surround_min_len)
        & (next_gap >= cfg.surround_min_len)
    )
    x = x.mask(island_points, np.nan)

    df[cols.count] = x
    return df
    


def _remove_measurement_errors_daily(
    df_counter: pd.Dataframe,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg:PreprocessConfig = PreprocessConfig(),
) -> pd.Dataframe:
                                    
    """
    Remove measurement errors for data with daily frequency per counter
    
    ------
    Parameters:
    
    - df_counter: daily dataset of a single counter
    - cols: column config
    - cfg: config for preprocessing

    -----
    Returns:

    - Single counter dataframe where daily counts suspcious of being made with measurement errors are set to NaN

    -----
    Notes:

    - The process includes removing counts in following conditions:
                    1. days recorded with 0 observations are set to NaN
                    2. a threshold is calculated for low-observation noise. this threshold is the maximum of a pre-set 'low_abs_daily' counts,
                       and 'low_rel_daily' (in %) of a baseline that is defined as the median of all counts. observations under this threshold
                       are considered as "low counts". if these low count observations persist for longer than 'low_run_min_daily', they will
                       be set to NaN as well.

    - Considering the aggregate nature of daily data, compared to hourly data, the removal of 0s and low counts are more generously applied.
    """
    
    df = df_counter.copy()

    # addition of temporal columns. important here: 'weekday', 'week_num'
    df = _add_temporal_columns(df, freq="daily", cols=cols)
    
    # calculate the median as baseline, and a threshold defined in parameters (1% here)
    baseline = df[cols.count].median(skipna=True)
    thr = max(cfg.low_abs_daily, cfg.low_rel_daily * baseline) if pd.notna(baseline) else cfg.low_abs_daily

    # identify running low count observations
    is_low = df[cols.count].le(thr)
    run_id = (is_low != is_low.shift()).cumsum()
    run_len = is_low.groupby(run_id).transform("sum")

    # set the identified elements to nan
    df.loc[is_low & (run_len >= cfg.low_run_min_daily), cols.count] = np.nan
    df.loc[df[cols.count] == 0, cols.count] = np.nan

    return df


def _remove_measurement_errors_wrapper_counter(
    df_counter: pd.Dataframe,
    data_is_hourly: bool,
    change_to_daily: bool,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg: PreprocessConfig = PreprocessConfig(),
) -> pd.Dataframe:

    """
    Wrapper for choosing the measurement error remover by frequency for a single counter
    
    ------
    Parameters:
    
    - df_counter: daily dataset of a single counter
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data
    - cols: column config
    - cfg: config for preprocessing

    -----
    Returns:

    - Single counter dataframe where counts suspcious of being made with measurement errors are set to NaN
    """

    # first the frequency should be identified
    if data_is_hourly and not change_to_daily:
        freq = "hourly"
        
    elif change_to_daily or not data_is_hourly:
        freq = "daily"
        
    else:
        raise ValueError("Temporal frequency is not valid. Enter 'data_is_hourly' and 'change_to_daily' accordingly.")
    
    df = df_counter.copy()

    # creating a full range
    if freq == "daily":
        range_freq = "D"
    elif freq == "hourly":
        range_freq = "h"
        
    start, end = df[cols.timestamp].min(), df[cols.timestamp].max()
    full_range = pd.date_range(start=start, end=end, freq=range_freq)
    
    df = df.set_index(cols.timestamp)                       # timestamp becomes the index (and is dropped as a column)
    df = df.reindex(full_range)                             # align to full range
    df = df.rename_axis(cols.timestamp).reset_index()       # index -> column named timestamp
    
    # applying the correct function based on frequency
    if freq == "daily":
        out = _remove_measurement_errors_daily(df, cols = cols, cfg = cfg)

    if freq == "hourly":
        out = _remove_measurement_errors_hourly(df, cols = cols, cfg = cfg)

    # getting the number of cleaned observations, and number of counters where any observations have been cleaned
    changed_to_nan = df[cols.count].notna() & out[cols.count].isna()

    n_obs_changed_to_nan = int(changed_to_nan.sum())
    n_counters_affected = int(out.loc[changed_to_nan, cols.counter].nunique())

    return out


def _remove_measurement_errors_all_network(
    df_network: pd.Dataframe,
    data_is_hourly: bool = True,
    change_to_daily: bool = False,
    cols: ColumnsConfig = ColumnsConfig(),
    cfg: PreprocessConfig = PreprocessConfig(),
) -> pd.Dataframe:

    """
    Applying the cleaning function on the entire network
    
    ------
    Parameters:
    
    - df_network: the dataframe containing all counters
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data
    - cols: column config
    - cfg: config for preprocessing

    -----
    Returns:

    - Single counter dataframe where counts suspcious of being made with measurement errors are set to NaN
    - number of observations that have been changed to NaN by the function
    - number of counters that have been affected by the cleaning function
    """

    df = df_network.copy()

    # run the cleaning process on each counter and then concat
    counters_out = []
    for counter, g in df.groupby(cols.counter):
        st = _remove_measurement_errors_wrapper_counter(g, data_is_hourly, change_to_daily, cols=cols, cfg=cfg)
        st[cols.counter] = counter
        counters_out.append(st)

    out = pd.concat(counters_out).reset_index(drop=True)

    # getting the number of cleaned observations, and number of counters where any observations have been cleaned
    changed_to_nan = df[cols.count].notna() & out[cols.count].isna()
    
    n_obs_changed_to_nan = int(changed_to_nan.sum())
    n_counters_affected = int(out.loc[changed_to_nan, cols.counter].nunique())

    return out, n_obs_changed_to_nan, n_counters_affected
