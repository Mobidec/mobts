"""
Module concerned with the donor-based imputations

This module contains:
- determining the minimum overlap period for scaled median imputation method based on project's temporal frequency
- building pivot tables for further operations, where timestamp would be index, counters as columns, and counts as values
- creating a correlation matrix of counters based on pearson correlation between counts
- scaled medians imputation
- regression imputation

"""

import numpy as np
import pandas as pd
from typing import Iterable, Optional
from sklearn.linear_model import LinearRegression

from ..configs.config_common import ColumnsConfig
from ..configs.config_imputation import STLConfig, DonorsConfig, OutputConfig
from .selector import _select_regression_donors, _get_min_mutual_period
from ..utils.formatting import _determine_temporal_frequency, _validate_frequency


def _get_min_overlap_period_sm(freq: str, donors_cfg: DonorsConfig = DonorsConfig()) -> int:
    """
    Determines the minimum overlap period necessary for scaled medians imputation

    ------
    Parameters:

    - freq: temporal frequency of the project
    - donors_cfg: donors' config

    -----
    Returns:

    - integar corresponding to the minimum necessary overlap period

    """

    freq = _validate_frequency(freq)
    if freq == 'hourly':
        return donors_cfg.sm_min_overlap_hour
    elif freq == 'daily':
        return donors_cfg.sm_min_overlap_day
    else:
        raise ValueError(f'Unsupported frequency: {freq}')


def _build_pivots(
    df: pd.DataFrame,
    cols: ColumnsConfig = ColumnsConfig(),
    stl_cfg: STLConfig = STLConfig(),
) -> pd.DataFrame:
    """
    builds pivots of data, where timestamp would be index, counters as columns, and counts as values

    ------
    Parameters:

    - df: full network DataFrame
    - cols: columns config
    - stl_cfg: STL config

    -----
    Returns:

    - pivot_raw: building a pivot based on raw observed counts
    - pivot_ts: building a pivot based on smoothed out time series of STL's trend + seasonality
    """

    out = df.copy()

    ts = (out['stl_trend'] + out['stl_season']).clip(lower=stl_cfg.clip_lower)
    out['_ts_'] = ts

    pivot_raw = out.pivot_table(index=cols.timestamp, columns=cols.counter, values=cols.count, aggfunc='mean')
    pivot_ts = out.pivot_table(index=cols.timestamp, columns=cols.counter, values='_ts_', aggfunc='mean')

    return pivot_raw, pivot_ts


def _corralation_matrix_donors(pivot_for_corr: pd.DataFrame) -> pd.DataFrame:
    """
    builds the correlation matrix of counters based on pearson correlation between counts, counters, and timestamps

    ------
    Parameters:

    - pivot_for_corr: the '_build_pivots' function's output, which is a pivot of counts

    -----
    Returns:

    - the correlation matrix of counters
    """

    corr = pivot_for_corr.corr()
    correlation_matrix = {s: corr[s].drop(labels=[s]).sort_values(ascending=False).index.tolist() for s in corr.columns}

    return correlation_matrix


def impute_scaled_median(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    donor_map: dict[str, list[str]],
    freq: str,
    counters=None,
    cols: ColumnsConfig = ColumnsConfig(),
    donors_cfg: DonorsConfig = DonorsConfig(),
    out_cfg: OutputConfig = OutputConfig(),
) -> pd.DataFrame:
    """
    Fills missing values using scaled median of donors (M7)

    ------
    Parameters:

    - df: the complete network dataset
    - pivot: pivotted dataset of counters
    - donor_map: dictionary map of donors
    - freq: temporal frequency of the project
    - counters: counters to be operated on. if NaN, all counters will be processed
    - cols: columns config
    - donors_cfg: donors' config
    - out_cfg: output config

    -----
    Returns:

    - Imputed DataFrame using scaled medians method (M7)

    -----
    Notes:

    - the 'counters' argument is added in order to be utilized through piepline, to skip counters which do not have data holes. this gives us the possibility to only process counters with holes
    """

    out = df.copy()

    # setting the imputed column as NaN to be filled later
    out[out_cfg.col_sm_imputed] = np.nan

    sm_min_overlap = _get_min_overlap_period_sm(freq=freq)

    targets = counters if counters is not None else donor_map.keys()

    # loop that goes through target counters, retrieves donors and identifies eligible ones, and calculates median
    for target in targets:
        donors = donor_map.get(target, [])

        if target not in pivot.columns:
            continue

        # a maximum of retrieved donors is set, to limit calculations on the entirety of donors (default set to 0.5)
        max_d = int(len(donors) * donors_cfg.max_donor_rate)
        donors = donors[:max_d]

        if not donors:
            continue

        # first checks if there are enough observations on the target itself
        y_t = pivot[target]
        avail_idx = y_t.index[y_t.notna()]

        if avail_idx.size < sm_min_overlap:
            continue

        median_target = np.nanmedian(y_t.loc[avail_idx])

        if not np.isfinite(median_target):
            continue

        # goes through each donor, checks validity, if valid -> adds the donor and its median to corresponding lists
        valid_donors = []
        donor_meds = []
        sm_counter = 0

        for d in donors:
            if d not in pivot.columns:
                continue

            if pivot[[target, d, *valid_donors]].notna().all(axis=1).sum() < sm_min_overlap:
                continue

            arr = pivot.loc[avail_idx, d].to_numpy(dtype=float)

            # infinity checks are run on multiple steps
            if np.isfinite(arr).any():
                md = np.nanmedian(arr)
                if np.isfinite(md):
                    valid_donors.append(d)
                    donor_meds.append(md)

            # once the counter hits 'top_k_donor', the donor loop ends
            sm_counter = +1

            if sm_counter == donors_cfg.top_k_donor:
                break

        median_donors = float(np.median(donor_meds))

        if not (np.isfinite(median_donors) and median_donors > 0):
            continue

        # the scale is used to fit median of donors to the target
        scale = median_target / median_donors

        if not np.isfinite(scale):
            continue

        # the mini pivot dataset of donors are produced
        mat = pivot[valid_donors].to_numpy(dtype=float)
        med_series = np.nanmedian(mat, axis=1) * scale
        donor_series = pd.Series(med_series, index=pivot.index)

        mask = (out[cols.counter] == target) & out[cols.count].isna()

        if mask.any():
            out.loc[mask, out_cfg.col_sm_imputed] = out.loc[mask, cols.timestamp].map(donor_series)

    out[out_cfg.col_sm_imputed] = out[cols.count].fillna(out[out_cfg.col_sm_imputed])

    return out


def impute_regression(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    freq: str,
    donor_map: dict[str, list[str]],
    counters=None,
    cols: ColumnsConfig = ColumnsConfig(),
    donors_cfg: DonorsConfig = DonorsConfig(),
    stl_cfg: STLConfig = STLConfig(),
    out_cfg: OutputConfig = OutputConfig(),
) -> pd.DataFrame:
    """
    Fills missing values using regression prediction of donors (M8)

    ------
    Parameters:

    - df: the complete network dataset
    - pivot: pivotted dataset of counters
    - donor_map: dictionary map of donors
    - freq: temporal frequency of the project
    - counters: counters to be operated on. if NaN, all counters will be processed
    - cols: columns config
    - donors_cfg: donors' config
    - out_cfg: output config
    - stl_cfg: STL config

    -----
    Returns:

    - Imputed DataFrame using regression method (M8)

    -----
    Notes:

    - the 'counters' argument is added in order to be utilized through piepline, to skip counters which do not have data holes. this gives us the possibility to only process counters with holes
    """

    # shortened values for ease of use
    s_col, d_col, v_col = cols.counter, cols.timestamp, cols.count

    # starts with NaN imputed column
    out = df.copy()
    pred_col = '_reg_pred_'
    out[pred_col] = np.nan

    min_mutual_period = _get_min_mutual_period(freq)

    targets = counters if counters is not None else donor_map.keys()

    # for each targer counter, gets donors using pre-defined function
    for target in targets:
        donors = donor_map.get(target, [])
        if target not in pivot.columns:
            continue

        selected = _select_regression_donors(target=target, pivot=pivot, freq=freq, donors=donors, donors_cfg=donors_cfg)

        # set y_imp as the pr
        y = pivot[target]
        y_imp = y.copy()

        if selected:
            X = pivot[selected]
            mask_fit = y.notna() & X.notna().all(axis=1)

            # masks to see if there are enough mutual observations between target and donors to build the model
            if mask_fit.sum() > min_mutual_period:
                # builds and fits the model
                model = LinearRegression()
                model.fit(X.loc[mask_fit], y.loc[mask_fit])

                # masks for prediction where target is null
                mask_pred = X.notna().all(axis=1)

                # replaces y_imp with the prediction (y_hat)
                if mask_pred.any():
                    y_hat = model.predict(X.loc[mask_pred])
                    y_hat = np.maximum(y_hat, stl_cfg.clip_lower)
                    y_imp.loc[mask_pred] = y_hat

        # updates the output for the target
        mask_rows = out[s_col] == target
        if mask_rows.any():
            out.loc[mask_rows, pred_col] = out.loc[mask_rows, d_col].map(y_imp)

    out[out_cfg.col_reg_imputed] = out[v_col].fillna(out[pred_col])
    out.drop(columns=[pred_col], inplace=True)

    return out
