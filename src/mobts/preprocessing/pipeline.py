"""
The pipeline for preprocessing subpackage
----------

This module contains:
- the function for the first stage of preprocessing, which includes:
    + standardizing the input given by the user
    + removing observations with undefined counter names (optional)
    + aggregate hourly to daily frequency (optional)
    + remove counters with sparse observations
    + calculate outlier scores for each observation
- the function for applying user's thresold, flagging outliers, and replacing them with NaN

- the 'preprocess' class containing a .run function that contains:
    + running the stage 1 procedure explained above
    + flagging and replacing outliers with NaN using given threshold
    + the plot function for visualizing detected outliers based on given threshold

It is worth noting that the main class is supposed to be called, however, a warning is given to user that
the default thresholds are set based on the study case data, and might not be suitable for other datasets.
The warning suggests the user to run 'run_preprocess_stage_1', and then tweak the threshold while monitoring
the outlier detection quality via the 'plot_outliers' function. The final dataset can be directly produced
after, using the 'apply_threshold' function.
"""


import pandas as pd
import numpy as np
import warnings

from ..configs.config_preprocessing import PipelineConfig
from ..utils.formatting import _standardize_input, _drop_nan_counters, _drop_sparse_counters, _format_datetime
from .cleaning import _aggregate_hourly_to_daily, _remove_measurement_errors_all_network
from .outliers import _calculate_outlier_score
from .plotting import _plot_outliers_hourly, _plot_outliers_daily


def run_preprocess_stage_1(
    df_raw: pd.DataFrame,
    counter_col: str,
    timestamp_col: str,
    count_col: str,
    cfg: PipelineConfig = PipelineConfig(),
    data_is_hourly: bool = True,
    change_to_daily: bool = False,
) -> pd.DataFrame:

    """
    First stage of preprocessing, from raw data to outlier score

    ------
    Parameters:
    
    - df_raw: raw data provided by user
    - counter_col: counter column's name fed by user
    - timestamp_col: timestamp column's name fed by user
    - count_col: count column's name fed by user
    - cfg: pipeline config containing all configs
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data

    -----
    Returns:

    - Figure visualizing time-series of counts for each counter, highlighting the outliers set by the given threshold
    """

    df_std = _standardize_input(
        df_raw,
        counter_col=counter_col,
        timestamp_col=timestamp_col,
        count_col=count_col,
        out_cols=cfg.cols)

    
    df_std, n_missing_names = _drop_nan_counters(df_std)

    df_std = _format_datetime(df_std)

    if data_is_hourly & change_to_daily:
        df_std = _aggregate_hourly_to_daily(df_std.copy(), cols=cfg.cols)

    df_clean, n_obs_changed_to_nan, n_counters_affected = _remove_measurement_errors_all_network(
        df_std,
        cols = cfg.cols,
        cfg = cfg.preprocess,
        data_is_hourly = data_is_hourly,
        change_to_daily = change_to_daily)

    if cfg.sparse.drop_sparse_counters:
        df_clean_non_sparse = _drop_sparse_counters(df_clean)

    n_sparse_counters = int(df_clean[cfg.cols.counter].nunique()) - int(df_clean_non_sparse[cfg.cols.counter].nunique())

    df_scored = _calculate_outlier_score(
        df_clean_non_sparse,
        data_is_hourly=data_is_hourly,
        change_to_daily=change_to_daily,
        cols=cfg.cols,
        stl_cfg=cfg.stl)

    return df_scored, n_missing_names, n_obs_changed_to_nan, n_counters_affected, n_sparse_counters


def apply_threshold(
    df_scored: pd.DataFrame,
    data_is_hourly: str,
    change_to_daily: str,
    cfg: PipelineConfig = PipelineConfig(),
    threshold: float | None = None,
) -> pd.DataFrame:

    """
    Applying the given threshold and replacing the outliers withy NaN

    ------
    Parameters:
    
    - df_scored: dataframe with outlier scores
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data
    - cfg: pipeline config containing all configs
    - threshold: outlier threshold set by user (defaults still set in config)

    -----
    Returns:

    - Cleaned final dataset with measurement errors removed and replaced by NaN
    """
    
    if data_is_hourly and not change_to_daily:
        threshold_default = cfg.outliers.threshold_hourly
        
    elif change_to_daily or not data_is_hourly:
        threshold_default = cfg.outliers.threshold_daily
    
    # additional check for if threshold is not defined by the user
    thr = threshold_default if threshold is None else threshold
    cols = cfg.cols

    out = df_scored.copy()
    out.loc[out["out_score"] > thr, cols.count] = np.nan

    # additional check to bring the timestamp into columns, in case it is still set as index
    if cols.timestamp not in out.columns:
        out = out.reset_index()

    # getting the number of measurement errors, and number of counters where any observations have been cleaned
    removed_outliers = (df_scored["out_score"] > thr) & df_scored[cols.count].notna()
    
    n_measurement_errors = int(removed_outliers.sum())
    n_counters_with_measurement_errors = int(df_scored.loc[removed_outliers, cols.counter].nunique())

    return out[[cols.counter, cols.timestamp, cols.count, "out_score"]].copy(), n_measurement_errors, n_counters_with_measurement_errors, thr



class preprocess:

    """
    This class consists of:
    + running the stage 1 procedure explained above
    + flagging and replacing outliers with NaN using given threshold

    ------
    Parameters:
    
    - df_raw: raw data provided by user
    - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
    - change_to_daily: indicator of if we are aggregating hourly to daily data
    - counter_col: counter column's name fed by user
    - timestamp_col: timestamp column's name fed by user
    - count_col: count column's name fed by user
    - threshold: outlier threshold set by user (defaults still set in config)
    - cfg: pipeline config containing all configs

    -----
    Returns:

    - Cleaned final dataset with measurement errors removed and replaced by NaN

    -----
    Notes:
     - Here, the function is run either by the default thresholds, or a threshold PREVIOUSLY OPTIMIZED by the user
    """
    
    def __init__(
        self,
        cfg: PipelineConfig = PipelineConfig(),
    ):
        
        self.cfg = cfg
        self.report_info = None
        self.output_plot = None

    def run(
        self,
        df_raw: pd.DataFrame,
        counter_col: str,
        timestamp_col: str,
        count_col: str,
        data_is_hourly: bool = True,
        change_to_daily: bool = False,
        threshold: float | None = None,
    ) -> pd.DataFrame:

        try:
            df_scored, n_missing_names, n_obs_changed_to_nan, n_counters_affected, n_sparse_counters = run_preprocess_stage_1(
                df_raw = df_raw,
                counter_col = counter_col,
                timestamp_col = timestamp_col,
                count_col = count_col,
                cfg = self.cfg,
                data_is_hourly = data_is_hourly,
                change_to_daily = change_to_daily,
            )

            out, n_measurement_errors, n_counters_with_measurement_errors, threshold = apply_threshold(
                df_scored = df_scored,
                data_is_hourly=data_is_hourly,
                change_to_daily=change_to_daily,
                cfg = self.cfg,
                threshold = threshold,
            )

            # save info for report
            self.report_info = {
                "n_entries" : int(len(df_raw)),
                "n_counters": int(out[self.cfg.cols.counter].nunique()) + n_sparse_counters,
                "n_missing_names": n_missing_names,
                "n_obs_changed_to_nan": n_obs_changed_to_nan,
                "n_counters_affected": n_counters_affected,
                "n_sparse_counters": n_sparse_counters,
                "n_measurement_errors": n_measurement_errors,
                "n_counters_with_measurement_errors": n_counters_with_measurement_errors,
                "data_is_hourly": data_is_hourly,
                "change_to_daily": change_to_daily,
                "threshold": threshold,
            }

            # save for plot
            self.output_plot = {
                "df_scored": df_scored,
                "data_is_hourly": data_is_hourly,
                "change_to_daily": change_to_daily,
                "threshold": threshold
            }

            return out


        except Exception as e:
            raise e

            
        finally:
            # warnings.simplefilter("always", UserWarning)  # always show
            warnings.warn("The default threshold is estimated based on a specific dataset, set to 20 for daily data, and 45 for hourly data. To optimize the threshold and the quality of outlier detection for measurement errors, plot with 'plot_outliers' function with a desired threshold to observe and change the threshold as needed")


    def report(
        self,
        print_output: bool = True,
        save: bool = False,
        filepath: str = "preprocess_report.txt",
    ) -> dict:
        
        """
        Returns a dictionary containing summary information from the latest run.

        -----
        Parameters
        
        - print_output : boolean for printing the operation info
        - save : boolean for saving the info in a text file
        - filepath : Path of the text file to save, default="preprocess_report.txt"

        -----
        Returns
        
        - Dictionary with summary information from the latest pipeline run.
        """

        if self.report_info is None:
            raise ValueError("No report information found. Run the pipeline first using the '.run' function.")

        info = self.report_info

        lines = [
            "=== PREPROCESS REPORT ===",
            f"Aggregate number of entries: {info['n_entries']}",
            f"Number of counters observed: {info['n_counters']}",            
            f"Number of entries with missing counter names: {info['n_missing_names']}",
            f"Number of zero/low entries cleaned: {info['n_obs_changed_to_nan']}",
            f"Number of counters with cleaned zero/low entries: {info['n_counters_affected']}",
            f"Number of counters removed due to low number of entries: {info['n_sparse_counters']}",
            f"Number of measurement errors flagged by threshold: {info['n_measurement_errors']}",
            f"Number of ounters with measurement errors: {info['n_counters_with_measurement_errors']}",
            f"Is data hourly: {info['data_is_hourly']}",
            f"Have hourly data been transformed to daily data: {info['change_to_daily']}",
            f"Threshold used: {info['threshold']}",
        ]

        text = "\n".join(lines)

        if print_output:
            print(text)

        if save:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

    def plot_outliers(
        self,
        # data_is_hourly: bool = True,
        # change_to_daily: bool = False,
        threshold: float = None,
        counters: Optional[Iterable[str]] = None,
        max_counters: int = None,
        # cfg: PipelineConfig = PipelineConfig(),
    ):
    
        """
        Plotting the outliers flagged for the counters
    
        ------
        Parameters:
        
        - df_scored: dataframe with outlier scores
        - data_is_hourly: indicator of if data is hourly or not (daily otherwise)
        - change_to_daily: indicator of if we are aggregating hourly to daily data
        - threshold: outlier threshold set by user (defaults still set in config)
        - counters: optional list of counters to be visualized
        - max_counters: maximum number of counters to be plotted, also optional
        - cfg: pipeline config containing all configs
    
        -----
        Returns:
    
        - Figure visualizing time-series of counts for each counter, highlighting the outliers set by the given threshold
        """

        if self.report_info is None:
            raise ValueError("No report information found. Run the pipeline first using the '.run' function.")
                    
        df_scored = self.output_plot["df_scored"]
        data_is_hourly = self.output_plot["data_is_hourly"]
        change_to_daily = self.output_plot["change_to_daily"]
        if threshold is None:
            threshold = self.output_plot["threshold"]

    
        if data_is_hourly and not change_to_daily:
            _plot_outliers_hourly(df_scored=df_scored, threshold=threshold, counters=counters, max_counters=max_counters, cols=self.cfg.cols, out_cfg=self.cfg.outliers, plot_cfg=self.cfg.plot)
    
        elif (data_is_hourly and change_to_daily) or not data_is_hourly:
            _plot_outliers_daily(df_scored=df_scored, threshold=threshold, counters=counters, max_counters=max_counters, cols=self.cfg.cols, out_cfg=self.cfg.outliers, plot_cfg=self.cfg.plot)
    
        else:
            raise ValueError("No valid temporal frequency. Try 'h' or 'D'.")

            
    