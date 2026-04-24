"""
The plotting functions for checking the quality of outlier detection

This module contains:
- the plot function for hourly data
- the plot function for hourly data
"""



import math
import random
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..configs.config_common import ColumnsConfig
from ..configs.config_preprocessing import OutlierConfig, PlotConfig



def _plot_outliers_daily(
    df_scored: pd.Dataframe,
    threshold: float,
    counters: Optional[Iterable[str]] = None,
    max_counters: float = None,
    cols: ColumnsConfig = ColumnsConfig(),
    out_cfg: OutlierConfig = OutlierConfig(),
    plot_cfg: PlotConfig = PlotConfig(),
):

    """
    Plotting outliers for daily data

    ------
    Parameters:
    
    - df_scored: daily dataset with outlier score
    - threshold: outlier threshold set by user (has a default of 20)
    - counters: optional list of counters to be visualized
    - max_counters: maximum number of counters to be plotted, also optional
    - cols: Column config
    - out_cfg: outlier config
    - plot_cfg: plotting config

    -----
    Returns:

    - Figure visualizing time-series of counts for each counter, highlighting the outliers set by the given threshold
    """

    # threshold is set as default, but can be changed by user
    if threshold is None:
        thr = out_cfg.threshold_daily
    else:
        thr = threshold

    # potential addition of counter list given by user
    if counters is None:
        counters_list = list(pd.unique(df_scored[cols.counter]))
    else:
        counters_list = list(counters)

    # potential addition of maximum number of counters to be plotted
    if max_counters is not None:
        counters_list = random.sample(list(pd.unique(df_scored[cols.counter])), max_counters)

    # plotting the figure
    # number of cols, height per row, figsize width, likewidth and etc can be modified through the config
    n = len(counters_list)
    ncols = plot_cfg.ncols
    nrows = math.ceil(n / ncols)

    fig_height = max(plot_cfg.min_fig_height, plot_cfg.height_per_row * nrows)
    fig = plt.figure(figsize=(plot_cfg.figsize_width, fig_height))
    

    
    i=1
    for st in counters_list:

        ax = fig.add_subplot(nrows, ncols, i)
        df_plot = df_scored[df_scored[cols.counter] == st]
        x = df_plot.index
    
        ax.plot(x, df_plot["count_filled"], linewidth = plot_cfg.linewidth_d)
    
        mask = df_plot["out_score"] > thr
        ax.scatter(x[mask], df_plot.loc[mask, "count_filled"], color='r')
    
        ax.set_title(str(st))
        ax.tick_params(axis = "x", labelrotation = 30)
        i = i+1


    fig.tight_layout()
        
    return fig



def _plot_outliers_hourly(
    df_scored: pd.Dataframe,
    threshold: float,
    counters: Optional[Iterable[str]] = None,
    max_counters: float = None,
    cols: ColumnsConfig = ColumnsConfig(),
    out_cfg: OutlierConfig = OutlierConfig(),
    plot_cfg: PlotConfig = PlotConfig(),
):
    
    """
    Plotting outliers for hourly data

    ------
    Parameters:
    
    - df_scored: daily dataset with outlier score
    - threshold: outlier threshold set by user (has a default of 20)
    - counters: optional list of counters to be visualized
    - max_counters: maximum number of counters to be plotted, also optional
    - cols: Column config
    - out_cfg: outlier config
    - plot_cfg: plotting config

    -----
    Returns:

    - Figure visualizing time-series of counts for each counter, highlighting the outliers set by the given threshold
    """

    # threshold is set as default, but can be changed by user
    if threshold is None:
        thr = out_cfg.threshold_hourly
    else:
        thr = threshold

    # potential addition of counter list given by user
    if counters is None:
        counters_list = list(pd.unique(df_scored[cols.counter]))
    else:
        counters_list = list(counters)

    # potential addition of maximum number of counters to be plotted
    if max_counters is not None:
        counters_list = random.sample(list(pd.unique(df_scored[cols.counter])), max_counters)

    n = len(counters_list)
    ncols = plot_cfg.ncols
    nrows = math.ceil(n / ncols)

    fig_height = max(plot_cfg.min_fig_height, plot_cfg.height_per_row * nrows)
    fig = plt.figure(figsize=(plot_cfg.figsize_width, fig_height))

    # plotting the figure
    # number of cols, height per row, figsize width, likewidth and etc can be modified through the config
    i=1
    for st in counters_list:

        ax = fig.add_subplot(nrows, ncols, i)
        df_plot = df_scored[df_scored[cols.counter] == st]

        # zoom to the 1-month period with the most flagged anomalies
        mask_full = df_plot["out_score"] > thr

        if mask_full.any():
            month_key = df_plot.index.tz_localize(None).to_period("M")
            flagged_per_month = mask_full.astype(int).groupby(month_key).sum()
            best_month = flagged_per_month.idxmax()

            month_mask = month_key == best_month
            df_plot = df_plot.loc[month_mask]
            
        x = df_plot.index
    
        ax.plot(x, df_plot[cols.count], linewidth = plot_cfg.linewidth_h)
    
        mask = df_plot["out_score"] > thr        
        ax.scatter(x[mask], df_plot.loc[mask, cols.count], color='r')
    
        ax.set_title(str(st))
        ax.tick_params(axis = "x", labelrotation = 30)

        ax.grid(alpha=0.3, linestyle="--")

        ax.margins(x=0)
        
        i = i+1


    fig.tight_layout()
    return fig



# def plot_outliers(
#     df_scored,
#     cols: ColumnsConfig = ColumnsConfig(),
#     out_cfg: OutlierConfig = OutlierConfig(),
#     plot_cfg: PlotConfig = PlotConfig(),
#     data_is_hourly: bool = True,
#     change_to_daily: bool = False,
#     # gran_cfg: ProjectConfig = ProjectConfig(),
#     threshold: float = None,
#     counters: Optional[Iterable[str]] = None,
#     max_counters: int = None):


#     if data_is_hourly and not change_to_daily:
#         _plot__outliers_hourly(df_scored, cols, out_cfg, plot_cfg, threshold, counters, max_counters)

#     elif (data_is_hourly and change_to_daily) or not data_is_hourly:
#         _plot_outliers_daily(df_scored, cols, out_cfg, plot_cfg, threshold, counters, max_counters)

#     else:
#         raise ValueError(f"'{pr_gran}' is not a valid granularity. Try 'h' or 'D'.")
        

    

    
    
