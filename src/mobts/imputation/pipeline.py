"""
The pipeline for imputation subpackage
----------

This module contains the 'impute' class. It's 'run' function includes:
- formatting and verifying the temporal elements of the input dataset
- identifying counters with holes
- applying STL, scaled medians imputation, and donor regression imputation
"""

import warnings
import numpy as np
import pandas as pd
from typing import Iterable, Optional
import timeit

from ..configs.config_common import ColumnsConfig, SparsityConfig
from ..configs.config_imputation import STLConfig, DonorsConfig, OutputConfig
from ..utils.formatting import _standardize_input, _drop_nan_counters, _format_datetime, _add_temporal_columns, _determine_temporal_frequency, _validate_frequency, _drop_sparse_counters
from .stl import impute_stl
from .donors import _build_pivots, _corralation_matrix_donors, impute_scaled_median, impute_regression
from .selector import _counter_method_choice, _find_counters_with_holes


class impute:
    """
    End-to-end pipeline of input data to imputed data

    ------
    Parameters:


    - df: input dataset
    - cols: columns config
    - stl_cfg: STL config
    - donors_cfg: donors config
    - out_cfg: output config
    - suppress_runtime_warnings: boolean for suppressing warnings


    -----
    Returns:
    - imputed dataset
    """

    def __init__(
        self,
        cols: ColumnsConfig = ColumnsConfig(),
        stl_cfg: STLConfig = STLConfig(),
        donors_cfg: DonorsConfig = DonorsConfig(),
        out_cfg: OutputConfig = OutputConfig(),
        cfg_spr: SparsityConfig = SparsityConfig(),
        suppress_runtime_warnings: bool = True,
    ):
        self.cols = cols
        self.stl_cfg = stl_cfg
        self.donors_cfg = donors_cfg
        self.out_cfg = out_cfg
        self.cfg_spr = cfg_spr
        self.suppress_runtime_warnings = suppress_runtime_warnings
        self.report_info = None

    # warnings for nan medians are numerous, cleaned for a cleaner output
    def run(self,
            df: pd.DataFrame,
            counter_col: str,
            timestamp_col: str,
            count_col: str,
            metadata_cols: list | None = None,
    ) -> pd.DataFrame:

        # to make the code cleaner
        c_col, t_col, v_col = self.cols.counter, self.cols.timestamp, self.cols.count

        ctx = warnings.catch_warnings()

        if self.suppress_runtime_warnings:
            ctx.__enter__()
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='DataFrameGroupBy.apply operated on the grouping columns', category=FutureWarning)

        try:
            # gets a meta for additional columns
            if metadata_cols:
                meta = df[[counter_col] + metadata_cols].drop_duplicates(subset=[counter_col])
                
            # first step, runs a mini-processing for preparing the dataset
            df_std = _standardize_input(df, counter_col=counter_col, timestamp_col=timestamp_col, count_col=count_col, out_cols=self.cols)

            df_std, n_missing_names = _drop_nan_counters(df_std)

            df_non_sparse = _drop_sparse_counters(df=df_std, cols=self.cols, cfg_spr=self.cfg_spr)

            n_sparse_counters = int(df_std[self.cols.counter].nunique()) - int(df_non_sparse[self.cols.counter].nunique())
            
            n_holes = int(df_non_sparse[self.cols.count].isna().sum())

            df_non_sparse = _format_datetime(df_non_sparse, cols=self.cols)
            freq = _determine_temporal_frequency(df_non_sparse, cols=self.cols)
            freq = _validate_frequency(freq)

            out_preprocessed = _add_temporal_columns(df_non_sparse, freq=freq, cols=self.cols)

            counters_holes = _find_counters_with_holes(out_preprocessed, count_col=v_col, counter_col=c_col)

            # STL imputation is implemented earlier, as its outputs are required for donor methods
            out_stl = impute_stl(out_preprocessed, cols=self.cols, stl_cfg=self.stl_cfg, out_cfg=self.out_cfg)

            # just a check for calculation time
            stl_time = timeit.default_timer()

            # pivoted forms of data, to be used by donor methods
            pivot_raw, pivot_ts = _build_pivots(out_stl, cols=self.cols, stl_cfg=self.stl_cfg)

            # smoothed time series is used for hourly data as it's much noisier
            if freq == 'hourly':
                pivot = pivot_ts

            elif freq == 'daily':
                pivot = pivot_raw

            donor_map = _corralation_matrix_donors(pivot_ts)

            # method_by_counter = {
            #     st: _counter_method_choice(target=st, pivot=pivot_raw, donor_map=donor_map, freq=freq, donors_cfg=self.donors_cfg, out_cfg=self.out_cfg) for st in counters_holes
            # }

            # scaled medians is only used for daily data, as STL shows superior for hourly data

            out_sm = (
                impute_scaled_median(
                    df=out_stl,
                    pivot=pivot,
                    donor_map=donor_map,
                    freq=freq,
                    counters=counters_holes,
                    cols=self.cols,
                    donors_cfg=self.donors_cfg,
                    out_cfg=self.out_cfg,
                )
                if freq == 'daily'
                else None
            )

            # regression imputation is optimal for both temporal frequencies, hence is done for both
            out_reg = impute_regression(
                df=out_stl,
                pivot=pivot,
                donor_map=donor_map,
                freq=freq,
                counters=counters_holes,
                cols=self.cols,
                donors_cfg=self.donors_cfg,
                stl_cfg=self.stl_cfg,
                out_cfg=self.out_cfg,
            )

            # regression, scaled median (for daily) and STL imputations are imported from respective outputs and are added to the final output dataset
            reg = out_reg[self.out_cfg.col_reg_imputed]
            sm = out_sm[self.out_cfg.col_sm_imputed] if freq == 'daily' else None
            stl = out_stl[self.out_cfg.col_stl_imputed]

            # retrieves the pre-imputed DataFrame
            out = out_preprocessed.copy()

            # adds each imputation column to the original DataFrame
            out[self.out_cfg.col_stl_imputed] = stl
            out[self.out_cfg.col_sm_imputed] = sm if freq == 'daily' else None
            out[self.out_cfg.col_reg_imputed] = reg

            out[self.out_cfg.col_final] = out[v_col].copy()

            if freq == 'daily':
                out[self.out_cfg.col_final] = out[self.out_cfg.col_final].fillna(reg).fillna(sm).fillna(stl)

            if freq == 'hourly':
                out[self.out_cfg.col_final] = out[self.out_cfg.col_final].fillna(reg).fillna(stl)

            # method used per row
            used = np.full(len(out), 'unfilled', dtype=object)
            used[out[v_col].notna().to_numpy()] = 'observed'

            miss = out[v_col].isna().to_numpy()
            final = out[self.out_cfg.col_final]

            used[miss & final.eq(reg).to_numpy() & reg.notna().to_numpy()] = 'donor_reg'
            if freq == 'daily':
                used[miss & final.eq(sm).to_numpy() & sm.notna().to_numpy() & ~final.eq(reg).to_numpy()] = 'donor_sm'
                used[miss & final.eq(stl).to_numpy() & stl.notna().to_numpy() & ~final.eq(reg).to_numpy() & ~final.eq(sm).to_numpy()] = (
                    'STL'
                )
            if freq == 'hourly':
                used[miss & final.eq(stl).to_numpy() & stl.notna().to_numpy() & ~final.eq(reg).to_numpy()] = 'STL'

            out[self.out_cfg.col_method_used] = used

            # save info for report
            self.report_info = {
                'n_entries': len(df),
                'n_counters': int(out[self.cols.counter].nunique()) + n_sparse_counters,
                'freq': freq,
                'n_missing_names': n_missing_names,
                'n_sparse_counters': n_sparse_counters,
                'n_holes': n_holes,
                'n_counter_holes': len(counters_holes),
                'n_reg': int((out[self.out_cfg.col_method_used] == 'donor_reg').sum()),
                'n_sm': int((out[self.out_cfg.col_method_used] == 'donor_sm').sum()),
                'n_stl': int((out[self.out_cfg.col_method_used] == 'STL').sum()),
            }

            # change column names back to original
            out = out.rename(columns={self.cols.counter: counter_col, self.cols.count: count_col, self.cols.timestamp: timestamp_col})

            # if metadata columns are provided, re-establish them
            if metadata_cols:
                out = out.merge(meta, on=counter_col, how='left')
                
            return out

        finally:
            if self.suppress_runtime_warnings:
                ctx.__exit__(None, None, None)

    def report(
        self,
        print_output: bool = True,
        save: bool = False,
        filepath: str = 'preprocess_report.txt',
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
            '=== Imputation REPORT ===',
            f'Aggregate number of entries: {info["n_entries"]}',
            f'Number of counters observed: {info["n_counters"]}',
            f'The temporal granularity of the project is: {info["freq"]}',
            f'Number of entries with missing counter names: {info["n_missing_names"]}',
            f'Number of counters removed due to low number of entries: {info["n_sparse_counters"]}',
            f'Number of holes in data: {info["n_holes"]}',
            f'Number of ounters with missing information: {info["n_counter_holes"]}',
            f'Number of holes imputed via regression: {info["n_reg"]}',
            f'Number of holes imputed via scaled medians: {info["n_sm"]}',
            f'Number of holes imputed via STL: {info["n_stl"]}',
        ]

        text = '\n'.join(lines)

        if print_output:
            print(text)

        if save:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
