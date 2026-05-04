from dataclasses import dataclass
from typing import Optional


@dataclass
class STLConfig:
    """
    Config used in STL imputation
    """

    # STL seasonal period (for daily)
    stl_season_daily = 7
    stl_season_hourly = 168

    # clipping
    clip_lower = 0

    # rollinng median
    rolling_median_window: int = 2
    rolling_median_min_valid: int = 1

    # STL robust
    stl_robust = False


@dataclass
class DonorsConfig:
    """
    Configs for Donor-based imputation
    """

    top_k_donor = 25
    max_donor_rate = 0.5

    # scaled median
    sm_min_overlap_day = 60
    sm_min_overlap_hour = sm_min_overlap_day * 24
    sm_min_neighbors = 20

    # regression
    min_mutual_days = 60
    min_mutual_hours = min_mutual_days * 24
    min_pred_days = 30
    min_pred_hours = min_pred_days * 24
    min_pred_coverage = 0.9


@dataclass
class OutputConfig:
    """
    Configs for output columns and final selection
    """

    # calculated column names
    col_intp = 'count_intp'
    col_stl_imputed = 'count_stl_imputed'
    col_sm_imputed: str = 'count_sm_imputed'
    col_reg_imputed: str = 'count_reg_imputed'
    col_final: str = 'count_imputed'
    col_method_used: str = 'imputation_method'

    stl_method: str = 'STL'
    sm_method: str = 'M7'
    reg_method: str = 'M8'
