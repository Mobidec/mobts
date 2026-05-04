from dataclasses import dataclass, field
from typing import Optional
from .config_common import ColumnsConfig, SparsityConfig


@dataclass
class PreprocessConfig:
    """
    Parameters for low-count/zero-run cleaning and operational window trimming.
    if avail_min_valid days out of avail_window is not present, the whole window will be set as non-operational.
    """

    low_rel_daily: float = 0.01  # threshold as fraction of station median
    low_abs_daily: float = 5  # absolute floor threshold to be considered low count noise
    low_run_min_daily: int = 2  # consecutive low count days to be set to NaN

    zero_rate_max: float = 0.05  # threshold to consider 0s normal

    night_hours = [1, 2, 3, 4, 5, 6]

    zero_run_min: int = 6
    island_max_len: int = 6
    surround_min_len: int = 12


@dataclass
class STLConfig:
    """
    Parameters for STL decomposition outlier scoring.
    """

    period: int = 28  # seasonal period in days, set to 4 weeks
    robust: bool = False  # set to False to avoid heavy computation


@dataclass
class OutlierConfig:
    """
    Parameters for thresholding STL outlier scores.
    """

    threshold_daily: float = 20  # threshold to be tuned via plotting
    threshold_hourly: float = 45  # threshold to be tuned via plotting


@dataclass
class PlotConfig:
    """
    Parameters for plotting the detected outliers.
    """

    ncols: int = 3
    figsize_width: float = 15
    min_fig_height: float = 10
    height_per_row: float = 3
    linewidth_d: float = 0.5
    linewidth_h: float = 0.3
    marker_size: float = 10
    x_label_rotation: int = 30
    max_stations: Optional[int] = None


@dataclass
class PipelineConfig:
    cols: ColumnsConfig = field(default_factory=ColumnsConfig)
    sparse: SparsityConfig = field(default_factory=SparsityConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    stl: STLConfig = field(default_factory=STLConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
