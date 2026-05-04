from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnsConfig:
    """
    Canonical column names used in the pipeline after standardization.
    """

    counter: str = 'name'
    timestamp: str = 'timestamp'
    count: str = 'count'

    weekday: str = 'weekday'
    week_num: str = 'week_num'
    how: str = 'how'
    hour: str = 'hour'
    date: str = 'date'


@dataclass
class SparsityConfig:
    """
    For removing counters with not enough valid counts
    """

    drop_sparse_counters: bool = True
    sparse_threshold: float = 0.5
