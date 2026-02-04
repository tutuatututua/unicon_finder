"""Typed configuration for feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for feature creation.

    Notes
    -----
    Defaults are chosen to match the project’s existing behavior.
    """

    forward_days: int = 252
    horizons: list[int] = field(default_factory=lambda: [21, 63, 252])
    ret_stats_window: int = 63
    min_history_rows: int = 252
    anchor_ticker: str = "AAPL"

    exclude_features: list[str] = field(default_factory=list)

    use_market_beta: bool = True
    use_volume_pctile: bool = True
    use_zscore_clip: bool = True

    sector_map_path: Optional[str] = "data/sector_map.csv"
    add_sector_relative: bool = True

    use_ret_kurtosis: bool = True
    ret_kurt_window: int = 63

    # Cross-sectional z-score options
    cs_zscore: Optional[str] = None  # None, 'global', 'sector', 'industry'
    cs_zscore_features: list[str] = field(default_factory=list)
