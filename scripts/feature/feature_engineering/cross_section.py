"""Cross-sectional feature transforms."""

from __future__ import annotations

import pandas as pd

from scripts.config.logging import get_logger

from .feature_config import FeatureConfig

logger = get_logger(__name__)


def apply_cs_zscores(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Add cross-sectional z-scores per date optionally grouped by sector/industry."""
    mode = cfg.cs_zscore
    if not mode:
        return df

    features = list(cfg.cs_zscore_features or [])
    if not features:
        return df

    features = [c for c in features if c in df.columns]
    if not features:
        return df

    if mode == "global":
        gcols = ["date"]
    elif mode == "sector":
        if "sector" not in df.columns:
            logger.warning("cs_zscore='sector' requested but 'sector' column missing; skipping")
            return df
        gcols = ["date", "sector"]
    elif mode == "industry":
        if "industry" not in df.columns:
            logger.warning("cs_zscore='industry' requested but 'industry' column missing; skipping")
            return df
        gcols = ["date", "industry"]
    else:
        logger.warning("Unknown cs_zscore mode %s; skipping", mode)
        return df

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    grp = work.groupby(gcols, dropna=False)

    for col in features:
        mu = grp[col].transform("mean")
        sd = grp[col].transform("std")
        cs = (work[col] - mu) / sd.replace(0, pd.NA)
        work[f"{col}_csz_{mode}"] = cs

    return work
