"""Panel assembly from per-ticker feature frames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.config.logging import get_logger

from .feature_config import FeatureConfig
from .core import compute_features_for_ticker, downcast_float64_inplace, prune_to_expected_columns
from .cross_section import apply_cs_zscores

logger = get_logger(__name__)


def _merge_sector(df: pd.DataFrame, sector_csv: Optional[str]) -> pd.DataFrame:
    if not sector_csv:
        return df
    try:
        p = Path(sector_csv)
        if not p.exists():
            logger.info("Sector map not found at %s; skipping merge", p)
            return df
        sm = pd.read_csv(p)
        if "ticker" not in sm.columns:
            logger.warning("Sector map missing ticker column; skipping")
            return df
        keep_cols = [c for c in ("ticker", "sector", "industry") if c in sm.columns]
        if "sector" not in keep_cols and "industry" not in keep_cols:
            logger.warning("Sector map missing 'sector'/'industry' columns; skipping")
            return df
        sn = sm[keep_cols].copy()
        sn["ticker"] = sn["ticker"].astype(str).str.upper()
        return df.merge(sn, on="ticker", how="left")
    except Exception as exc:
        logger.warning("Failed merging sector map: %s", exc)
        return df


def assemble_dataset(
    cfg: FeatureConfig | None = None,
    *,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatenate per-ticker frames; drop rows where every feature is NaN.

    If processed_dir is provided, persists details of dropped rows to
    `processed_dir / 'dropped_all_nan_rows.parquet'`.
    """
    cfg = cfg or FeatureConfig()
    paths = sorted(raw_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No raw parquet files found in data/raw. Run data download first.")

    frames: list[pd.DataFrame] = []
    for p in paths:
        fdf = compute_features_for_ticker(p, cfg)
        if fdf is not None and not fdf.empty:
            frames.append(fdf)

    if not frames:
        raise RuntimeError("No ticker produced features (insufficient history?).")

    full = pd.concat(frames, axis=0, ignore_index=True)

    full = _merge_sector(full, cfg.sector_map_path)
    full = apply_cs_zscores(full, cfg)
    full = prune_to_expected_columns(full, cfg)
    downcast_float64_inplace(full)

    id_cols = {"ticker", "date", f"target_fwd_{cfg.forward_days}d"}
    feature_cols = [c for c in full.columns if c not in id_cols]
    mask_all_na = full[feature_cols].isna().all(axis=1)
    removed = int(mask_all_na.sum())
    if removed:
        if processed_dir is not None:
            try:
                processed_dir.mkdir(parents=True, exist_ok=True)
                dropped_rows = full.loc[mask_all_na, ["ticker", "date"]].copy()
                dropped_rows["reason"] = "all_nan_features"
                dropped_path = processed_dir / "dropped_all_nan_rows.parquet"
                dropped_rows.to_parquet(dropped_path, index=False)
                logger.info(
                    "Removed %d rows with all NaN features; details saved to %s",
                    removed,
                    dropped_path,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to persist dropped rows detail: %s", exc)
        full = full.loc[~mask_all_na].reset_index(drop=True)

    # Final normalize date
    full["date"] = pd.to_datetime(full["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

    # Replace infs (safety)
    num_cols = full.select_dtypes(include=[np.number]).columns
    full[num_cols] = full[num_cols].replace([np.inf, -np.inf], np.nan)

    return full
