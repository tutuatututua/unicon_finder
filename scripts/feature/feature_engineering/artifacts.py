"""Artifact generation for feature engineering.

Writes:
- data/processed/raw_training.parquet
- data/processed/latest_data.parquet
- data/processed/extract_training.parquet
- data/processed/dropped_all_nan_rows.parquet (when applicable)
- models/features.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.config.logging import get_logger

from .feature_config import FeatureConfig
from .core import compute_features_for_ticker, downcast_float64_inplace, prune_to_expected_columns
from .cross_section import apply_cs_zscores
from .dataset import assemble_dataset

logger = get_logger(__name__)


def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s: %s (rows=%d)", label, path, len(df))


def _filter_latest_by_anchor(df: pd.DataFrame, anchor_ticker: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    last_per = df.sort_values(["ticker", "date"]).groupby("ticker").tail(1).reset_index(drop=True)
    anchor_dates = last_per.loc[last_per["ticker"] == anchor_ticker, "date"]
    if anchor_dates.empty:
        anchor_date = last_per["date"].max()
        logger.warning(
            "Anchor ticker %s not found in snapshot frame; using global max date %s",
            anchor_ticker,
            anchor_date,
        )
    else:
        anchor_date = anchor_dates.max()

    before = len(last_per)
    filtered = last_per[last_per["date"] == anchor_date].copy()
    dropped = before - len(filtered)
    logger.info(
        "Snapshot anchor=%s date=%s kept %d rows (dropped %d)",
        anchor_ticker,
        anchor_date,
        len(filtered),
        dropped,
    )
    return filtered


def build_features(
    *,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
) -> pd.DataFrame:
    """Build the full feature panel and snapshot artifacts.

    Keeps output locations and schemas consistent with the existing pipeline.
    """

    cfg = FeatureConfig()

    raw_features_path = processed_dir / "raw_training.parquet"
    scoring_path = processed_dir / "latest_data.parquet"
    training_feature_path = processed_dir / "extract_training.parquet"
    dropped_path = processed_dir / "dropped_all_nan_rows.parquet"
    meta_path = models_dir / "features.json"

    processed_dir.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame | None
    if raw_features_path.exists():
        try:
            existing = pd.read_parquet(raw_features_path)
            existing["date"] = pd.to_datetime(existing["date"], utc=True).dt.tz_localize(None).dt.normalize()
        except Exception as exc:
            logger.warning(
                "Failed reading existing features parquet (%s). Will rebuild from scratch. Err=%s",
                raw_features_path,
                exc,
            )
            existing = None
    else:
        existing = None

    if existing is None or existing.empty:
        full = assemble_dataset(cfg, raw_dir=raw_dir, processed_dir=processed_dir)
    else:
        last_by_ticker = existing.groupby("ticker")["date"].max()
        margin_days = int(cfg.forward_days or 252)
        paths = sorted(raw_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError("No raw parquet files found in data/raw. Run data download first.")

        updated_frames: list[pd.DataFrame] = []
        for p in paths:
            tkr = p.stem
            raw_dates = pd.read_parquet(p, columns=["date"])
            raw_dates["date"] = pd.to_datetime(raw_dates["date"], utc=True).dt.tz_localize(None).dt.normalize()
            raw_max = raw_dates["date"].max() if not raw_dates.empty else None

            last_done = last_by_ticker.get(tkr, pd.NaT)
            if pd.isna(last_done):
                recompute_start = None
            else:
                if raw_max is not None and raw_max <= last_done:
                    continue
                recompute_start = last_done - pd.Timedelta(days=margin_days)

            fdf = compute_features_for_ticker(p, cfg, min_date=recompute_start, warmup_days=margin_days)
            if fdf is not None and not fdf.empty:
                fdf["date"] = pd.to_datetime(fdf["date"], utc=True).dt.tz_localize(None).dt.normalize()
                updated_frames.append(fdf)

        if not updated_frames:
            logger.info("No tickers required feature updates; keeping existing features unchanged.")
            full = existing
        else:
            new_part = pd.concat(updated_frames, axis=0, ignore_index=True)
            all_cols = sorted(set(existing.columns).union(set(new_part.columns)))
            existing_u = existing.reindex(columns=all_cols)
            new_part_u = new_part.reindex(columns=all_cols)

            combined = pd.concat([existing_u, new_part_u], axis=0, ignore_index=True)
            combined.sort_values(["ticker", "date"], inplace=True)
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")

            downcast_float64_inplace(combined)
            full = combined.reset_index(drop=True)

            # Re-merge sector/industry metadata to ensure newly computed rows have it
            try:
                pth = Path(cfg.sector_map_path or "")
                if cfg.sector_map_path and pth.exists():
                    sm = pd.read_csv(pth)[["ticker", "sector", "industry"]]
                    sm["ticker"] = sm["ticker"].astype(str).str.upper()
                    drop_si = [c for c in ("sector", "industry") if c in full.columns]
                    if drop_si:
                        full = full.drop(columns=drop_si)
                    full = full.merge(sm, on="ticker", how="left")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed re-merging sector map during incremental build: %s", exc)

            full = apply_cs_zscores(full, cfg)
            full = prune_to_expected_columns(full, cfg)

    _save_parquet(full, raw_features_path, "training + unlabeled dataset")

    target_col = f"target_fwd_{cfg.forward_days}d"

    id_cols = {"ticker", "date", target_col}
    feature_cols = [c for c in full.columns if c not in id_cols]
    mask_all_na = full[feature_cols].isna().all(axis=1)
    if mask_all_na.any():
        dropped_rows = full.loc[mask_all_na, ["ticker", "date"]].copy()
        dropped_rows["reason"] = "all_nan_features"
        try:
            dropped_rows.to_parquet(dropped_path, index=False)
            logger.info(
                "Removed %d rows with all NaN features; details saved to %s",
                int(mask_all_na.sum()),
                dropped_path,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist dropped rows detail: %s", exc)
        full = full.loc[~mask_all_na].reset_index(drop=True)

    anchor = cfg.anchor_ticker

    unlabeled = full[full[target_col].isna()]
    latest_unlabeled = _filter_latest_by_anchor(unlabeled, anchor)
    if latest_unlabeled.empty:
        logger.warning("No unlabeled rows available for scoring snapshot.")
    else:
        _save_parquet(latest_unlabeled, scoring_path, "scoring snapshot (latest unlabeled)")

    labeled = full[full[target_col].notna()]
    last_labeled = _filter_latest_by_anchor(labeled, anchor)
    if last_labeled.empty:
        logger.warning("No labeled rows available for last labeled snapshot.")
    else:
        _save_parquet(last_labeled, training_feature_path, "last labeled snapshot")

    # Persist metadata
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"features": feature_cols, "target": target_col}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved feature metadata: %s", meta_path)

    return full
