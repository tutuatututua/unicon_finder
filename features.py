"""Feature engineering module.

Primary public API:
  * FeatureConfig          - configuration dataclass
  * assemble_dataset(cfg)  - build full panel of engineered features
  * build_features(...)    - orchestrate dataset assembly + artifact snapshots

The refactor focuses on: cleaner defaults handling, smaller helpers, and
clearer sequencing of operations while preserving backward compatibility
for external modules (paths, function names, column names).
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json

import numpy as np
import pandas as pd

from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Path configuration (override-able via config.yaml -> paths)
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_FEATURES_PATH = PROC_DIR / "raw_training.parquet"
SCORING_PATH = PROC_DIR / "latest_data.parquet"
META_PATH = Path("models/features.json")
TRAINING_FEATURE_PATH = PROC_DIR / "extract_training.parquet"

# ---------------------------------------------------------------------------
def _downcast_float64_inplace(df: pd.DataFrame) -> None:
    """Safely downcast float64 columns to float32 in-place without select_dtypes.

    Rationale: DataFrame.select_dtypes can trigger a large block consolidation,
    which may allocate huge intermediate arrays on very wide frames. Iterating
    over columns avoids that memory spike.
    """
    try:
        for col, dtype in df.dtypes.items():
            # Only downcast exact float64 dtypes
            if dtype == np.float64:
                # pd.to_numeric with downcast='float' chooses float32 when safe
                df[col] = pd.to_numeric(df[col], downcast='float')
    except Exception as e:  # pragma: no cover
        logger.warning("Downcast failure (continuing without full downcast): %s", e)

    # No return; mutation in place

    
    
@dataclass(slots=True)
class FeatureConfig:
    """
    Configuration for feature creation (simplified and stationarized).

    - Uses compact horizons to reduce multicollinearity (short/medium/long)
    - Focuses on stationary log/z-score-based features
    """

    forward_days: int = 252          # 1-year forward return horizon
    horizons: List[int] = field(default_factory=lambda: [21, 63, 252])  # short, medium, long
    ret_stats_window: int = 63       # for rolling skewness and pos return ratio
    min_history_rows: int = 252      # minimum rows for valid feature generation
    anchor_ticker: str = "AAPL"      # used for alignment in snapshots

    # Excluded features (can be auto-filled after correlation checks)
    exclude_features: List[str] = field(default_factory=lambda: [])

    # Optional toggles (if you want to control inclusion later)
    use_market_beta: bool = True     # include rolling beta to market if available
    use_volume_pctile: bool = True   # include rolling volume percentile
    use_zscore_clip: bool = True     # apply clipping on z-scores to avoid extreme values
    # Optional sector usage
    sector_map_path: Optional[str] = "data/sector_map.csv"  # if exists, used for sector-relative features
    add_sector_relative: bool = True  # compute sector-demeaned variants of selected signals
    # Optional: include rolling kurtosis of daily returns
    use_ret_kurtosis: bool = True
    ret_kurt_window: int = 63


    # Cross-sectional z-score options
    cs_zscore: Optional[str] = None  # one of None, 'global', 'sector', 'industry'
    cs_zscore_features: List[str] = field(default_factory=list)
                  # reference ticker used to align latest snapshot dates
  


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _compute_rsi(price: pd.Series, window: int) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing."""
    if window <= 0:
        return pd.Series(np.nan, index=price.index)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _rank_pct_last(x: np.ndarray) -> float:
    """Percentile rank (0..1] of the last non-NaN element within the window.

    Implements pandas' rank(method='average', pct=True) for the last value in the
    rolling window, but computed with NumPy for speed. Returns NaN if the last
    value is NaN or if the window has no valid values.
    """
    if x.size == 0 or np.isnan(x[-1]):
        return np.nan
    # Drop NaNs from the window for a stable rank count
    y = x[~np.isnan(x)]
    n = y.size
    if n == 0:
        return np.nan
    last = x[-1]
    # Sort valid values and use searchsorted bounds for average rank
    s = np.sort(y)
    left = np.searchsorted(s, last, side='left')   # count of values < last
    right = np.searchsorted(s, last, side='right') # count of values <= last
    # Average rank in 1..n scale, then convert to percentile (0..1]
    avg_rank = 0.5 * (left + right + 1)
    return avg_rank / n


def _apply_feature_filters(df: pd.DataFrame, cfg: FeatureConfig) -> Optional[pd.DataFrame]:
    """Apply include/exclude feature column filtering and enforce invariants.

    Returns the possibly reduced DataFrame or None if no feature columns remain.
    """
    target_col = f'target_fwd_{cfg.forward_days}d'
    id_cols = {'ticker', 'date', target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    original = set(feature_cols)
    if cfg.exclude_features:
        drop = [c for c in feature_cols if c in set(cfg.exclude_features)]
        if drop:
            df = df.drop(columns=drop)
            feature_cols = [c for c in feature_cols if c not in drop]
    if not feature_cols:
        return None
    return df


def _prepare_price_volume(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize dates and extract price, volume, daily returns."""
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    price = pd.to_numeric(df["close"], errors='coerce')
    # Guard against non-positive prices (can break log/ratio logic) by setting to NaN
    price = price.where(price > 0)
    volume = pd.to_numeric(df.get('volume', pd.Series(np.nan, index=df.index)), errors='coerce')
    daily_ret = price.pct_change(fill_method=None)
    # Replace impossible returns (>|500%| daily) with NaN to stabilize rolling moments
    daily_ret = daily_ret.where(daily_ret.abs() <= 5.0)
    # Final sanitation: drop infs
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan)
    return price, volume, daily_ret


# ---------------------------------------------------------------------------
# Core per‑ticker feature computation
# ---------------------------------------------------------------------------
def compute_features_for_ticker(path: Path, cfg: FeatureConfig, min_date: Optional[pd.Timestamp] = None, warmup_days: int = 252) -> Optional[pd.DataFrame]:
    """Compute per-ticker features in clear, well-scoped steps.

    Contract
    - Inputs: path to a single-ticker OHLCV parquet with at least columns ['date','close'].
    - Output: DataFrame with id cols ['ticker','date', target] + engineered features.
    - Behavior: preserves existing feature names and formulas used in the pipeline.
    """
    # 1) Load and basic validation
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("Failed reading %s: %s", path.name, e)
        return None

    if df.empty or ('date' not in df.columns) or ('close' not in df.columns):
        logger.info("Invalid or empty file for %s; skipping", path.stem)
        return None
    # Optional range filter to support incremental recompute with warmup
    if min_date is not None:
        try:
            dser = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
            cutoff = (pd.to_datetime(min_date).tz_localize(None) - pd.Timedelta(days=max(1, int(warmup_days))))
            df = df.loc[dser >= cutoff].reset_index(drop=True)
        except Exception:
            # If anything goes wrong, fall back to full frame
            pass
    if len(df) < cfg.min_history_rows:
        logger.info("Insufficient data for %s (%d < %d)", path.stem, len(df), cfg.min_history_rows)
        return None

    logger.info("Computing features for %s with %d rows", path.stem, len(df))

    # 2) Core series preparation
    price, volume, daily_ret = _prepare_price_volume(df, cfg)
    out = pd.DataFrame({"ticker": path.stem, "date": df["date"]})

    # Temporary references for cleanup
    out["_price"] = price
    out["_daily_ret"] = daily_ret

    # Common numeric transforms
    logp = pd.Series(np.log(price.replace(0, np.nan).to_numpy(dtype=float)), index=price.index)

    # Handy helpers (scoped to this function)
    def rolling_robust_z(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
        r = s.rolling(window=w, min_periods=minp or w)
        med = r.median()
        def _mad_fn(x: np.ndarray) -> float:
            if x.size == 0:
                return np.nan
            m = np.nanmedian(x)
            return float(np.nanmedian(np.abs(x - m)))
        mad = r.apply(_mad_fn, raw=True)
        scale = mad.where(~mad.isna() & (mad > 0), r.std(ddof=0))
        return _safe_div(s - med, scale)

    def lin_slope(x: np.ndarray) -> float:
        n = len(x)
        if np.isfinite(x).sum() <= 10:
            return np.nan
        idx = np.arange(n)
        try:
            slope, _ = np.polyfit(idx, x, 1)
            return slope
        except Exception:
            return np.nan

    feat: dict[str, pd.Series] = {}

    # 3) Momentum: z of log-return over windows
    for w in [21, 63, 126, 252]:
        mom_w = logp.diff(w)
        r = mom_w.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"ret_z_{w}d"] = _safe_div(mom_w - mu, sd)

    # 4) Price vs moving average (raw for 126/252, z for all)
    feat["price_ma_ratio_252d"] = price / price.rolling(252).mean() - 1.0
    feat["price_ma_ratio_126d"] = price / price.rolling(126).mean() - 1.0
    for w in [21, 63, 126, 252]:
        ma = price.rolling(w).mean()
        ratio = _safe_div(price, ma) - 1.0
        r = ratio.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"price_ma_ratio_z_{w}d"] = _safe_div(ratio - mu, sd)

    # 5) Volume percentile ranks
    for w in [63, 126, 252]:
        feat[f"volume_pctile_{w}d"] = volume.rolling(window=w, min_periods=w).apply(_rank_pct_last, raw=True)

    # 6) Drawdown/up from rolling extremes (z)
    for w in [126, 252]:
        roll_max = price.rolling(w).max()
        roll_alltime = price.cummax()
        dd = price / roll_max - 1.0
        r = dd.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"dd_cur_z_{w}d"] = _safe_div(dd - mu, sd)
        feat[f"dd_alltime_z_{w}d"] = _safe_div((price / roll_alltime - 1.0) - mu, sd)

        roll_min = price.rolling(w).min()
        du = price / roll_min - 1.0
        r2 = du.rolling(window=w, min_periods=w)
        mu2, sd2 = r2.mean(), r2.std(ddof=0)
        feat[f"du_cur_z_{w}d"] = _safe_div(du - mu2, sd2)

    # 7) All-time-high derived features
    ath = price.cummax()
    off_ath = price / ath - 1.0
    feat["off_ath"] = off_ath
    # days since last ATH
    is_new_ath = ath.diff().fillna(0.0).gt(0)
    grp = is_new_ath.cumsum()
    feat["days_since_ath"] = grp.groupby(grp).cumcount().astype("float")
    # robust z over 252d
    feat["off_ath_z_252d"] = rolling_robust_z(off_ath, 252)

    # 8) ATR family, range position, breakout, and HV contraction (requires high/low)
    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        prev_close = price.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        for w in [63, 126, 252]:
            atr = _safe_div(tr.rolling(w, min_periods=w).mean(), price)
            feat[f"atr_z_{w}d"] = rolling_robust_z(atr, w)
            feat[f"atr_trend_{w}d"] = atr.rolling(w).apply(lin_slope, raw=True)

            H = high.rolling(w, min_periods=w).max()
            L = low.rolling(w, min_periods=w).min()
            rng = (H - L).where((H - L) > 0)
            pos = (price - L) / rng
            signed_pos = pos - 0.5
            feat[f"pos_in_range_z_{w}d"] = rolling_robust_z(signed_pos, w)

            # breakout strength vs 52w range
            H52 = high.rolling(252, min_periods=252).max()
            L52 = low.rolling(252, min_periods=252).min()
            rng52 = (H52 - L52).replace(0, np.nan)
            feat["breakout_strength_52w"] = _safe_div(price - H52, rng52)

        for wv in [126, 252]:
            hv = daily_ret.rolling(wv).std(ddof=0)
            feat[f"hv_contraction_z_{wv}d"] = rolling_robust_z(hv, wv)

    # 9) Upside vs downside volatility ratio (log-ratio z)
    for w in (63, 126, 252):
        pos_vol = daily_ret.where(daily_ret > 0).rolling(window=w, min_periods=2).std(ddof=0)
        neg_vol = daily_ret.where(daily_ret < 0).rolling(window=w, min_periods=2).std(ddof=0)
        ratio = _safe_div(pos_vol, (neg_vol + 1e-8)).clip(0, 50)
        log_ratio = pd.Series(np.log(ratio.replace(0, np.nan).to_numpy(dtype=float)), index=ratio.index)
        _mp = max(20, w // 5)
        r = log_ratio.rolling(window=w, min_periods=_mp)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"up_down_vol_ratio_z_{w}d"] = _safe_div(log_ratio - mu, sd)

    # 10) Dollar volume z
    dollar_vol = price * volume
    for w in (21, 63, 126, 252):
        r = dollar_vol.rolling(w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"dollar_vol_z_{w}d"] = _safe_div(dollar_vol - mu, sd)

    # 11) RSI percentile features
    stats_w = int(getattr(cfg, "ret_stats_window", 63) or 63)
    for rw in (126, 252):
        rsi = _compute_rsi(price, rw)
        feat[f"rsi_pctile_{rw}d"] = rsi.rolling(window=stats_w, min_periods=stats_w).apply(_rank_pct_last, raw=True)

    # 12) Return moments
    feat["ret_skew_63d"] = daily_ret.rolling(window=63, min_periods=63).skew()
    if getattr(cfg, "use_ret_kurtosis", False):
        try:
            kw = int(getattr(cfg, "ret_kurt_window", 63) or 63)
            feat[f"ret_kurt_{kw}d"] = daily_ret.rolling(window=kw, min_periods=kw).kurt()
        except Exception:
            feat[f"ret_kurt_{int(getattr(cfg, 'ret_kurt_window', 63) or 63)}d"] = pd.Series(np.nan, index=daily_ret.index)

    # 13) Trend strength via correlation of log price with time index
    try:
        t_series = pd.Series(np.arange(len(logp), dtype=float), index=logp.index)
        feat["trend_strength_126d"] = logp.rolling(window=126, min_periods=126).corr(t_series)
    except Exception:
        feat["trend_strength_126d"] = pd.Series(np.nan, index=logp.index)

    # 14) Sharpe t-stat on 252d
    try:
        roll = daily_ret.rolling(window=252, min_periods=252)
        mu, sd, n = roll.mean(), roll.std(ddof=0), roll.count()
        feat["sharpe_tstat_252d"] = _safe_div(mu * np.sqrt(n), sd)
    except Exception:
        feat["sharpe_tstat_252d"] = pd.Series(np.nan, index=daily_ret.index)

    # 15) Stability: negative robust z of vol-of-vol over 252d
    try:
        hv21 = daily_ret.rolling(21, min_periods=21).std(ddof=0)
        vov252 = hv21.rolling(252, min_periods=252).std(ddof=0)
        # Robust center/scale over a shorter window to avoid all-NaNs early on
        _rob = max(5, 252 // 4)
        def _mad_arr(x: np.ndarray) -> float:
            if x.size == 0:
                return np.nan
            m = np.nanmedian(x)
            return float(np.nanmedian(np.abs(x - m)))
        v_med = vov252.rolling(_rob, min_periods=1).median()
        v_mad = vov252.rolling(_rob, min_periods=1).apply(_mad_arr, raw=True)
        v_scale = v_mad.where(~v_mad.isna() & (v_mad > 0), vov252.rolling(_rob, min_periods=1).std(ddof=0))
        feat["stability_z_252d"] = -_safe_div(vov252 - v_med, v_scale)
    except Exception:
        feat["stability_z_252d"] = pd.Series(np.nan, index=daily_ret.index)

    # 16) Target construction
    fwd = cfg.forward_days
    out[f"target_fwd_{fwd}d"] = _safe_div(price.shift(-fwd), price) - 1

    # 17) Merge, clean, and filter
    out = pd.concat([out, pd.DataFrame(feat)], axis=1)
    out.drop(columns=["_price", "_daily_ret"], errors="ignore", inplace=True)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).clip(-20, 20)

    out = _apply_feature_filters(out, cfg)
    if out is None:
        logger.warning("No features remain after filtering for %s", path.stem)
        return None
    # If incremental range provided, keep only rows on/after min_date
    if min_date is not None and not out.empty:
        try:
            out['date'] = pd.to_datetime(out['date'], utc=True).dt.tz_localize(None).dt.normalize()
            out = out.loc[out['date'] >= pd.to_datetime(min_date).tz_localize(None)].reset_index(drop=True)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Assembly & orchestration
# ---------------------------------------------------------------------------
def assemble_dataset(cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Concatenate per‑ticker frames; drop rows where every feature is NaN.

    Parameters
    ----------
    cfg : FeatureConfig | None
        Optional configuration. If omitted, a default FeatureConfig() is used.
    """
    cfg = cfg or FeatureConfig()
    paths = sorted(RAW_DIR.glob('*.parquet'))
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

    # Optionally merge sector/industry metadata if available
    def _merge_sector(df: pd.DataFrame, sector_csv: Optional[str]) -> pd.DataFrame:
        if not sector_csv:
            return df
        try:
            p = Path(sector_csv)
            if not p.exists():
                logger.info("Sector map not found at %s; skipping merge", p)
                return df
            sm = pd.read_csv(p)
            sn = sm[["ticker", "sector", "industry"]]
            if "ticker" not in sm.columns:
                logger.warning("Sector map missing ticker column; skipping")
                return df
            sn["ticker"] = sn["ticker"].astype(str).str.upper()
            df = df.merge(sn, on="ticker", how="left")
            return df
        except Exception as e:
            logger.warning("Failed merging sector map: %s", e)
            return df

    full = _merge_sector(full, getattr(cfg, "sector_map_path", None))

    # Apply cross-sectional z-scores if requested
    full = _apply_cs_zscores(full, cfg)

    
    # Downcast float columns to save disk / memory (avoid select_dtypes to prevent large consolidation)
    _downcast_float64_inplace(full)

    # Use dynamic target column based on configuration
    id_cols = {'ticker', 'date', f'target_fwd_{cfg.forward_days}d'}
    feature_cols = [c for c in full.columns if c not in id_cols]
    mask_all_na = full[feature_cols].isna().all(axis=1)
    removed = int(mask_all_na.sum())
    if removed:
        dropped_rows = full.loc[mask_all_na, ['ticker', 'date']].copy()
        # Optionally include reason column for clarity
        dropped_rows['reason'] = 'all_nan_features'
        dropped_path = PROC_DIR / 'dropped_all_nan_rows.parquet'
        try:
            dropped_rows.to_parquet(dropped_path, index=False)
            logger.info(
                "Removed %d rows with all NaN features; details saved to %s (showing first 5: %s)",
                removed,
                dropped_path,
                dropped_rows.head().to_dict(orient='records')
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist dropped rows detail: %s", exc)
        full = full.loc[~mask_all_na].reset_index(drop=True)

    return full

def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s: %s (rows=%d)", label, path, len(df))


def build_features() -> pd.DataFrame:
    """Build the full feature panel and snapshot artifacts.

    NOTE: Former helper `apply_config_overrides` has been inlined here for
    readability and to keep all orchestration logic in one place.

    Artifacts created:
      * raw_training.parquet (full panel; name retained for compatibility)
      * latest_data.parquet  (latest unlabeled row per ticker)
      * extract_training.parquet (most recent labeled row per ticker)
      * dropped_all_nan_rows.parquet (all-NaN features)
      * models/features.json (feature list + target name)
    """

    cfg = FeatureConfig()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # If a previous features parquet exists, perform an incremental update.
    # Otherwise, build from scratch.
    if RAW_FEATURES_PATH.exists():
        try:
            existing = pd.read_parquet(RAW_FEATURES_PATH)
            # Normalize datatypes
            existing['date'] = pd.to_datetime(existing['date'], utc=True).dt.tz_localize(None).dt.normalize()
        except Exception as e:
            raise FileNotFoundError(f"Failed reading existing features parquet; rebuilding full. Err={e}")
            
    else:
        existing = None

    if existing is None or existing.empty:
        full = assemble_dataset(cfg)
    else:
        target_col = f'target_fwd_{cfg.forward_days}d'
        # Map last processed date per ticker
        last_by_ticker = existing.groupby('ticker')['date'].max()
        # Determine max lookback margin to recompute for stability and to refresh labels
        margin_days = getattr(cfg, 'forward_days', 252)
        paths = sorted(RAW_DIR.glob('*.parquet'))
        if not paths:
            raise FileNotFoundError("No raw parquet files found in data/raw. Run data download first.")

        updated_frames: list[pd.DataFrame] = []
        updated_tickers: list[str] = []
        for p in paths:
            tkr = p.stem
            # Quick peek at raw max date to decide if update is needed
            raw_dates = pd.read_parquet(p, columns=['date'])
            raw_dates['date'] = pd.to_datetime(raw_dates['date'], utc=True).dt.tz_localize(None).dt.normalize()
            raw_max = raw_dates['date'].max() if not raw_dates.empty else None

            last_done = last_by_ticker.get(tkr, pd.NaT)
            if pd.isna(last_done):
                recompute_start = None  # new ticker -> full compute for ticker
            else:
                # Skip if no new raw rows
                if raw_max is not None and raw_max <= last_done:
                    continue
                recompute_start = last_done - pd.Timedelta(days=margin_days)

            fdf = compute_features_for_ticker(p, cfg, min_date=recompute_start, warmup_days=margin_days)
            if fdf is not None and not fdf.empty:
                # Ensure normalized date
                fdf['date'] = pd.to_datetime(fdf['date'], utc=True).dt.tz_localize(None).dt.normalize()
                updated_frames.append(fdf)
                updated_tickers.append(tkr)

        if not updated_frames:
            logger.info("No tickers required feature updates; keeping existing features unchanged.")
            full = existing
        else:
            new_part = pd.concat(updated_frames, axis=0, ignore_index=True)

            # Union columns between existing and new frames
            all_cols = sorted(set(existing.columns).union(set(new_part.columns)))
            existing_u = existing.reindex(columns=all_cols)
            new_part_u = new_part.reindex(columns=all_cols)

            # Concatenate and keep the newest version for duplicate (ticker, date)
            combined = pd.concat([existing_u, new_part_u], axis=0, ignore_index=True)
            combined.sort_values(['ticker', 'date'], inplace=True)
            combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')

            # Downcast float columns to save disk / memory (avoid select_dtypes to prevent large consolidation)
            _downcast_float64_inplace(combined)

            # Re-merge sector/industry metadata to ensure newly computed rows have it
            # (newly computed per-ticker frames don't include sector columns)
            full = combined.reset_index(drop=True)
            try:
                p = Path(getattr(cfg, "sector_map_path", "data/sector_map.csv"))
                if getattr(cfg, "sector_map_path", None) and p.exists():
                    sm = pd.read_csv(p)[["ticker", "sector", "industry"]]
                    sm["ticker"] = sm["ticker"].astype(str).str.upper()
                    # Drop any pre-existing sector/industry columns to avoid _x/_y duplicates
                    drop_si = [c for c in ("sector", "industry") if c in full.columns]
                    if drop_si:
                        full = full.drop(columns=drop_si)
                    full = full.merge(sm, on="ticker", how="left")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed re-merging sector map during incremental build: %s", e)

            # Recompute cross-sectional z-scores globally for consistency
            full = _apply_cs_zscores(full, cfg)

    _save_parquet(full, RAW_FEATURES_PATH, "training + unlabeled dataset")
    
    # Snapshot helpers (filter by anchor_ticker latest date)
    def _filter_latest_by_anchor(df: pd.DataFrame, anchor_ticker: str) -> pd.DataFrame:
        """Return last row per ticker all aligned to the anchor ticker's latest date.

        If the anchor ticker is missing, it falls back to the global max date across tickers.
        """
        if df.empty:
            return df.copy()
        last_per = df.sort_values(['ticker', 'date']).groupby('ticker').tail(1).reset_index(drop=True)
        anchor_dates = last_per.loc[last_per['ticker'] == anchor_ticker, 'date']
        if anchor_dates.empty:
            anchor_date = last_per['date'].max()
            logger.warning("Anchor ticker %s not found in snapshot frame; using global max date %s", anchor_ticker, anchor_date)
        else:
            anchor_date = anchor_dates.max()
        before = len(last_per)
        filtered = last_per[last_per['date'] == anchor_date].copy()
        dropped = before - len(filtered)
        logger.info("Snapshot anchor=%s date=%s kept %d rows (dropped %d)", anchor_ticker, anchor_date, len(filtered), dropped)
        return filtered

    target_col = f'target_fwd_{cfg.forward_days}d'
    anchor = cfg.anchor_ticker

    # Unlabeled (scoring) snapshot
    unlabeled = full[full[target_col].isna()]
    latest_unlabeled = _filter_latest_by_anchor(unlabeled, anchor)
    if latest_unlabeled.empty:
        logger.warning("No unlabeled rows available for scoring snapshot.")
    else:
        _save_parquet(latest_unlabeled, SCORING_PATH, "scoring snapshot (latest unlabeled)")

    # Labeled (training extraction) snapshot
    labeled = full[full[target_col].notna()]
    last_labeled = _filter_latest_by_anchor(labeled, anchor)
    if last_labeled.empty:
        logger.warning("No labeled rows available for last labeled snapshot.")
    else:
        _save_parquet(last_labeled, TRAINING_FEATURE_PATH, "last labeled snapshot")

    feature_cols = [c for c in full.columns if c not in {'ticker', 'date', target_col}]
    # Log each feature name so user can inspect the final set
    for i, fname in enumerate(sorted(feature_cols), start=1):
        logger.info("Feature %03d: %s", i, fname)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps({'features': feature_cols, 'target': target_col}, indent=2),
        encoding='utf-8',
    )
    logger.info("Saved feature metadata: %s", META_PATH)
    return full


__all__ = ["FeatureConfig", "build_features", "assemble_dataset"]

# ---------------------------------------------------------------------------
# Helpers: cross-sectional z-scores
# ---------------------------------------------------------------------------
def _apply_cs_zscores(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Add cross-sectional z-scores per date optionally grouped by sector/industry.

    Creates columns: f"{base}_csz_{mode}" for base in cfg.cs_zscore_features.
    Modes: None (disabled), 'global', 'sector', 'industry'.
    """
    mode = getattr(cfg, 'cs_zscore', None)
    if not mode:
        return df
    features = list(getattr(cfg, 'cs_zscore_features', []) or [])
    if not features:
        return df

    # Ensure base columns exist
    features = [c for c in features if c in df.columns]
    if not features:
        return df

    if mode == 'global':
        gcols = ['date']
    elif mode == 'sector':
        if 'sector' not in df.columns:
            logger.warning("cs_zscore='sector' requested but 'sector' column missing; skipping CS z-scores")
            return df
        gcols = ['date', 'sector']
    elif mode == 'industry':
        if 'industry' not in df.columns:
            logger.warning("cs_zscore='industry' requested but 'industry' column missing; skipping CS z-scores")
            return df
        gcols = ['date', 'industry']
    else:
        logger.warning("Unknown cs_zscore mode %s; skipping CS z-scores", mode)
        return df

    work = df.copy()
    # Normalize date
    work['date'] = pd.to_datetime(work['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    grp = work.groupby(gcols, dropna=False)
    for col in features:
        mu = grp[col].transform('mean')
        sd = grp[col].transform('std')
        cs = (work[col] - mu) / sd.replace(0, np.nan)
        work[f"{col}_csz_{mode}"] = cs
    return work
