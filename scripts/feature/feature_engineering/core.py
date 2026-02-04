"""Core feature engineering primitives.

These are kept separate so they can be unit-tested and reused by the dataset
assembly + artifact writers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.config.logging import get_logger

from .feature_config import FeatureConfig

logger = get_logger(__name__)


def prune_to_expected_columns(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Drop any columns not in the current expected set (keeps IDs + sector metadata)."""
    target_col = f"target_fwd_{cfg.forward_days}d"
    keep: set[str] = {"ticker", "date", target_col}
    for c in ("sector", "industry"):
        if c in df.columns:
            keep.add(c)

    horizons = sorted({int(x) for x in (cfg.horizons or []) if int(x) > 0})
    names: list[str] = []

    # Momentum
    names.extend([f"ret_z_{w}d" for w in horizons])

    # Price vs MA
    names.extend([f"price_ma_ratio_z_{w}d" for w in horizons])

    # Volume percentile ranks (medium/long only)
    if cfg.use_volume_pctile:
        names.extend([f"volume_pctile_{w}d" for w in horizons if w >= 63])

    # Drawdown / run-up (long only)
    for w in (w for w in horizons if w >= 252):
        names.extend([f"dd_cur_z_{w}d", f"dd_alltime_z_{w}d", f"du_cur_z_{w}d"])

    # ATH
    names.append("off_ath_z_252d")

    # ATR / range features
    for w in (w for w in horizons if w >= 63):
        names.extend([f"atr_z_{w}d", f"pos_in_range_z_{w}d"])
    names.append("breakout_strength_52w")
    for wv in (w for w in horizons if w >= 252):
        names.append(f"hv_contraction_z_{wv}d")

    # Upside vs downside vol
    names.extend([f"up_down_vol_ratio_z_{w}d" for w in horizons if w >= 63])

    # Dollar volume
    names.extend([f"dollar_vol_z_{w}d" for w in horizons if w >= 63])

    # RSI percentile (long only)
    names.extend([f"rsi_pctile_{w}d" for w in horizons if w >= 252])

    # Return moments
    names.append("ret_skew_63d")
    if cfg.use_ret_kurtosis:
        kw = int(cfg.ret_kurt_window or 63)
        names.append(f"ret_kurt_{kw}d")

    # Sharpe t-stat / stability
    names.append("sharpe_tstat_252d")
    names.append("stability_z_252d")

    # Optional cross-sectional z-scores
    if cfg.cs_zscore and (cfg.cs_zscore_features or []):
        names.extend([f"{c}_csz_{cfg.cs_zscore}" for c in list(cfg.cs_zscore_features)])

    keep |= set(names)
    drop = [c for c in df.columns if c not in keep]
    if drop:
        logger.info("Pruning %d legacy/unexpected columns", len(drop))
        return df.drop(columns=drop)
    return df


def downcast_float64_inplace(df: pd.DataFrame) -> None:
    """Downcast float64 columns to float32 in-place (best-effort)."""
    try:
        for col, dtype in df.dtypes.items():
            if dtype == np.float64:
                df[col] = pd.to_numeric(df[col], downcast="float")
    except Exception as exc:  # pragma: no cover
        logger.warning("Downcast failure (continuing without full downcast): %s", exc)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def compute_rsi(price: pd.Series, window: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    if window <= 0:
        return pd.Series(np.nan, index=price.index)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = safe_div(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def rank_pct_last(x: np.ndarray) -> float:
    """Percentile rank (0..1] of the last non-NaN element within the window."""
    if x.size == 0 or np.isnan(x[-1]):
        return np.nan
    y = x[~np.isnan(x)]
    n = y.size
    if n == 0:
        return np.nan
    last = x[-1]
    s = np.sort(y)
    left = np.searchsorted(s, last, side="left")
    right = np.searchsorted(s, last, side="right")
    avg_rank = 0.5 * (left + right + 1)
    return avg_rank / n


def apply_feature_filters(df: pd.DataFrame, cfg: FeatureConfig) -> Optional[pd.DataFrame]:
    """Apply exclude list and ensure at least one feature remains."""
    target_col = f"target_fwd_{cfg.forward_days}d"
    id_cols = {"ticker", "date", target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    if cfg.exclude_features:
        drop = [c for c in feature_cols if c in set(cfg.exclude_features)]
        if drop:
            df = df.drop(columns=drop)
            feature_cols = [c for c in feature_cols if c not in drop]
    return df if feature_cols else None


def prepare_price_volume(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize dates and extract price, volume, daily returns."""
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    price = pd.to_numeric(df["close"], errors="coerce")
    price = price.where(price > 0)
    volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce")
    daily_ret = price.pct_change(fill_method=None)
    daily_ret = daily_ret.where(daily_ret.abs() <= 5.0)
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan)
    return price, volume, daily_ret


def compute_features_for_ticker(
    path: Path,
    cfg: FeatureConfig,
    min_date: Optional[pd.Timestamp] = None,
    warmup_days: int = 252,
) -> Optional[pd.DataFrame]:
    """Compute per-ticker features.

    This function preserves the project’s existing feature names/formulas.
    """
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed reading %s: %s", path.name, exc)
        return None

    if df.empty or ("date" not in df.columns) or ("close" not in df.columns):
        logger.info("Invalid or empty file for %s; skipping", path.stem)
        return None

    if min_date is not None:
        try:
            dser = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
            cutoff = pd.to_datetime(min_date).tz_localize(None) - pd.Timedelta(days=max(1, int(warmup_days)))
            df = df.loc[dser >= cutoff].reset_index(drop=True)
        except Exception:
            pass

    if len(df) < cfg.min_history_rows:
        logger.info("Insufficient data for %s (%d < %d)", path.stem, len(df), cfg.min_history_rows)
        return None

    logger.info("Computing features for %s with %d rows", path.stem, len(df))

    price, volume, daily_ret = prepare_price_volume(df)
    out = pd.DataFrame({"ticker": path.stem, "date": df["date"]})

    logp = pd.Series(np.log(price.replace(0, np.nan).to_numpy(dtype=float)), index=price.index)

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
        return safe_div(s - med, scale)

    feat: dict[str, pd.Series] = {}

    horizons = sorted({int(x) for x in (cfg.horizons or []) if int(x) > 0})

    # Momentum
    for w in horizons:
        mom_w = logp.diff(w)
        r = mom_w.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"ret_z_{w}d"] = safe_div(mom_w - mu, sd)

    # Price vs MA
    for w in horizons:
        ma = price.rolling(w).mean()
        ratio = safe_div(price, ma) - 1.0
        r = ratio.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"price_ma_ratio_z_{w}d"] = safe_div(ratio - mu, sd)

    # Volume percentile
    if cfg.use_volume_pctile:
        for w in [w for w in horizons if w >= 63]:
            feat[f"volume_pctile_{w}d"] = volume.rolling(window=w, min_periods=w).apply(rank_pct_last, raw=True)

    # Drawdown/run-up
    for w in [w for w in horizons if w >= 252]:
        roll_max = price.rolling(w).max()
        roll_alltime = price.cummax()

        dd = price / roll_max - 1.0
        r = dd.rolling(window=w, min_periods=w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"dd_cur_z_{w}d"] = safe_div(dd - mu, sd)

        dd_all = price / roll_alltime - 1.0
        r_all = dd_all.rolling(window=w, min_periods=w)
        mu_all, sd_all = r_all.mean(), r_all.std(ddof=0)
        feat[f"dd_alltime_z_{w}d"] = safe_div(dd_all - mu_all, sd_all)

        roll_min = price.rolling(w).min()
        du = price / roll_min - 1.0
        r2 = du.rolling(window=w, min_periods=w)
        mu2, sd2 = r2.mean(), r2.std(ddof=0)
        feat[f"du_cur_z_{w}d"] = safe_div(du - mu2, sd2)

    # ATH
    ath = price.cummax()
    off_ath = price / ath - 1.0
    feat["off_ath_z_252d"] = rolling_robust_z(off_ath, 252)

    # ATR / range / breakout / HV contraction
    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        prev_close = price.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        for w in [w for w in horizons if w >= 63]:
            atr = safe_div(tr.rolling(w, min_periods=w).mean(), price)
            feat[f"atr_z_{w}d"] = rolling_robust_z(atr, w)

            H = high.rolling(w, min_periods=w).max()
            L = low.rolling(w, min_periods=w).min()
            rng = (H - L).where((H - L) > 0)
            pos = (price - L) / rng
            signed_pos = pos - 0.5
            feat[f"pos_in_range_z_{w}d"] = rolling_robust_z(signed_pos, w)

        H52 = high.rolling(252, min_periods=252).max()
        L52 = low.rolling(252, min_periods=252).min()
        rng52 = (H52 - L52).replace(0, np.nan)
        feat["breakout_strength_52w"] = safe_div(price - H52, rng52)

        for wv in [w for w in horizons if w >= 252]:
            hv = daily_ret.rolling(wv).std(ddof=0)
            feat[f"hv_contraction_z_{wv}d"] = rolling_robust_z(hv, wv)

    # Upside vs downside vol
    for w in [w for w in horizons if w >= 63]:
        pos_vol = daily_ret.where(daily_ret > 0).rolling(window=w, min_periods=2).std(ddof=0)
        neg_vol = daily_ret.where(daily_ret < 0).rolling(window=w, min_periods=2).std(ddof=0)
        ratio = safe_div(pos_vol, (neg_vol + 1e-8)).clip(0, 50)
        log_ratio = pd.Series(np.log(ratio.replace(0, np.nan).to_numpy(dtype=float)), index=ratio.index)
        minp = max(20, w // 5)
        r = log_ratio.rolling(window=w, min_periods=minp)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"up_down_vol_ratio_z_{w}d"] = safe_div(log_ratio - mu, sd)

    # Dollar volume
    dollar_vol = price * volume
    for w in [w for w in horizons if w >= 63]:
        r = dollar_vol.rolling(w)
        mu, sd = r.mean(), r.std(ddof=0)
        feat[f"dollar_vol_z_{w}d"] = safe_div(dollar_vol - mu, sd)

    # RSI percentile
    stats_w = int(cfg.ret_stats_window or 63)
    for rw in [w for w in horizons if w >= 252]:
        rsi = compute_rsi(price, rw)
        feat[f"rsi_pctile_{rw}d"] = rsi.rolling(window=stats_w, min_periods=stats_w).apply(rank_pct_last, raw=True)

    # Return moments
    feat["ret_skew_63d"] = daily_ret.rolling(window=63, min_periods=63).skew()
    if cfg.use_ret_kurtosis:
        kw = int(cfg.ret_kurt_window or 63)
        try:
            feat[f"ret_kurt_{kw}d"] = daily_ret.rolling(window=kw, min_periods=kw).kurt()
        except Exception:
            feat[f"ret_kurt_{kw}d"] = pd.Series(np.nan, index=daily_ret.index)

    # Sharpe t-stat
    try:
        roll = daily_ret.rolling(window=252, min_periods=252)
        mu, sd, n = roll.mean(), roll.std(ddof=0), roll.count()
        feat["sharpe_tstat_252d"] = safe_div(mu * np.sqrt(n), sd)
    except Exception:
        feat["sharpe_tstat_252d"] = pd.Series(np.nan, index=daily_ret.index)

    # Stability
    try:
        hv21 = daily_ret.rolling(21, min_periods=21).std(ddof=0)
        vov252 = hv21.rolling(252, min_periods=252).std(ddof=0)
        rob = max(5, 252 // 4)

        def _mad_arr(x: np.ndarray) -> float:
            if x.size == 0:
                return np.nan
            m = np.nanmedian(x)
            return float(np.nanmedian(np.abs(x - m)))

        v_med = vov252.rolling(rob, min_periods=1).median()
        v_mad = vov252.rolling(rob, min_periods=1).apply(_mad_arr, raw=True)
        v_scale = v_mad.where(~v_mad.isna() & (v_mad > 0), vov252.rolling(rob, min_periods=1).std(ddof=0))
        feat["stability_z_252d"] = -safe_div(vov252 - v_med, v_scale)
    except Exception:
        feat["stability_z_252d"] = pd.Series(np.nan, index=daily_ret.index)

    # Target
    fwd = int(cfg.forward_days)
    out[f"target_fwd_{fwd}d"] = safe_div(price.shift(-fwd), price) - 1

    out = pd.concat([out, pd.DataFrame(feat)], axis=1)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).clip(-20, 20)

    out = apply_feature_filters(out, cfg)
    if out is None:
        logger.warning("No features remain after filtering for %s", path.stem)
        return None

    if min_date is not None and not out.empty:
        try:
            out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None).dt.normalize()
            out = out.loc[out["date"] >= pd.to_datetime(min_date).tz_localize(None)].reset_index(drop=True)
        except Exception:
            pass

    return out
