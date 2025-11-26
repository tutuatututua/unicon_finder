"""Historical daily price downloader (yfinance) with incremental updates.

Primary public function: ``download_full_history``

Behavior:
    - If ``data/raw/<TICKER>.parquet`` does not exist, fetch full history (period='max').
    - If the file exists, fetch only missing dates since the last saved date and append.
    - Idempotent and safe: rows are normalized to date (UTC-naive, normalized) and de-duplicated.
"""

from pathlib import Path
from typing import List, Optional
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from logging_config import get_logger  # switched to absolute import for script execution
import yaml

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
try:
    cfg_p = Path("config.yaml")
    if cfg_p.exists():
        with cfg_p.open("r", encoding="utf-8") as _f:
            _raw_cfg = yaml.safe_load(_f) or {}
        _paths = (_raw_cfg.get("paths") or {})
        RAW_DIR = Path(_paths.get("raw_dir", RAW_DIR))
except Exception:
    pass
RAW_DIR.mkdir(parents=True, exist_ok=True)







def _fetch_yahoo_daily_full(ticker: str) -> pd.DataFrame:
    """Fetch full available daily history for a ticker using period='max'.

    We call yfinance with ``period='max'`` to retrieve the entire available history.
    Returns empty DataFrame on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        # Request the full available history from Yahoo
        df = tk.history(period='max', interval='1d', auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = 'date'
        cols = {c: c.lower().replace(' ', '_') for c in df.columns}
        df.rename(columns=cols, inplace=True)
        return df
    except Exception as e:  # pragma: no cover
        logger.warning(f"Yahoo full-history fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_yahoo_daily_range(ticker: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Fetch daily history for a ticker in [start, end] inclusive using yfinance.

    If start is None, falls back to full history (period='max').
    """
    try:
        tk = yf.Ticker(ticker)
        if start is None:
            df = tk.history(period='max', interval='1d', auto_adjust=True)
        else:
            end_plus = (end + pd.Timedelta(days=1)) if end is not None else None
            df = tk.history(start=start, end=end_plus, interval='1d', auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = 'date'
        cols = {c: c.lower().replace(' ', '_') for c in df.columns}
        df.rename(columns=cols, inplace=True)
        return df
    except Exception as e:  # pragma: no cover
        logger.warning(f"Yahoo range fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _sanitize_universe(tickers: List[str]) -> List[str]:
    import re
    pat = re.compile(r"^[A-Z]+$")
    out: List[str] = []
    for t in tickers:
        tu = t.upper()
        if pat.match(tu):
            out.append(tu)
    return out

def download_full_history(
    universe_csv: str | Path = "data/universe.csv",
    raw_dir: str | Path = RAW_DIR,
    allow_incremental: bool = True,
) -> List[str]:
    """Download or update daily price history for all tickers in the universe.

    - If a ticker parquet is missing: fetch full history and create it.
    - If it exists and ``allow_incremental`` is True: append only missing rows.
    - If it exists and ``allow_incremental`` is False: skip.

    Returns: list of tickers written/updated.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    uni = Path(universe_csv)
    if not uni.exists():
        raise FileNotFoundError(f"Universe CSV not found: {uni}")
    df = pd.read_csv(uni)
    first_col = df.columns[0]
    raw_tickers = df[first_col].astype(str).tolist()


    tickers = _sanitize_universe(raw_tickers)

    if not tickers:
        raise ValueError("No valid tickers found in universe CSV.")
    
    processed: List[str] = []
    for t in tqdm(tickers, desc="download/update daily history"):
        p = Path(raw_dir) / f"{t}.parquet"
        if not p.exists():
            # Bootstrap full history
            df_full = _fetch_yahoo_daily_full(t)
            if df_full.empty:
                logger.warning(f"Skipping {t}: empty dataframe returned from yfinance")
                continue
            out_df = df_full.reset_index()
            out_df['date'] = pd.to_datetime(out_df['date'], utc=True).dt.tz_localize(None).dt.normalize()
            out_df.sort_values('date', inplace=True)
            out_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            out_df.to_parquet(p, index=False)
            processed.append(t)
            logger.info(f"Wrote full history {t}: {len(out_df)} rows")
            continue

        if not allow_incremental:
            logger.info(f"Skipping {t}: file exists and incremental disabled")
            continue

        # Incremental update: read last saved date and fetch [last+1d, today]
        try:
            cur = pd.read_parquet(p, columns=['date'])
            cur['date'] = pd.to_datetime(cur['date'], utc=True).dt.tz_localize(None).dt.normalize()
        except Exception as e:
            logger.warning(f"Failed reading existing parquet for {t}, refetch full. Err={e}")
            cur = pd.DataFrame(columns=['date'])
        last_dt: Optional[pd.Timestamp] = None
        if not cur.empty:
            last_dt = pd.to_datetime(cur['date']).max()

        start = (last_dt + pd.Timedelta(days=1)) if last_dt is not None else None
        # Compute 'end' as UTC midnight (tz-naive). In newer pandas, Timestamp.utcnow() is tz-aware (UTC),
        # so we must not tz_localize('UTC') on it. Normalize first (keeps UTC), then drop tz to get naive.
        end = pd.Timestamp.utcnow().normalize().tz_localize(None)
        df_new = _fetch_yahoo_daily_range(t, start=start, end=end)
        if df_new.empty:
            logger.info(f"No new rows for {t} since {last_dt}")
            continue
        inc = df_new.reset_index()
        inc['date'] = pd.to_datetime(inc['date'], utc=True).dt.tz_localize(None).dt.normalize()
        inc.sort_values('date', inplace=True)
        try:
            base = pd.read_parquet(p)
        except Exception:
            base = pd.DataFrame()
        merged = pd.concat([base, inc], axis=0, ignore_index=True)
        merged.sort_values('date', inplace=True)
        merged.drop_duplicates(subset=['date'], keep='last', inplace=True)
        merged.to_parquet(p, index=False)
        processed.append(t)
        logger.info(f"Updated {t}: +{len(inc)} new rows (now {len(merged)})")
    logger.info(f"Download complete. Tickers written/updated: {len(processed)}")
    return processed


__all__ = [
    'download_full_history',
]