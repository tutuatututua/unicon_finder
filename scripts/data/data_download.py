"""Historical daily price downloader (yfinance) with incremental updates.

Primary public function: ``download_full_history``

Behavior:
    - If ``data/raw/<TICKER>.parquet`` does not exist, fetch full history (period='max').
    - If the file exists, fetch only missing dates since the last saved date and append.
    - Idempotent and safe: rows are normalized to date (UTC-naive, normalized) and de-duplicated.
"""

from pathlib import Path
from typing import List, Optional

import logging

from scripts.config.config import load_yaml_config
from scripts.config.paths import ProjectPaths
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from scripts.config.logging import get_logger  # switched to absolute import for script execution
logger = get_logger(__name__)

# yfinance can emit noisy ERROR logs for tickers with no data. We rely on our own
# logging for pipeline visibility.
logging.getLogger("yfinance").setLevel(logging.WARNING)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_RAW_DIR = Path("data/raw")


def _resolve_raw_dir(raw_dir: str | Path | None, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Resolve the raw data directory.

    Priority:
      1) explicit `raw_dir` argument
      2) config.yaml `paths.raw_dir` override
      3) default `data/raw`
    """
    if raw_dir is not None:
        return Path(raw_dir)
    try:
        cfg = load_yaml_config(config_path)
        return ProjectPaths.from_config(cfg).raw_dir
    except Exception:
        return DEFAULT_RAW_DIR


def _fetch_yahoo_daily_range(ticker: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Fetch daily history for a ticker in [start, end] inclusive using yfinance.

    If start is None, falls back to full history (period='max').
    """
    try:
        if start is not None and end is not None:
            s = pd.to_datetime(start)
            e = pd.to_datetime(end)
            if s > e:
                return pd.DataFrame()
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
    raw_dir: str | Path | None = None,
    allow_incremental: bool = True,
) -> List[str]:
    """Download or update daily price history for all tickers in the universe.

    - If a ticker parquet is missing: fetch full history and create it.
    - If it exists and ``allow_incremental`` is True: append only missing rows.
    - If it exists and ``allow_incremental`` is False: skip.

    Returns: list of tickers written/updated.
    """
    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)

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
        p = resolved_raw_dir / f"{t}.parquet"
        if not p.exists():
            # Bootstrap full history
            df_full = _fetch_yahoo_daily_range(t, start=None, end=None)
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

        # Incremental update: read last saved date and fetch [last+1d, last_business_day].
        # Daily bars for "today" are often unavailable until after market close;
        # requesting through a future date can cause noisy yfinance errors.
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

        # End date (inclusive): last business day in UTC (tz-naive midnight).
        today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        end = (today_utc - pd.tseries.offsets.BDay(1))

        if start is not None and pd.to_datetime(start) > pd.to_datetime(end):
            logger.info(f"No new rows for {t} since {last_dt}")
            continue

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
    logger.info("Download complete. Tickers written/updated: %d", len(processed))
    return processed


__all__ = [
    'download_full_history',
]