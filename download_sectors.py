"""Fetch sector/industry metadata for the universe and cache to CSV.

- Reads tickers from data/universe.csv (first column)
- Uses yfinance to fetch sector & industry for each ticker
- Writes/updates data/sector_map.csv with columns: ticker, sector, industry, source, fetched_at
- Skips tickers that already exist in the cache unless --force is set

Notes:
- Yahoo can be flaky; we add small retries and backoff
- Some tickers (ETFs, preferreds, SPAC warrants) may not have sector/industry
- This script is optional; the feature builder will only use the map if present
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import time
import sys
import argparse
import datetime as dt

import pandas as pd
import yfinance as yf

from logging_config import get_logger

logger = get_logger(__name__)

UNIVERSE_CSV = Path("data/universe.csv")
SECTOR_MAP_CSV = Path("data/sector_map.csv")


def _load_universe(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe not found: {csv_path}")
    df = pd.read_csv(csv_path)
    tickers = df.iloc[:, 0].astype(str).str.upper().tolist()
    # Filter out symbols with non A-Z characters to avoid yfinance errors
    tickers = [t for t in tickers if t.isalpha()]
    return tickers


def _load_cache(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["ticker", "sector", "industry", "source", "fetched_at"]).astype({"ticker": str})
    df = pd.read_csv(csv_path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _fetch_one(ticker: str, retries: int = 2, pause: float = 0.25) -> Dict[str, str | None]:
    for i in range(retries + 1):
        try:
            tk = yf.Ticker(ticker)
            # yfinance get_info() returns dict; no timeout argument supported
            info = tk.get_info()
            sector = info.get("sector") if isinstance(info, dict) else None
            industry = info.get("industry") if isinstance(info, dict) else None
            if sector or industry:
                return {"sector": sector, "industry": industry, "source": "yfinance"}
        except Exception as e:
            if i == retries:
                logger.warning("[sector] %s: failed after %d retries: %s", ticker, retries, e)
            time.sleep(pause * (2 ** i))
    return {"sector": None, "industry": None, "source": "yfinance"}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download sector/industry for tickers")
    parser.add_argument("--universe", type=str, default=str(UNIVERSE_CSV), help="Path to universe.csv")
    parser.add_argument("--out", type=str, default=str(SECTOR_MAP_CSV), help="Output CSV path for sector map")
    parser.add_argument("--force", action="store_true", help="Refetch all tickers even if cached")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers to fetch (for quick tests)")
    args = parser.parse_args(argv)

    uni_p = Path(args.universe)
    out_p = Path(args.out)

    tickers = _load_universe(uni_p)
    cache = _load_cache(out_p)

    if not args.force and not cache.empty:
        have = set(cache["ticker"].unique())
        tickers = [t for t in tickers if t not in have]
        logger.info("Skipping %d cached tickers; %d to fetch", len(have), len(tickers))

    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]

    rows: List[Dict[str, str]] = []
    now = pd.Timestamp.utcnow().normalize().tz_localize(None)

    for i, t in enumerate(tickers, 1):
        meta = _fetch_one(t)
        rows.append({
            "ticker": t,
            "sector": meta.get("sector") or "Unknown",
            "industry": meta.get("industry") or "Unknown",
            "source": meta.get("source") or "yfinance",
            "fetched_at": str(now),
        })
        if i % 50 == 0:
            logger.info("Fetched %d/%d sectors...", i, len(tickers))
        time.sleep(0.05)  # gentle pacing

    new_df = pd.DataFrame(rows)
    if cache.empty:
        out_df = new_df
    else:
        out_df = pd.concat([cache, new_df], ignore_index=True)
        # de-duplicate keeping the latest entry
        out_df.sort_values(["ticker", "fetched_at"], inplace=True)
        out_df = out_df.groupby("ticker", as_index=False).tail(1)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    logger.info("Saved sector map: %s (rows=%d)", out_p, len(out_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
