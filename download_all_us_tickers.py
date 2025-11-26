import os
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Reliable sources from Nasdaq Trader Symbol Directory
NASDAQLISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"  # NYSE/AMEX/ARCA etc.


def _requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "unicon-finder/1.0 (+https://github.com/)",
            "Accept": "text/plain, text/csv, */*",
        }
    )
    return session


def _read_symdir_text(text: str, sep: str = "|") -> pd.DataFrame:
    # Files end with a footer line like "File Creation Time:" — drop it.
    lines = [ln for ln in text.splitlines() if not ln.startswith("File Creation Time:")]
    cleaned = "\n".join(lines)
    df = pd.read_csv(StringIO(cleaned), sep=sep)
    return df


def get_all_tickers(include_etfs: bool = False) -> list[str]:
    session = _requests_session()
    tickers: set[str] = set()

    # 1) NASDAQ listed
    print("Downloading NASDAQ tickers (nasdaqlisted.txt)...")
    resp = session.get(NASDAQLISTED_URL, timeout=20)
    resp.raise_for_status()
    df_nas = _read_symdir_text(resp.text)
    # Filter test issues; optionally filter ETFs
    if "Test Issue" in df_nas.columns:
        df_nas = df_nas[df_nas["Test Issue"].astype(str).str.upper() != "Y"]
    if not include_etfs and "ETF" in df_nas.columns:
        df_nas = df_nas[df_nas["ETF"].astype(str).str.upper() != "Y"]
    if "Symbol" in df_nas.columns:
        for sym in tqdm(df_nas["Symbol"].astype(str), desc="NASDAQ symbols"):
            s = sym.strip().upper()
            if s:
                tickers.add(s)

    # 2) OTHER LISTED (NYSE/AMEX/ARCA)
    print("Downloading NYSE/AMEX/ARCA tickers (otherlisted.txt)...")
    resp = session.get(OTHERLISTED_URL, timeout=20)
    resp.raise_for_status()
    df_other = _read_symdir_text(resp.text)
    # Filter test issues; optionally filter ETFs
    if "Test Issue" in df_other.columns:
        df_other = df_other[df_other["Test Issue"].astype(str).str.upper() != "Y"]
    if not include_etfs and "ETF" in df_other.columns:
        df_other = df_other[df_other["ETF"].astype(str).str.upper() != "Y"]
    # Keep only primary exchanges NYSE/AMEX/ARCA codes if present
    valid_exch = {"N", "A", "P"}  # NYSE, AMEX, ARCA
    if "Exchange" in df_other.columns:
        df_other = df_other[df_other["Exchange"].astype(str).str.upper().isin(valid_exch)]

    symbol_col = (
        "ACT Symbol" if "ACT Symbol" in df_other.columns else ("NASDAQ Symbol" if "NASDAQ Symbol" in df_other.columns else None)
    )
    if symbol_col:
        for sym in tqdm(df_other[symbol_col].astype(str), desc="OTHER symbols"):
            s = sym.strip().upper()
            if s:
                tickers.add(s)

    return sorted(tickers)


if __name__ == "__main__":
    print("Fetching all US stock tickers (NASDAQ, NYSE, AMEX)...")
    syms = get_all_tickers(include_etfs=False)
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"ticker": syms}).to_csv("data/universe.csv", index=False)
    print(f"Saved {len(syms)} tickers to data/universe.csv")
