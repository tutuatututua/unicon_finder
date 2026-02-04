from pathlib import Path
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(f"yfinance not available in this environment: {e}")

BENC_DIR = Path("data/benchmark")
BENC_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching ^GSPC from Yahoo Finance...")
tk = yf.Ticker("^SPX")
hist = tk.history(period="max", interval="1d", auto_adjust=True)
if hist is None or hist.empty:
    raise SystemExit("Failed to fetch ^GSPC or empty history returned.")

hist = hist.reset_index()
hist["Date"] = pd.to_datetime(hist["Date"], utc=True, errors="coerce").dt.normalize()
# Normalize columns
if "Date" in hist.columns:
    hist.rename(columns={"Date": "date"}, inplace=True)
if "Close" in hist.columns:
    hist.rename(columns={"Close": "close"}, inplace=True)

if "date" not in hist.columns or "close" not in hist.columns:
    # Try heuristic mapping
    date_col = hist.columns[0]
    close_col = [c for c in hist.columns if c.lower().startswith("close")] or [hist.columns[-1]]
    hist = hist.rename(columns={date_col: "date", close_col[0]: "close"})

hist["date"] = pd.to_datetime(hist["date"], utc=True, errors="coerce").dt.tz_localize(None)
spx = hist[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)

out_path = BENC_DIR / "sp500.parquet"
spx.to_parquet(out_path, index=False)
print(f"Wrote {len(spx)} rows to {out_path}")