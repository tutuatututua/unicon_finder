"""Cross-sectional ranking backtest (strict mode).

This script evaluates a ranking signal (either a LightGBM model score or a
single feature column) against a forward return target over time.

For each date it computes:
- Mean forward return of the Top-N and Bottom-N names by score and their spread
- Per-date decile mean returns
- NDCG@K and mean forward returns for top-k buckets (k in {5,10,20})
- Optional S&P 500 benchmark alignment and excess returns for top-k

Outputs:
- backtest/backtest_timeseries.csv
- backtest/backtest_deciles.csv
- backtest/backtest_ndcg.csv (+ backtest_ndcg.xlsx)
- backtest/backtest_picks_top.csv (+ backtest_picks_top.xlsx)
- backtest/backtest_summary.json

Strict mode: no defensive fallbacks. If a resource is missing or a step fails,
the script raises and exits. Keep your data and paths correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, cast

import argparse
import json
import numpy as np
import pandas as pd
import re

import lightgbm as lgb  # require if using model scoring

from logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths and target resolution
# ---------------------------------------------------------------------------
PROC_DIR = Path("backtest")
RAW_FEATURES_PATH = Path("data/processed/raw_training.parquet")
FEATURE_META = Path("models/features.json")
MODEL_PATH = Path("models/lightgbm_model.txt")
BENC_CSV = Path("data/benchmark/sp500.csv")


def _infer_forward_days_from_target(target: str) -> int:
    """Infer numeric forward days from a target column like 'target_fwd_252d'."""
    m = re.search(r"(\d+)", str(target))
    if not m:
        raise ValueError(f"Could not infer forward days from target name: {target}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Config and results dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    features_path: Path = RAW_FEATURES_PATH
    model_path: Path = MODEL_PATH
    use_model_score: bool = True           # if False, use score_col
    score_col: Optional[str] = None        # e.g., "price_ma_ratio_z_63d"
    target_col: Optional[str] = None       # if None, read from models/features.json
    top_n: int = 20
    bottom_n: int = 20
    require_min_cross_section: int = 30    # skip dates with too few names
    save_dir: Path = PROC_DIR
    benchmark_csv: Path = BENC_CSV


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("Features frame missing required 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _score_with_model(df: pd.DataFrame, model_path: Path) -> Tuple[pd.DataFrame, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    feats = list(booster.feature_name())
    if not feats:
        raise RuntimeError("Loaded model has no feature names")
    for c in ("sector", "industry"):
        if c in feats and c in df.columns:
            df[c] = pd.Categorical(df[c].astype("string"))
    X = df[feats]
    out = df.copy()
    out["score"] = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
    return out, feats


def _score_with_column(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, list[str]]:
    if col not in df.columns:
        raise KeyError(f"Requested score_col '{col}' not found in features")
    out = df.copy()
    out["score"] = out[col]
    return out, [col]


def _per_date_top_bottom(df: pd.DataFrame, *, top_n: int, bottom_n: int, min_cs: int, target_col: str) -> pd.DataFrame:
    records = []
    grouped = df.groupby("date", sort=True)
    for dt, g in grouped:
        g_eval = g.dropna(subset=["score", target_col])
        if len(g_eval) < max(min_cs, top_n + bottom_n):
            continue
        g_eval = g_eval.sort_values("score", ascending=False)
        top = g_eval.head(top_n)
        bottom = g_eval.tail(bottom_n)
        rec = {
            "date": dt,
            "n_eval": int(len(g_eval)),
            "top_mean": float(top[target_col].mean()),
            "bottom_mean": float(bottom[target_col].mean()),
        }
        rec["spread"] = rec["top_mean"] - rec["bottom_mean"]
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _load_sp500_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize columns
    col_map = {}
    if "Date" in df.columns:
        col_map["Date"] = "date"
    if "Close" in df.columns:
        col_map["Close"] = "close"
    if col_map:
        df = df.rename(columns=col_map)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("Benchmark CSV must contain 'date' and 'close' columns (or 'Date'/'Close').")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date", "close"]].dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def _attach_next_date_and_spx(df: pd.DataFrame, *, target_col: str, benchmark_csv: Path) -> pd.DataFrame:
    """Attach 'sp500_return' and 'sp500_return_date' to a per-date frame.

    - sp500_return is the forward return from 'date' to the S&P500 trading day
      forward_days ahead: close(t+fwd)/close(t) - 1.
    - sp500_return_date is that destination S&P500 trading date (t+fwd).
    """
    if "date" not in df.columns:
        return df
    spx = _load_sp500_csv(benchmark_csv)
    # Ensure unique dates in SPX
    spx = spx.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Build mapping from date to forward-days ahead date and compute return
    fwd_days = _infer_forward_days_from_target(target_col)
    spx_map = spx[["date", "close"]].copy()
    # Base columns
    spx_map = spx_map.rename(columns={"close": "sp500_close"})
    # Per-date absolute delta (close - previous close)
    spx_map["sp500_per_date_prev"] = spx_map["sp500_close"] - spx_map["sp500_close"].shift(1)
    spx_map["sp500_per_date_at_return"] = spx_map["sp500_per_date_prev"].shift(-fwd_days)
    # Forward mapping for target horizon (align all outputs to the forward date)
    spx_map["sp500_return_date"] = spx_map["date"].shift(-fwd_days)
    # Forward return from t to t+fwd
    spx_map["sp500_return"] = (spx_map["sp500_close"].shift(-fwd_days) / spx_map["sp500_close"]) - 1.0

    

    spx_map.drop(columns=["sp500_close"], inplace=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    spx_map["date"] = pd.to_datetime(spx_map["date"]).dt.date
    out = out.merge(spx_map, on="date", how="left")
    return out


def _per_date_deciles(df: pd.DataFrame, *, min_cs: int, target_col: str) -> pd.DataFrame:
    """Compute decile mean forward returns per date.

    Returns long-form DataFrame with columns: date, decile (1..10), mean_ret.
    """
    rows = []
    for dt, g in df.groupby("date", sort=True):
        ge = g.dropna(subset=["score", target_col]).copy()
        if len(ge) < max(min_cs, 50):  # need enough for deciles
            continue
        # rank into 10 bins (1=lowest, 10=highest)
        ge["decile"] = pd.qcut(ge["score"], q=10, labels=False, duplicates="drop")
        ge["decile"] = ge["decile"].astype("Int64")
        # normalize to 1..10 if available
        if ge["decile"].notna().any():
            dmin = int(ge["decile"].min())
            ge["decile"] = ge["decile"] - dmin + 1
        for dval, gg in ge.groupby("decile"):
            if pd.isna(dval):
                continue
            # Robust int conversion for pandas/NumPy scalars
            dval_int = int(dval)  # type: ignore[arg-type]
            rows.append({
                "date": dt,
                "decile": dval_int,
                "mean_ret": float(gg[target_col].mean()),
                "count": int(len(gg)),
            })
    return pd.DataFrame(rows)


def _label_relevance(y: pd.Series, n_bins: int = 7, force_negatives_to_zero: bool = True) -> pd.Series:
    """Bin continuous targets into non-negative integer relevance labels.

    Uses quantiles; falls back to rank-based bins if needed. Optionally
    forces negative targets to relevance 0.
    """
    y = pd.to_numeric(y, errors="coerce")
    rel = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    rel = rel.astype("Int64")
    if rel.notna().any():
        rel = rel - int(rel.min())
    if force_negatives_to_zero:
        neg_mask = (y < 0) & y.notna() & rel.notna()
        if neg_mask.any():
            rel.loc[neg_mask] = 0
    return rel.astype("Int64")


def _ndcg_at_k(labels_sorted: np.ndarray, k: int) -> float:
    k_eff = min(k, len(labels_sorted))
    if k_eff <= 0:
        return 0.0
    gains = (2.0 ** labels_sorted[:k_eff] - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels_sorted)[::-1][:k_eff]
    idcg = float(np.sum((2.0 ** ideal - 1.0) * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _ndcg_timeseries(df: pd.DataFrame, *, ks: Tuple[int, ...], n_bins: int, min_cs: int, target_col: str) -> pd.DataFrame:
    recs = []
    for dt, g in df.groupby("date", sort=True):
        ge = g.dropna(subset=["score", target_col]).copy()
        if len(ge) < min_cs:
            continue
        rel = _label_relevance(ge[target_col], n_bins=n_bins)
        ge = ge.loc[rel.index].copy()
        ge["rel"] = rel
        ge = ge.dropna(subset=["rel"]).astype({"rel": int})
        if ge.empty:
            continue
        ge = ge.sort_values("score", ascending=False)
        labs = ge["rel"].to_numpy(dtype=float)
        row = {"date": dt, "n_eval": int(len(ge))}
        for k in ks:
            row[f"ndcg@{k}"] = _ndcg_at_k(labs, k)
            # also compute mean forward return of top-k by score
            row[f"mean_return_{k}"] = float(pd.to_numeric(ge[target_col], errors="coerce").head(k).mean())
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def _collect_top_picks(df: pd.DataFrame, *, top_n: int, min_cs: int, target_col: str) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby("date", sort=True):
        ge = g.dropna(subset=["score", target_col]).copy()
        if len(ge) < max(min_cs, top_n):
            continue
        ge = ge.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        ge["rank"] = np.arange(1, len(ge) + 1, dtype=int)
        for _, r in ge.iterrows():
            rows.append({
                "date": dt,
                "ticker": r.get("ticker"),
                "rank": int(r["rank"]),
                "score": float(r["score"]),
                "fwd_return": float(r[target_col]),
                "sector": r.get("sector"),
                "industry": r.get("industry"),
            })
    return pd.DataFrame(rows)


def _read_target_from_meta(meta_path: Path) -> str:
    if not meta_path.exists():
        raise FileNotFoundError(f"Feature meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target = str(meta.get("target") or "").strip()
    if not target:
        raise ValueError("'target' missing in features meta")
    return target


def _tstat(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(len(x)))


def run_backtest(cfg: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    """Run the cross-sectional ranking backtest and persist artifacts."""
    cfg = cfg or BacktestConfig()
    target_col = cfg.target_col or _read_target_from_meta(FEATURE_META)

    df = _load_features(cfg.features_path)
    df = df[df[target_col].notna()].copy()
    logger.info("Backtest frame: rows=%d cols=%d dates=%d", len(df), df.shape[1], df["date"].nunique())

    # Score
    if cfg.use_model_score:
        df_scored, used_signal = _score_with_model(df, cfg.model_path)
        logger.info("Scored with model: features=%d", len(used_signal))
    else:
        if not cfg.score_col:
            raise ValueError("score_col must be provided when use_model_score is False")
        df_scored, used_signal = _score_with_column(df, cfg.score_col)

    # Per-date top/bottom
    ts = _per_date_top_bottom(
        df_scored,
        top_n=cfg.top_n,
        bottom_n=cfg.bottom_n,
        min_cs=cfg.require_min_cross_section,
        target_col=target_col,
    )
    if ts.empty:
        raise RuntimeError("Backtest produced no time-series (insufficient cross-section per date?)")
    ts = _attach_next_date_and_spx(ts, target_col=target_col, benchmark_csv=cfg.benchmark_csv)

    # Deciles
    dec = _per_date_deciles(df_scored, min_cs=cfg.require_min_cross_section, target_col=target_col)

    # NDCG time series
    nd = _ndcg_timeseries(df_scored, ks=(5, 10, 20, 100,500), n_bins=7, min_cs=cfg.require_min_cross_section, target_col=target_col)
    if not nd.empty:
        nd = _attach_next_date_and_spx(nd, target_col=target_col, benchmark_csv=cfg.benchmark_csv)
        for k in (5, 10, 20, 100,500):
            mean_col = f"mean_return_{k}"
            out_col = f"ndcg{k}-sp500"
            if mean_col in nd.columns and "sp500_return" in nd.columns:
                nd[out_col] = pd.to_numeric(nd[mean_col], errors="coerce") - pd.to_numeric(nd["sp500_return"], errors="coerce")

    # Top picks per day
    picks = _collect_top_picks(df_scored, top_n=cfg.top_n, min_cs=cfg.require_min_cross_section, target_col=target_col)

    # Summary stats
    summary = {
        "dates": int(ts.shape[0]),
        "avg_top": float(ts["top_mean"].mean()),
        "avg_bottom": float(ts["bottom_mean"].mean()),
        "avg_spread": float(ts["spread"].mean()),
        "t_spread": _tstat(ts["spread"]),
        "avg_ndcg@5": float(nd["ndcg@5"].mean()) if not nd.empty else float("nan"),
        "avg_ndcg@10": float(nd["ndcg@10"].mean()) if not nd.empty else float("nan"),
        "avg_ndcg@20": float(nd["ndcg@20"].mean()) if not nd.empty else float("nan"),
        "signal": used_signal,
        "target_col": target_col,
    }

    # Persist artifacts
    out_dir = cfg.save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = out_dir / "backtest_timeseries.csv"
    dec_path = out_dir / "backtest_deciles.csv"
    sum_path = out_dir / "backtest_summary.json"
    nd_path = out_dir / "backtest_ndcg.csv"
    nd_xlsx_path = out_dir / "backtest_ndcg.xlsx"
    picks_path = out_dir / "backtest_picks_top.csv"
    picks_xlsx_path = out_dir / "backtest_picks_top.xlsx"

    ts.to_csv(ts_path, index=False, date_format="%Y-%m-%d")
    dec.to_csv(dec_path, index=False, date_format="%Y-%m-%d")
    if not nd.empty:
        nd.to_csv(nd_path, index=False, date_format="%Y-%m-%d")
    if not picks.empty:
        picks.to_csv(picks_path, index=False, date_format="%Y-%m-%d")

    # Styled Excel exports
    def _write_excel_with_sign_colors(df_in: pd.DataFrame, path: Path, columns_to_color: list[str]) -> None:
        if df_in.empty:
            return
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            sheet_name = path.stem[:31]
            df_in.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = cast(Any, writer.book)
            worksheet = writer.sheets[sheet_name]
            green_fmt = workbook.add_format({"font_color": "#006100", "bg_color": "#C6EFCE"})
            red_fmt = workbook.add_format({"font_color": "#9C0006", "bg_color": "#FFC7CE"})
            n_rows, _ = df_in.shape
            for col_name in columns_to_color:
                if col_name not in df_in.columns:
                    continue
                cidx = list(df_in.columns).index(col_name)
                worksheet.conditional_format(1, cidx, n_rows, cidx, {
                    "type": "cell", "criteria": ">", "value": 0, "format": green_fmt,
                })
                worksheet.conditional_format(1, cidx, n_rows, cidx, {
                    "type": "cell", "criteria": "<=", "value": 0, "format": red_fmt,
                })

    if not picks.empty:
        _write_excel_with_sign_colors(picks, picks_xlsx_path, ["fwd_return"])

    if not nd.empty:
        cols_to_color = [c for c in nd.columns if c.startswith("mean_return_")]
        cols_to_color.append("sp500_return")
        cols_to_color.append("sp500_per_date")
        cols_to_color.append("sp500_per_date_prev")
        cols_to_color.append("sp500_per_date_at_return")

        for k in (5, 10, 20, 100, 500):
            col = f"ndcg{k}-sp500"
            if col in nd.columns and col not in cols_to_color:
                cols_to_color.append(col)
        _write_excel_with_sign_colors(nd, nd_xlsx_path, cols_to_color)

    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Backtest outputs: %s , %s , %s%s%s",
        ts_path,
        dec_path,
        sum_path,
        f" , {nd_path}" if nd_path.exists() else "",
        f" , {picks_path}" if picks_path.exists() else "",
    )

    return {
        **summary,
        "timeseries_csv": str(ts_path),
        "deciles_csv": str(dec_path),
        "ndcg_csv": str(nd_path) if nd_path.exists() else None,
        "ndcg_xlsx": str(nd_xlsx_path) if nd_xlsx_path.exists() else None,
        "picks_csv": str(picks_path) if picks_path.exists() else None,
        "picks_xlsx": str(picks_xlsx_path) if picks_xlsx_path.exists() else None,
        "summary_json": str(sum_path),
    }


def backtest_from_aggregated_scores(
    agg_csv: Path,
    *,
    ks: tuple[int, ...] = (5, 10, 20, 100, 500),
    min_cs: int = 100,
    features_path: Path = RAW_FEATURES_PATH,
    benchmark_csv: Path = BENC_CSV,
    dates: Optional[list[pd.Timestamp]] = None,
    output_stem: str = "backtest_ndcg_from_sum_scores",
) -> Dict[str, Any]:
    """Compute an NDCG-style backtest using an aggregated per-ticker score.

    Contract
    - Inputs: CSV with at least columns [ticker, total_score] (others ignored)
    - For each date in features where those tickers have targets, we treat
      total_score as a constant ranking signal and compute ndcg@K and mean_return_K.
    - Output: CSV under backtest/ with same schema as backtest_ndcg.csv
    """
    if not Path(agg_csv).exists():
        raise FileNotFoundError(f"Aggregated score CSV not found: {agg_csv}")

    target_col = _read_target_from_meta(FEATURE_META)

    agg = pd.read_csv(agg_csv)
    if "ticker" not in agg.columns or ("total_score" not in agg.columns and "score" not in agg.columns):
        raise ValueError("Aggregated CSV must contain 'ticker' and 'total_score' (or 'score') columns")
    score_col = "total_score" if "total_score" in agg.columns else "score"
    agg = agg[["ticker", score_col]].rename(columns={score_col: "score"})
    agg["ticker"] = agg["ticker"].astype(str)

    df = _load_features(features_path)
    df = df[df[target_col].notna()].copy()
    df["ticker"] = df["ticker"].astype(str)

    # Optional date filter (e.g., to align with aggregation window)
    if dates is not None and len(dates) > 0:
        dates_norm = pd.to_datetime(pd.Index(dates)).normalize().unique()
        df = df[df["date"].isin(dates_norm)].copy()

    # Keep only tickers present in the aggregated list and attach constant score
    df = df.merge(agg, on="ticker", how="inner")
    if df.empty:
        raise RuntimeError("No overlap between aggregated tickers and features")

    nd = _ndcg_timeseries(df, ks=ks, n_bins=7, min_cs=min_cs, target_col=target_col)
    if nd.empty:
        raise RuntimeError("Aggregated-score backtest produced no rows; check cross-section per date")

    nd = _attach_next_date_and_spx(nd, target_col=target_col, benchmark_csv=benchmark_csv)
    # Excess returns vs SP500
    for k in ks:
        mean_col = f"mean_return_{k}"
        out_col = f"ndcg{k}-sp500"
        if mean_col in nd.columns and "sp500_return" in nd.columns:
            nd[out_col] = pd.to_numeric(nd[mean_col], errors="coerce") - pd.to_numeric(nd["sp500_return"], errors="coerce")

    # Persist
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{output_stem}.csv"
    out_xlsx = PROC_DIR / f"{output_stem}.xlsx"
    nd.to_csv(out_csv, index=False, date_format="%Y-%m-%d")

    # Optional styled Excel
    try:
        cols_to_color = [c for c in nd.columns if c.startswith("mean_return_")]
        if "sp500_return" in nd.columns:
            cols_to_color.append("sp500_return")
        if "sp500_per_date" in nd.columns:
            cols_to_color.append("sp500_per_date")
        if "sp500_per_date_prev" in nd.columns:
            cols_to_color.append("sp500_per_date_prev")
        if "sp500_per_date_at_return" in nd.columns:
            cols_to_color.append("sp500_per_date_at_return")
        for k in ks:
            col = f"ndcg{k}-sp500"
            if col in nd.columns and col not in cols_to_color:
                cols_to_color.append(col)
        # Reuse inner writer util
        def _write_excel_with_sign_colors(df_in: pd.DataFrame, path: Path, columns_to_color: list[str]) -> None:
            if df_in.empty:
                return
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                sheet_name = path.stem[:31]
                df_in.to_excel(writer, sheet_name=sheet_name, index=False)
                workbook = cast(Any, writer.book)
                worksheet = writer.sheets[sheet_name]
                green_fmt = workbook.add_format({"font_color": "#006100", "bg_color": "#C6EFCE"})
                red_fmt = workbook.add_format({"font_color": "#9C0006", "bg_color": "#FFC7CE"})
                n_rows, _ = df_in.shape
                for col_name in columns_to_color:
                    if col_name not in df_in.columns:
                        continue
                    cidx = list(df_in.columns).index(col_name)
                    worksheet.conditional_format(1, cidx, n_rows, cidx, {
                        "type": "cell", "criteria": ">", "value": 0, "format": green_fmt,
                    })
                    worksheet.conditional_format(1, cidx, n_rows, cidx, {
                        "type": "cell", "criteria": "<=", "value": 0, "format": red_fmt,
                    })
        _write_excel_with_sign_colors(nd, out_xlsx, cols_to_color)
    except Exception:
        # Excel export is best-effort; ignore errors in headless envs
        pass

    return {
        "ndcg_csv": str(out_csv),
        "ndcg_xlsx": str(out_xlsx),
        "rows": int(nd.shape[0]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-sectional ranking backtest (strict)")
    p.add_argument("--features-path", type=Path, default=RAW_FEATURES_PATH, help="Path to features parquet")
    p.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to LightGBM model.txt")
    p.add_argument("--use-model-score", action="store_true", default=True, help="Use LightGBM model score")
    p.add_argument("--no-model-score", dest="use_model_score", action="store_false", help="Use a feature column as score")
    p.add_argument("--score-col", type=str, default=None, help="Feature column to use as score when not using model")
    p.add_argument("--target-col", type=str, default=None, help="Target forward return column; default reads models/features.json")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--bottom-n", type=int, default=20)
    p.add_argument("--min-cross-section", type=int, default=30)
    p.add_argument("--save-dir", type=Path, default=PROC_DIR)
    p.add_argument("--benchmark-csv", type=Path, default=BENC_CSV, help="S&P500 benchmark CSV with date,close")
    return p


if __name__ == "__main__":
    ap = _build_arg_parser()
    args = ap.parse_args()
    cfg = BacktestConfig(
        features_path=args.features_path,
        model_path=args.model_path,
        use_model_score=args.use_model_score,
        score_col=args.score_col,
        target_col=args.target_col,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        require_min_cross_section=args.min_cross_section,
        save_dir=args.save_dir,
        benchmark_csv=args.benchmark_csv,
    )
    res = run_backtest(cfg)
    print(json.dumps(res, indent=2))
