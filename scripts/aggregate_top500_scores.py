"""Aggregate per-ticker total score across days they appear in the daily Top-N by model score.

Outputs two files under backtest/:
- topN_per_day.parquet: per-day Top-N with rank and score
- topN_sum_scores.csv: aggregated totals per ticker to identify overall "best" stocks

Notes
- Uses the trained LightGBM ranking model at models/lightgbm_model.txt
- Scores a feature panel (default: data/processed/raw_training.parquet)
- By default, includes the most recent dates even if the target column is missing (unlabeled).
    NDCG is calculated only on labeled dates.

CLI
    python scripts/aggregate_top500_scores.py --top-n 500 --min-cross-section 100
"""
from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

RAW_FEATURES_PATH = Path("data/processed/raw_training.parquet")
FEATURE_META = Path("models/features.json")
MODEL_PATH = Path("models/lightgbm_model.txt")
OUT_DIR = Path("backtest")
SECTOR_MAP_CSV = Path("data/sector_map.csv")

# Ensure repository root is importable for `import backtest`
try:
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
except Exception:
    pass


def _read_target_from_meta(meta_path: Path) -> str:
    if not meta_path.exists():
        raise FileNotFoundError(f"Feature meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target = str(meta.get("target") or "").strip()
    if not target:
        raise ValueError("'target' missing in features meta")
    return target


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("Features frame missing required 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _maybe_attach_sector_industry(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure sector/industry columns exist and fill missing values from sector_map.csv.

    Previously, we only merged when "all" values were NA, which left partially missing
    labels unfixed. Now we merge if the column is missing or has any NA values, and we
    prefer any existing non-null values from df.
    """
    sector_na_any = True if ("sector" not in df.columns) else df["sector"].isna().any()
    industry_na_any = True if ("industry" not in df.columns) else df["industry"].isna().any()
    need_sector = ("sector" not in df.columns) or sector_na_any
    need_industry = ("industry" not in df.columns) or industry_na_any
    if not (need_sector or need_industry):
        return df
    if not SECTOR_MAP_CSV.exists():
        return df
    try:
        sm = pd.read_csv(SECTOR_MAP_CSV)
        if "ticker" not in sm.columns:
            return df
        keep = [c for c in ["ticker", "sector", "industry"] if c in sm.columns]
        sm = sm[keep].copy()
        sm["ticker"] = sm["ticker"].astype(str).str.upper()
        df2 = df.copy()
        df2["ticker"] = df2["ticker"].astype(str).str.upper()
        df2 = df2.merge(sm, on="ticker", how="left", suffixes=("", "_sm"))
        # Prefer existing non-null columns; otherwise fill from _sm
        for c in ("sector", "industry"):
            col_sm = f"{c}_sm"
            if c not in df2.columns and col_sm in df2.columns:
                df2[c] = df2[col_sm]
            elif col_sm in df2.columns:
                df2[c] = df2[c].where(df2[c].notna(), df2[col_sm])
            df2.drop(columns=[col_sm], errors="ignore", inplace=True)
        return df2
    except Exception:
        return df


def _score_with_model(df: pd.DataFrame, model_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    feats = list(booster.feature_name())
    if not feats:
        raise RuntimeError("Loaded model has no feature names")
    # Ensure categorical stability for sector/industry if present
    for c in ("sector", "industry"):
        if c in feats and c in df.columns:
            df[c] = pd.Categorical(df[c].astype("string"))
    X = df[feats]
    out = df.copy()
    out["score"] = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
    return out, feats


def aggregate_topN(
    top_n: int = 500,
    min_cross_section: int = 500,
    generate_ndcg: bool = False,
    last_days: Optional[int] = 21,
    ndcg_full_window: bool = False,
    features_path: Path = RAW_FEATURES_PATH,
    labeled_only: bool = False,
) -> dict:
    target_col = _read_target_from_meta(FEATURE_META)
    df_all = _load_features(features_path)

    # Choose last N dates from the full panel (including unlabeled) so we can include latest data
    uniq_all = np.sort(df_all["date"].unique())
    if last_days is not None and last_days > 0:
        keep_dates = set(uniq_all[-last_days:])
        df_win = df_all[df_all["date"].isin(keep_dates)].copy()
    else:
        df_win = df_all.copy()

    # Optionally restrict rows to labeled only; otherwise keep unlabeled too
    if labeled_only and target_col in df_win.columns:
        df = df_win[df_win[target_col].notna()].copy()
    else:
        df = df_win.copy()
    # Ensure sector/industry availability if possible
    df = _maybe_attach_sector_industry(df)

    df_scored, used_feats = _score_with_model(df, MODEL_PATH)

    # Build per-day Top-N table with ranks
    top_rows = []
    for dt, g in df_scored.groupby("date", sort=True):
        ge = g.dropna(subset=["score"])  # score should be present; allow missing target
        if len(ge) < max(min_cross_section, min(top_n, 1)):
            continue
        ge = ge.sort_values("score", ascending=False)
        take = min(top_n, len(ge))
        ge = ge.head(take).reset_index(drop=True)
        ge["rank"] = np.arange(1, len(ge) + 1, dtype=int)
        cols = [c for c in ["ticker", "sector", "industry", target_col] if c in ge.columns]
        ge = ge[["date", "rank", "score"] + cols]
        top_rows.append(ge)
    if not top_rows:
        raise RuntimeError("No dates produced Top-N selection; check cross-section size or inputs.")
    top_per_day = pd.concat(top_rows, ignore_index=True)

    # If last_days is provided but earlier filtering had some dates skipped by min_cross_section,
    # ensure we still end up with at most last_days dates in the per-day selection.
    if last_days is not None and last_days > 0:
        # Work on the produced per-day table (post cross-section checks)
        dts = np.sort(top_per_day["date"].unique())
        if len(dts) > last_days:
            keep = set(dts[-last_days:])
            top_per_day = top_per_day[top_per_day["date"].isin(keep)].copy()

    # Helpers to pick a stable sector/industry label per ticker
    def _mode_or_first(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    # Aggregate numeric metrics first
    agg = (
        top_per_day
        .groupby("ticker", as_index=False)
        .agg(
            total_score=("score", "sum"),
            days_in_topN=("date", "count"),
            avg_rank=("rank", "mean"),
            rank_var=("rank", lambda s: float(np.var(s.to_numpy(), ddof=0)))  # population variance of daily ranks
        )
    )

    # Then attach sector/industry via separate groupbys to avoid type issues
    if "sector" in top_per_day.columns:
        si = (
            top_per_day.dropna(subset=["sector"])
            .groupby("ticker", as_index=False)["sector"].agg(_mode_or_first)
        )
        agg = agg.merge(si, on="ticker", how="left")
    if "industry" in top_per_day.columns:
        ii = (
            top_per_day.dropna(subset=["industry"])
            .groupby("ticker", as_index=False)["industry"].agg(_mode_or_first)
        )
        agg = agg.merge(ii, on="ticker", how="left")

    # Reorder columns for readability and sort by strength
    cols_order = [c for c in ["ticker", "sector", "industry", "total_score", "days_in_topN", "avg_rank", "rank_var"] if c in agg.columns]
    agg = agg[cols_order].sort_values(["total_score", "days_in_topN"], ascending=[False, False]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_day_path_parquet = OUT_DIR / f"top{top_n}_per_day.parquet"
    agg_path = OUT_DIR / f"top{top_n}_sum_scores.csv"

    # Save with stable formatting (parquet for large file) 
    top_per_day.to_parquet(per_day_path_parquet, index=False)
    agg.to_csv(agg_path, index=False)

    result = {
        "dates_processed": int(top_per_day["date"].nunique()),
        "tickers_ranked": int(agg.shape[0]),
        "per_day_parquet": str(per_day_path_parquet),
        "aggregate_csv": str(agg_path),
        "used_features": used_feats,
    }
    if last_days is not None and last_days > 0:
        result["last_days"] = int(last_days)

    # Optional: trigger NDCG generation using aggregated scores
    if generate_ndcg:
        try:
            from backtest import backtest_from_aggregated_scores
            # Evaluate only on labeled dates within the produced per-day window
            produced_dates = set(pd.to_datetime(top_per_day["date"]).dt.normalize().unique())
            labeled_dates = set()
            if target_col in df_all.columns:
                labeled_dates = set(pd.to_datetime(df_all.loc[df_all[target_col].notna(), "date"]).dt.normalize().unique())
            eval_dates = sorted(list(produced_dates & labeled_dates))
            nd_res = backtest_from_aggregated_scores(
                agg_path,
                dates=None if ndcg_full_window else list(eval_dates),
                output_stem=f"backtest_ndcg_sum{top_n}",
            )
            result.update({
                "ndcg_csv": nd_res.get("ndcg_csv"),
                "ndcg_xlsx": nd_res.get("ndcg_xlsx"),
            })
        except Exception as e:
            result["ndcg_error"] = str(e)

    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Top-N model scores across dates")
    ap.add_argument("--top-n", type=int, default=500, help="Select this many top names per day by score")
    ap.add_argument("--min-cross-section", type=int, default=200, help="Skip dates with fewer names than this")
    ap.add_argument("--generate-ndcg", action="store_true", help="After saving parquet, generate backtest_ndcg.xlsx via backtest.py")
    ap.add_argument("--last-days", type=int, default=10, help="Restrict aggregation window to the last N unique dates")
    ap.add_argument("--ndcg-full-window", action="store_true", help="When generating NDCG, evaluate across the full labeled window instead of only the aggregation dates")
    ap.add_argument("--features-path", type=str, default=str(RAW_FEATURES_PATH), help="Path to features parquet to score")
    ap.add_argument("--labeled-only", action="store_true", help="Restrict aggregation to labeled rows/dates only")
    args = ap.parse_args()
    res = aggregate_topN(
        top_n=args.top_n,
        min_cross_section=args.min_cross_section,
        generate_ndcg=args.generate_ndcg,
        last_days=args.last_days,
        ndcg_full_window=args.ndcg_full_window,
        features_path=Path(args.features_path),
        labeled_only=args.labeled_only,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli()
