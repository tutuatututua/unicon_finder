from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.config.logging import get_logger

from .config import TrainValidSplitConfig
from .types import DatasetBundle

logger = get_logger(__name__)


def add_ticker_chunk_id(
    df: pd.DataFrame,
    *,
    chunk_size: int,
    out_col: str = "chunk_id",
) -> pd.DataFrame:
    """Assign each row to a per-date ticker chunk.

    Chunking is stable by ticker sort within each date.
    If chunk_size <= 0, returns df unchanged.
    """

    if df.empty:
        return df
    if out_col in df.columns:
        return df

    cs = int(chunk_size)
    if cs <= 0:
        return df
    if "date" not in df.columns:
        raise KeyError("Expected 'date' column for chunking")
    if "ticker" not in df.columns:
        raise KeyError("Expected 'ticker' column for chunking")

    df = df.copy()

    # Rank by ticker within each date (1..N), then make 0-based chunk ids.
    # Using rank(method='first') ensures a deterministic ordering.
    within_date_pos0 = (
        df.groupby("date", sort=False)["ticker"].rank(method="first", ascending=True).astype(int) - 1
    )
    df[out_col] = (within_date_pos0 // cs).astype(int)
    return df


def add_date_rank_relevance(
    df: pd.DataFrame,
    *,
    target_col: str,
    n_bins: int,
    force_negatives_to_zero: bool,
    group_cols: Optional[Sequence[str]] = None,
    out_col: str = "relevance",
) -> pd.DataFrame:
    if df.empty:
        return df
    if out_col in df.columns:
        return df

    df[out_col] = _date_rank_relevance(
        df,
        target_col=str(target_col),
        n_bins=int(n_bins),
        force_negatives_to_zero=bool(force_negatives_to_zero),
        group_cols=group_cols,
    )
    return df


def _date_rank_relevance(
    df: pd.DataFrame,
    *,
    target_col: str,
    n_bins: int,
    force_negatives_to_zero: bool,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)

    n_bins = max(2, int(n_bins))

    if group_cols is None:
        group_cols_eff: Sequence[str] = ("date",)
    else:
        group_cols_eff = tuple(str(c) for c in group_cols)
        if not group_cols_eff:
            group_cols_eff = ("date",)

    missing = [c for c in group_cols_eff if c not in df.columns]
    if missing:
        raise KeyError(f"Missing group columns for relevance ranking: {missing}")

    pct = df.groupby(list(group_cols_eff), sort=False)[target_col].rank(method="average", pct=True)
    rel = (np.floor(pct.to_numpy(dtype=float) * float(n_bins)).astype(int) - 1).clip(0, n_bins - 1)
    out = pd.Series(rel, index=df.index, dtype=int)

    if force_negatives_to_zero and target_col in df.columns:
        out.loc[df[target_col].astype(float) < 0.0] = 0
    return out


def build_split_bundle(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    cfg: TrainValidSplitConfig,
    target_col: str,
    feature_cols: List[str],
    train_end_date: pd.Timestamp,
) -> DatasetBundle:
    if "relevance" not in train_df.columns or "relevance" not in valid_df.columns:
        raise RuntimeError("Expected 'relevance' column to be present before dataset build")

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    # Optional query chunking: per-date ticker chunks.
    chunk_size = int(getattr(cfg, "ticker_chunk_size", 0) or 0)
    if chunk_size > 0:
        train_df = add_ticker_chunk_id(train_df, chunk_size=chunk_size, out_col="chunk_id")
        valid_df = add_ticker_chunk_id(valid_df, chunk_size=chunk_size, out_col="chunk_id")
        query_cols = [str(getattr(cfg, "group_col", "date")), "chunk_id"]
    else:
        query_cols = [str(getattr(cfg, "group_col", "date"))]

    # Sort so LightGBM groups are contiguous.
    sort_cols = [c for c in (*query_cols, "ticker") if c in train_df.columns]
    train_df = train_df.sort_values(sort_cols)
    valid_df = valid_df.sort_values(sort_cols)

    # LambdaRank needs at least 2 items per query/date and at least 2 distinct labels per query.
    def _filter_queries(df: pd.DataFrame) -> pd.DataFrame:
        sizes = df.groupby(query_cols, sort=False).size()
        keep_keys = sizes[sizes >= 2].index
        df = df.set_index(query_cols)
        df = df.loc[df.index.isin(keep_keys)].reset_index()

        nuniq = df.groupby(query_cols, sort=False)["relevance"].nunique(dropna=False)
        keep_keys = nuniq[nuniq >= 2].index
        df = df.set_index(query_cols)
        df = df.loc[df.index.isin(keep_keys)].reset_index()
        return df

    train_df = _filter_queries(train_df)
    valid_df = _filter_queries(valid_df)

    group_train = train_df.groupby(query_cols, sort=False).size().tolist()
    group_valid = valid_df.groupby(query_cols, sort=False).size().tolist()

    if not group_train or not group_valid:
        raise RuntimeError("Split produced empty ranking groups; reduce min_cross_section or valid_days")

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_valid = valid_df[feature_cols].reset_index(drop=True)

    y_rel_train = train_df["relevance"].astype(int).reset_index(drop=True)
    y_rel_valid = valid_df["relevance"].astype(int).reset_index(drop=True)

    y_cont_valid = valid_df[target_col].reset_index(drop=True)
    date_valid = valid_df["date"].reset_index(drop=True)

    # Optional recency weights for training only.
    train_weights: Optional[np.ndarray]
    if float(cfg.recency_lambda) > 0 and len(train_df):
        date_arr = train_df["date"].to_numpy(dtype="datetime64[ns]")
        age_td = (np.datetime64(pd.Timestamp(train_end_date)) - date_arr).astype("timedelta64[D]")
        age_days = age_td.astype(np.int64)
        age_years = age_days.astype(float) / 365.25
        train_weights = np.exp(-float(cfg.recency_lambda) * age_years)
    else:
        train_weights = None

    meta: Dict[str, Any] = {
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "n_groups_train": int(len(group_train)),
        "n_groups_valid": int(len(group_valid)),
        "avg_group_size_train": float(np.mean(group_train) if group_train else 0.0),
        "avg_group_size_valid": float(np.mean(group_valid) if group_valid else 0.0),
        "ticker_chunk_size": int(chunk_size),
    }

    return DatasetBundle(
        X_train=X_train,
        X_valid=X_valid,
        y_rel_train=y_rel_train,
        y_rel_valid=y_rel_valid,
        y_cont_valid=y_cont_valid,
        date_valid=date_valid,
        group_train=group_train,
        group_valid=group_valid,
        train_weights=train_weights,
        meta=meta,
    )
