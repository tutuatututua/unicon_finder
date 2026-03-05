from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from scripts.config.logging import get_logger

from .types import TrainingPaths

logger = get_logger(__name__)


def resolve_target(paths: TrainingPaths, *, default: str = "target_fwd_252d") -> str:
    """Resolve the target column name.

    Priority:
    1) models/features.json: {"target": "..."}
    2) provided default
    """

    try:
        if paths.feature_meta_path.exists():
            raw = json.loads(paths.feature_meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                t = raw.get("target")
                if isinstance(t, str) and t.strip():
                    return t.strip()
    except Exception as e:
        logger.debug("Target resolution failed; using default: %s", e)
    return default


def load_panel(
    paths: TrainingPaths,
    *,
    target_col: str,
    min_cross_section: int,
    target_clip: float,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    keep_dates: Optional[pd.Index] = None,
) -> pd.DataFrame:
    if not paths.raw_panel_path.exists():
        raise FileNotFoundError(f"Missing {paths.raw_panel_path}")

    # If features.json exists with a feature list, try to read only those columns.
    cols: List[str] | None = None
    if paths.feature_meta_path.exists():
        raw = json.loads(paths.feature_meta_path.read_text(encoding="utf-8"))
        feats = raw.get("features") if isinstance(raw, dict) else None
        if isinstance(feats, list) and feats:
            # Keep the panel read as narrow as possible.
            # Always attempt to include sector/industry as optional categoricals.
            # If the parquet doesn't have them, schema validation will drop them.
            base = ["ticker", "date", target_col, "sector", "industry"]
            helper_cols = {"chunk_id", "relevance", "split"}

            # Preserve order, de-duplicate, and never request helper columns from parquet.
            cols = list(base)
            seen = set(cols)
            for c in feats:
                if not isinstance(c, str):
                    continue
                c = str(c)
                if c in helper_cols:
                    continue
                if c in seen:
                    continue
                cols.append(c)
                seen.add(c)

            # Validate against parquet schema so we fail with a clear message.
            try:
                schema_cols = set(pq.read_schema(paths.raw_panel_path).names)
            except Exception as e:
                logger.debug("Could not read parquet schema for %s: %s", str(paths.raw_panel_path), str(e))
                schema_cols = set()

            if schema_cols:
                missing = [c for c in cols if c not in schema_cols]
                if missing:
                    optional = {"sector", "industry"}
                    missing_required = [c for c in missing if c not in optional]
                    if missing_required:
                        # Helper columns are already filtered above; if anything else is missing,
                        # it's likely a stale/incompatible models/features.json.
                        raise KeyError(
                            "Panel parquet is missing columns referenced by models/features.json: "
                            f"{missing_required}. Either rebuild features to match this panel, or delete "
                            f"{paths.feature_meta_path} so all columns are loaded."
                        )

                    # Only optional columns missing; drop them quietly.
                    logger.warning(
                        "Panel parquet missing optional columns %s; continuing without them.",
                        missing,
                    )
                    cols = [c for c in cols if c in schema_cols]

    parquet_filters: list[tuple[str, str, Any]] = []
    if min_date is not None:
        parquet_filters.append(("date", ">=", pd.Timestamp(min_date).to_datetime64()))
    if max_date is not None:
        parquet_filters.append(("date", "<=", pd.Timestamp(max_date).to_datetime64()))

    read_kwargs: dict[str, Any] = {"engine": "pyarrow"}
    if parquet_filters:
        read_kwargs["filters"] = parquet_filters

    df = (
        pd.read_parquet(paths.raw_panel_path, columns=cols, **read_kwargs)
        if cols
        else pd.read_parquet(paths.raw_panel_path, **read_kwargs)
    )

    if df.empty:
        raise RuntimeError("Panel features empty")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in panel. Available columns include: {list(df.columns)[:20]}"
        )

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

    # Faster + lower-memory than DataFrame.dropna(subset=[...]) on very large panels.
    # `dropna` can materialize a large boolean mask across the full block manager.
    before_drop = int(len(df))
    mask = df[str(target_col)].notna()
    if not bool(mask.all()):
        df = df.loc[mask]
    logger.debug("Dropped rows with missing target: %d -> %d", int(before_drop), int(len(df)))

    for c in ("sector", "industry"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    if target_clip and float(target_clip) > 0:
        df[target_col] = df[target_col].clip(-float(target_clip), float(target_clip))

    if keep_dates is None:
        counts = df.groupby("date").size()
        keep_dates = counts[counts >= int(min_cross_section)].index

    df = df[df["date"].isin(pd.Index(keep_dates))]
    if df.empty:
        raise RuntimeError("No dates pass min_cross_section")

    return df


def load_usable_dates(
    paths: TrainingPaths,
    *,
    target_col: str,
    min_cross_section: int,
) -> pd.Index:
    """Return sorted dates that have enough cross-section and non-missing targets.

    This intentionally reads only `date` + `target_col` from the parquet so callers can
    plan walk-forward splits without loading the full feature matrix.
    """

    if not paths.raw_panel_path.exists():
        raise FileNotFoundError(f"Missing {paths.raw_panel_path}")

    cols = ["date", str(target_col)]
    try:
        df = pd.read_parquet(paths.raw_panel_path, columns=cols)
    except Exception as e:
        # Provide a clearer message when the target column doesn't exist.
        try:
            schema_cols = set(pq.read_schema(paths.raw_panel_path).names)
        except Exception:
            schema_cols = set()
        if schema_cols and str(target_col) not in schema_cols:
            raise KeyError(
                f"Target column '{target_col}' not found in panel parquet."
            ) from e
        raise

    if df.empty:
        return pd.Index([], dtype="datetime64[ns]")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    mask = df[str(target_col)].notna()
    if not bool(mask.all()):
        df = df.loc[mask]

    if df.empty:
        return pd.Index([], dtype="datetime64[ns]")

    counts = df.groupby("date").size()
    keep = counts[counts >= int(min_cross_section)].index
    return pd.Index(pd.to_datetime(keep).sort_values().unique())


def collect_feature_cols(df: pd.DataFrame, *, target_col: str) -> List[str]:
    # Training/pipeline helper columns should never become model features.
    exclude = {"ticker", "date", target_col, "relevance", "split", "chunk_id"}
    return [c for c in df.columns if c not in exclude]


def filter_feature_cols(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    drop_constant: bool = True,
    max_missing_frac: float = 0.98,
    min_unique_values: int = 2,
) -> Tuple[List[str], Dict[str, int]]:
    keep: List[str] = []
    dropped_constant = 0
    dropped_missing = 0
    dropped_unique = 0

    max_missing_frac = float(max_missing_frac)
    min_unique_values = int(min_unique_values)

    for c in feature_cols:
        if c not in df.columns:
            continue

        if c in {"sector", "industry"}:
            keep.append(c)
            continue

        s = df[c]
        miss = float(s.isna().mean()) if len(s) else 1.0
        if max_missing_frac < 1.0 and miss >= max_missing_frac:
            dropped_missing += 1
            continue

        try:
            nun = int(s.nunique(dropna=True))
        except Exception:
            nun = 0

        if nun < min_unique_values:
            dropped_unique += 1
            if drop_constant:
                dropped_constant += 1
            continue

        if drop_constant and nun <= 1:
            dropped_constant += 1
            continue

        keep.append(c)

    stats = {
        "input": int(len(feature_cols)),
        "kept": int(len(keep)),
        "dropped_missing": int(dropped_missing),
        "dropped_unique": int(dropped_unique),
        "dropped_constant": int(dropped_constant),
    }
    return keep, stats


def extract_categorical_levels(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in ("sector", "industry"):
        if c in df.columns and hasattr(df[c], "cat"):
            out[c] = list(map(str, list(df[c].cat.categories)))
    return out
