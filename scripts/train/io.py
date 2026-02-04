from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd

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


def load_panel(paths: TrainingPaths, *, target_col: str, min_cross_section: int, target_clip: float) -> pd.DataFrame:
    if not paths.raw_panel_path.exists():
        raise FileNotFoundError(f"Missing {paths.raw_panel_path}")

    # If features.json exists with a feature list, try to read only those columns.
    cols: List[str] | None = None
    if paths.feature_meta_path.exists():
        try:
            raw = json.loads(paths.feature_meta_path.read_text(encoding="utf-8"))
            feats = raw.get("features") if isinstance(raw, dict) else None
            if isinstance(feats, list) and feats:
                base = ["ticker", "date", target_col, "sector", "industry"]
                cols = base + [str(c) for c in feats if isinstance(c, str) and c not in base]
        except Exception:
            cols = None

    try:
        df = pd.read_parquet(paths.raw_panel_path, columns=cols) if cols else pd.read_parquet(paths.raw_panel_path)
    except Exception:
        df = pd.read_parquet(paths.raw_panel_path)

    if df.empty:
        raise RuntimeError("Panel features empty")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in panel. Available columns include: {list(df.columns)[:20]}"
        )

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

    before_drop = int(len(df))
    df = df.dropna(subset=[target_col], how="all")
    logger.debug("Dropped rows with missing target: %d -> %d", int(before_drop), int(len(df)))

    for c in ("sector", "industry"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    if target_clip and float(target_clip) > 0:
        df[target_col] = df[target_col].clip(-float(target_clip), float(target_clip))

    counts = df.groupby("date").size()
    keep_dates = counts[counts >= int(min_cross_section)].index
    df = df[df["date"].isin(keep_dates)]
    if df.empty:
        raise RuntimeError("No dates pass min_cross_section")

    return df


def collect_feature_cols(df: pd.DataFrame, *, target_col: str) -> List[str]:
    exclude = {"ticker", "date", target_col, "relevance", "split"}
    return [c for c in df.columns if c not in exclude]


def filter_feature_cols(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    drop_constant: bool,
    max_missing_frac: float,
    min_unique_values: int,
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
