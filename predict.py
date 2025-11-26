"""Simple prediction script (ranking-only).

Straight, minimal flow:
 - pick snapshot -> load features
 - load trained LightGBM ranking model
 - align columns by model.feature_name()
 - predict and sort
 - save (optionally per snapshot)

Notes:
 - No error fallbacks; let exceptions propagate.
 - Keep public API predict(force, top_n, snapshot, save_per_snapshot) for pipeline compatibility.
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from logging_config import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Configuration / Paths
# --------------------------------------------------------------------------------------

class Snapshot(str, Enum):
    """Feature snapshot selection options."""
    LATEST = "latest"
    LAST_LABELED = "last_labeled"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations for required artifacts."""
    scoring_features: Path = Path("data/processed/latest_data.parquet")
    # Aligned with features.build_features() which writes extract_training.parquet
    last_labeled_features: Path = Path("data/processed/extract_training.parquet")
    # Full historical feature panel (all dates, labeled + unlabeled)
    all_features: Path = Path("data/processed/raw_training.parquet")
    rank_model: Path = Path("models/lightgbm_model.txt")
    predictions: Path = Path("models/predictions.csv")


PATHS = Paths()
TARGET_COL = "target_fwd_252d"


# --------------------------------------------------------------------------------------
# Minimal helpers (kept only where it meaningfully reduces duplication)
# --------------------------------------------------------------------------------------

def _select_snapshot(snapshot: str | Snapshot) -> Path:
    snap = Snapshot(snapshot) if not isinstance(snapshot, Snapshot) else snapshot
    return PATHS.scoring_features if snap is Snapshot.LATEST else PATHS.last_labeled_features


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def predict(
    force: bool = True,
    top_n: int | None = None,
    snapshot: str = "latest",
    save_per_snapshot: bool = False,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """Generate ranked forward 252-day return predictions (no fallbacks).

    If `date` (YYYY-MM-DD) is provided, predictions are computed for that
    specific cross-section by loading the full historical feature panel and
    filtering to the requested date. Otherwise, a prebuilt snapshot is used.
    """
    # Resolve and load features
    if date:
        # Parse and normalize date
        dt = pd.to_datetime(date, utc=True, errors="raise").tz_localize(None).normalize()
        if not PATHS.all_features.exists():
            raise FileNotFoundError(f"Missing {PATHS.all_features}. Build features first.")
        feats = pd.read_parquet(PATHS.all_features)
        feats["date"] = pd.to_datetime(feats["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        feats = feats.loc[feats["date"] == dt].copy()
        if feats.empty:
            raise ValueError(f"No feature rows available for date {dt.date()}.")
        snapshot = f"on_{dt.date()}"
    else:
        src_path = _select_snapshot(snapshot)
        feats = pd.read_parquet(src_path)

    # Load model
    rank_booster: Optional[lgb.Booster] = lgb.Booster(model_file=str(PATHS.rank_model))
    model_features = list(rank_booster.feature_name())
    if not model_features:
        raise RuntimeError("Loaded model has no feature names.")

    # Align categorical levels with training (if available in models/features.json)
    cat_levels = {}
    try:
        meta_path = Path("models/features.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cat_levels = meta.get("categorical_levels", {}) or {}
    except Exception:
        cat_levels = {}

    # Apply categorical dtypes with fixed category sets for stable coding
    for c in ("sector", "industry"):
        if c in model_features and c in feats.columns:
            if c in cat_levels and isinstance(cat_levels[c], list) and len(cat_levels[c]) > 0:
                feats[c] = pd.Categorical(feats[c].astype("string"), categories=cat_levels[c])
            else:
                feats[c] = pd.Categorical(feats[c].astype("string"))

    # Prepare design matrix (strict selection; will raise if missing columns)
    X = feats[model_features]

    # Predict (use raw ranking score directly)
    rank_score = np.asarray(
        rank_booster.predict(X, num_iteration=getattr(rank_booster, "best_iteration", None))
    )

    # Assemble & sort
    base_cols = [c for c in ["ticker", "date", "sector", "industry"] if c in feats.columns]
    out = feats[base_cols].copy()
    out["rank_score"] = rank_score
    out.sort_values("rank_score", ascending=False, inplace=True)
    out = out.reset_index(drop=True)

    # Save
    save_path = PATHS.predictions if not save_per_snapshot else PATHS.predictions.with_name(
        f"{PATHS.predictions.stem}_{snapshot}.csv"
    )
    out.to_csv(save_path, index=False)
    logger.info("Saved predictions to %s (rows=%d)", save_path, len(out))

    # Simple optional evaluation for last_labeled (will error if target missing)
    if snapshot == Snapshot.LAST_LABELED.value:
        joined = out.merge(feats[["ticker", "date", TARGET_COL]], on=["ticker", "date"], how="left")
        joined.rename(columns={TARGET_COL: "true_fwd_252d_return"}, inplace=True)
        joined["abs_error"] = (joined["rank_score"] - joined["true_fwd_252d_return"]).abs()
        logger.info("MAE (last_labeled): %.4f", float(joined["abs_error"].mean()))
        return joined
    return out


# --------------------------------------------------------------------------------------
# CLI Entrypoint
# --------------------------------------------------------------------------------------
def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate predictions using trained ranking model.")
    parser.add_argument("--top-n", type=int, default=20, help="Limit output to top N rows (default 20; 0=all).")
    parser.add_argument("--snapshot", choices=[s.value for s in Snapshot], default=Snapshot.LATEST.value, help="Which snapshot to score.")
    parser.add_argument("--save-per-snapshot", action="store_true", help="Save predictions to snapshot-specific file name.")
    parser.add_argument("--date", type=str, default=None, help="Predict for a specific YYYY-MM-DD date (overrides --snapshot).")
    args = parser.parse_args()
    df = predict(
        top_n=None if args.top_n == 0 else args.top_n,
        snapshot=args.snapshot,
        save_per_snapshot=args.save_per_snapshot,
        date=args.date,
    )
    print(df.head(args.top_n or 20).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
