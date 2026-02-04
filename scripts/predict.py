from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.config.logging import get_logger
from scripts.feature.metadata import load_features_meta, resolve_target_column

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
    feature_meta: Path = Path("models/features.json")
    metrics: Path = Path("models/metrics.json")


PATHS = Paths()


# --------------------------------------------------------------------------------------
# Minimal helpers (kept only where it meaningfully reduces duplication)
# --------------------------------------------------------------------------------------

def _select_snapshot(snapshot: str | Snapshot) -> Path:
    snap = Snapshot(snapshot) if not isinstance(snapshot, Snapshot) else snapshot
    return PATHS.scoring_features if snap is Snapshot.LATEST else PATHS.last_labeled_features


def _load_score_sign(metrics_path: Path) -> float:
    """Load score direction from training metrics.

    Returns +1.0 if missing/invalid. If -1.0, downstream should multiply raw
    model scores by -1 so higher is better (higher expected target).
    """

    if not metrics_path.exists():
        return 1.0
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        s = float(payload.get("score_sign", 1.0))
        if s == 0 or not np.isfinite(s):
            return 1.0
        return 1.0 if s > 0 else -1.0
    except Exception:
        return 1.0


def _load_best_iteration(metrics_path: Path) -> Optional[int]:
    """Load early-stopping best iteration from training metrics.

    Note: LightGBM's Booster.best_iteration is not reliably preserved after
    saving/loading the model. We persist it in models/metrics.json.

    Returns None if missing/invalid, meaning "use all trees".
    """

    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        bi = payload.get("best_iteration")
        bi_int = int(bi)
        return bi_int if bi_int > 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def predict(
    force: bool = True,
    top_n: int | None = None,
    snapshot: str = "latest",
    save_per_snapshot: bool = False,
    date: Optional[str] = None,
    paths: Optional[Paths] = None,
) -> pd.DataFrame:
    """Generate ranked forward 252-day return predictions (no fallbacks).

    If `date` (YYYY-MM-DD) is provided, predictions are computed for that
    specific cross-section by loading the full historical feature panel and
    filtering to the requested date. Otherwise, a prebuilt snapshot is used.
    """
    p = paths or PATHS

    # Resolve target column dynamically (avoid stale import-time config).
    target_col = resolve_target_column(p.feature_meta, default="target_fwd_252d")

    # Resolve and load features
    if date:
        # Parse and normalize date
        dt = pd.to_datetime(date, utc=True, errors="raise").tz_localize(None).normalize()
        if not p.all_features.exists():
            raise FileNotFoundError(f"Missing {p.all_features}. Build features first.")
        feats = pd.read_parquet(p.all_features)
        feats["date"] = pd.to_datetime(feats["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        feats = feats.loc[feats["date"] == dt].copy()
        if feats.empty:
            raise ValueError(f"No feature rows available for date {dt.date()}.")
        snapshot = f"on_{dt.date()}"
    else:
        # _select_snapshot uses module-level PATHS; replicate behavior for custom Paths.
        src_path = p.scoring_features if Snapshot(snapshot) is Snapshot.LATEST else p.last_labeled_features
        feats = pd.read_parquet(src_path)

    # Load model
    rank_booster: Optional[lgb.Booster] = lgb.Booster(model_file=str(p.rank_model))
    model_features = list(rank_booster.feature_name())
    if not model_features:
        raise RuntimeError("Loaded model has no feature names.")

    score_sign = _load_score_sign(p.metrics)

    # Align categorical levels with training (if available in models/features.json)
    meta = load_features_meta(p.feature_meta)
    cat_levels = dict(meta.categorical_levels)

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
    # Prefer metrics.json, since Booster.best_iteration is not stable after reload.
    best_iter_int = _load_best_iteration(p.metrics)

    # If no early stopping was used, LightGBM can report best_iteration as -1/0.
    # Treat that as "use all trees".
    rank_score = np.asarray(
        rank_booster.predict(X, num_iteration=(best_iter_int if best_iter_int and best_iter_int > 0 else None))
    )
    if score_sign < 0:
        rank_score = -rank_score

    # Assemble & sort
    base_cols = [c for c in ["ticker", "date", "sector", "industry"] if c in feats.columns]
    out = feats[base_cols].copy()
    out["rank_score"] = rank_score
    out.sort_values("rank_score", ascending=False, inplace=True)
    out = out.reset_index(drop=True)

    # Save
    save_path = p.predictions if not save_per_snapshot else p.predictions.with_name(
        f"{p.predictions.stem}_{snapshot}.csv"
    )
    out.to_csv(save_path, index=False)
    logger.info("Saved predictions to %s (rows=%d)", save_path, len(out))

    result = out
    if snapshot == Snapshot.LAST_LABELED.value and target_col in feats.columns:
        joined = out.merge(feats[["ticker", "date", target_col]], on=["ticker", "date"], how="left")
        joined.rename(columns={target_col: "true_forward_return"}, inplace=True)
        valid = joined.dropna(subset=["true_forward_return", "rank_score"])
        if len(valid) >= 3:
            # Spearman is undefined (and warns) if either input is constant.
            if valid["rank_score"].nunique(dropna=True) > 1 and valid["true_forward_return"].nunique(dropna=True) > 1:
                rho = float(valid["rank_score"].corr(valid["true_forward_return"], method="spearman"))
                logger.info("Spearman(rank_score, target): %.4f", rho)
        result = joined

    # Return top-N view if requested (keep full CSV saved).
    if top_n is not None and top_n > 0:
        return result.head(int(top_n)).reset_index(drop=True)
    return result


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
