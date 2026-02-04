from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.config.logging import get_logger

from .artifacts import export_feature_importance, export_learning_curve, save_model, write_json
from .config import TrainValidSplitConfig, build_lgb_params
from .dataset import add_date_rank_relevance, build_split_bundle
from .io import collect_feature_cols, extract_categorical_levels, filter_feature_cols, load_panel, resolve_target
from .types import TrainingPaths

logger = get_logger(__name__)


def time_train_valid_split(
    df: pd.DataFrame,
    *,
    valid_days: int,
    forward_gap_days: int,
    train_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Single chronological train/validation split.

    - Validation is the last `valid_days` unique dates.
    - Training ends `forward_gap_days` before validation starts.
    - If `train_days` is provided, training is limited to the last `train_days`
      unique dates ending at the computed train_end.
    """

    if df.empty:
        raise ValueError("Cannot split an empty dataframe")
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column")

    valid_days = int(valid_days)
    forward_gap_days = int(forward_gap_days)
    train_days_int: Optional[int]
    if train_days is None:
        train_days_int = None
    else:
        train_days_int = int(train_days)
    if valid_days <= 0:
        raise ValueError("valid_days must be > 0")
    if forward_gap_days < 0:
        raise ValueError("forward_gap_days must be >= 0")
    if train_days_int is not None and train_days_int <= 0:
        raise ValueError("train_days must be > 0 when provided")

    dates = pd.Index(pd.to_datetime(df["date"]).dt.normalize().unique()).sort_values()
    if len(dates) < (valid_days + forward_gap_days + 2):
        raise RuntimeError(
            f"Not enough unique dates for split: n_dates={len(dates)} "
            f"need>= {valid_days + forward_gap_days + 2}"
        )

    valid_dates = dates[-valid_days:]
    valid_start = pd.Timestamp(valid_dates[0]).normalize()
    valid_end = pd.Timestamp(valid_dates[-1]).normalize()

    valid_start_idx = int(dates.get_indexer([valid_start])[0])
    train_end_idx = valid_start_idx - forward_gap_days - 1
    if train_end_idx < 0:
        raise RuntimeError("forward_gap_days too large for available history")

    train_end = pd.Timestamp(dates[train_end_idx]).normalize()
    if train_days_int is None:
        train_start = pd.Timestamp(dates[0]).normalize()
    else:
        train_start_idx = max(0, train_end_idx - (train_days_int - 1))
        train_start = pd.Timestamp(dates[train_start_idx]).normalize()

    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError("Split produced empty train or validation set")

    meta: Dict[str, Any] = {
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "valid_start": str(valid_start.date()),
        "valid_end": str(valid_end.date()),
        "valid_days": int(valid_days),
        "forward_gap_days": int(forward_gap_days),
        "train_days": (int(train_days_int) if train_days_int is not None else None),
        "n_dates": int(len(dates)),
    }

    logger.info(
        "Split snapshot dates: train_start=%s train_end=%s valid_start=%s valid_end=%s (train_days=%s valid_days=%d gap=%d)",
        meta["train_start"],
        meta["train_end"],
        meta["valid_start"],
        meta["valid_end"],
        str(meta.get("train_days")),
        int(valid_days),
        int(forward_gap_days),
    )

    return train_df, valid_df, meta


def _build_lgb_datasets(
    bundle,
    *,
    dataset_params: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.Dataset, lgb.Dataset]:
    lgb_train = lgb.Dataset(
        bundle.X_train,
        label=bundle.y_rel_train.to_numpy(),
        group=bundle.group_train,
        weight=(bundle.train_weights if bundle.train_weights is not None else None),
        free_raw_data=False,
        params=(dataset_params or None),
    )
    lgb_valid = lgb.Dataset(
        bundle.X_valid,
        label=bundle.y_rel_valid.to_numpy(),
        group=bundle.group_valid,
        reference=lgb_train,
        free_raw_data=False,
        params=(dataset_params or None),
    )
    return lgb_train, lgb_valid


def train_ranker(
    bundle,
    *,
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    log_evaluation_period: Optional[int] = None,
) -> Tuple[lgb.Booster, Dict[str, Dict[str, List[float]]]]:
    ds_params = {"max_bin": int(params["max_bin"])} if params.get("max_bin") is not None else None
    lgb_train, lgb_valid = _build_lgb_datasets(bundle, dataset_params=ds_params)

    evals: Dict[str, Dict[str, List[float]]] = {}
    callbacks = [
        lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True, verbose=False),
        lgb.record_evaluation(evals),
    ]

    try:
        period = int(log_evaluation_period) if log_evaluation_period is not None else 0
    except Exception:
        period = 0
    if period and period > 0:
        callbacks.insert(0, lgb.log_evaluation(period=period))

    booster = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid, lgb_train],
        valid_names=["valid", "train"],
        num_boost_round=int(num_boost_round),
        callbacks=callbacks,
    )

    return booster, evals


def _best_iter_from_evals(evals: Dict[str, Dict[str, List[float]]], *, primary_k: int) -> int:
    try:
        valid_metrics = evals.get("valid") or {}
        if not valid_metrics:
            return -1
        want = f"ndcg@{int(primary_k)}"
        metric_key = want if want in valid_metrics else next(iter(valid_metrics.keys()))
        curve = valid_metrics.get(metric_key)
        if not isinstance(curve, list) or not curve:
            return -1
        arr = np.asarray(curve, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            return -1
        return int(np.nanargmax(arr)) + 1
    except Exception:
        return -1


def train_single_split(
    *,
    cfg: TrainValidSplitConfig,
    paths: Optional[TrainingPaths] = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 40,
    primary_k: Optional[int] = None,
) -> Dict[str, Any]:
    p = paths or TrainingPaths()
    target_col = cfg.target_col or resolve_target(p)

    logger.info(
        "Train/valid split training: target_col=%s train_days=%s valid_days=%d forward_gap_days=%d min_cross_section=%d feature_scaling=%s",
        str(target_col),
        str(cfg.train_days),
        int(cfg.valid_days),
        int(cfg.forward_gap_days),
        int(cfg.min_cross_section),
        str(cfg.feature_scaling),
    )

    df = load_panel(p, target_col=str(target_col), min_cross_section=int(cfg.min_cross_section), target_clip=float(cfg.target_clip))

    # Relevance labels (safe: per-date cross-sectional ranking only)
    df = add_date_rank_relevance(
        df,
        target_col=str(target_col),
        n_bins=int(cfg.n_relevance_bins),
        force_negatives_to_zero=bool(cfg.force_negatives_to_zero),
        out_col="relevance",
    )

    feature_cols = collect_feature_cols(df, target_col=str(target_col))
    if not bool(cfg.include_sector_industry):
        feature_cols = [c for c in feature_cols if c not in {"sector", "industry"}]

    feature_cols, filt_stats = filter_feature_cols(
        df,
        feature_cols,
        drop_constant=bool(cfg.drop_constant_features),
        max_missing_frac=float(cfg.max_missing_frac),
        min_unique_values=int(cfg.min_unique_values),
    )

    cat_levels = {k: v for k, v in extract_categorical_levels(df).items() if k in set(feature_cols)}

    # Chronological split
    train_df, valid_df, split_meta = time_train_valid_split(
        df,
        valid_days=int(cfg.valid_days),
        forward_gap_days=int(cfg.forward_gap_days),
        train_days=(int(cfg.train_days) if cfg.train_days is not None else None),
    )

    train_end = pd.to_datetime(split_meta["train_end"]).normalize()

    bundle = build_split_bundle(
        train_df,
        valid_df,
        cfg=cfg,
        target_col=str(target_col),
        feature_cols=list(feature_cols),
        train_end_date=pd.Timestamp(train_end),
    )

    # No train-time preprocessing.

    lgb_params = build_lgb_params(cfg, primary_k=primary_k)

    primary_eval_k = int((lgb_params.get("eval_at") or [200])[0])
    booster, evals = train_ranker(
        bundle,
        params=lgb_params,
        num_boost_round=int(num_boost_round),
        early_stopping_rounds=int(early_stopping_rounds),
        log_evaluation_period=int(cfg.log_evaluation_period or 0),
    )

    best_iter = _best_iter_from_evals(evals, primary_k=int(primary_eval_k))
    if best_iter <= 0:
        best_iter = int(getattr(booster, "best_iteration", -1) or -1)

    metric_key = f"ndcg@{int(primary_eval_k)}"
    valid_curve = (evals.get("valid") or {}).get(metric_key)
    if not isinstance(valid_curve, list) or not valid_curve:
        # fall back to first valid metric
        vm = evals.get("valid") or {}
        metric_key = str(next(iter(vm.keys()))) if vm else metric_key
        valid_curve = (evals.get("valid") or {}).get(metric_key) if vm else None

    valid_ndcg = float("nan")
    if isinstance(valid_curve, list) and valid_curve:
        idx = max(0, min(int(best_iter) - 1, len(valid_curve) - 1))
        valid_ndcg = float(valid_curve[idx])

    # Persist artifacts
    save_model(p, booster)
    export_feature_importance(p, booster)
    export_learning_curve(p, evals, best_iteration=int(best_iter) if int(best_iter) > 0 else None)

    # Runtime-critical metadata for predict/backtest
    features_payload = {
        "features": list(feature_cols),
        "target": str(target_col),
        "categorical_levels": cat_levels,
    }
    write_json(p.feature_meta_path, features_payload)

    write_json(p.config_path, cfg.to_json_dict())

    metrics_payload: Dict[str, Any] = {
        "best_iteration": int(best_iter),
        "primary_k": int(primary_eval_k),
        f"valid_{metric_key}": valid_ndcg,
        "split": split_meta,
        "feature_filter": filt_stats,
        "score_sign": 1.0,
    }
    write_json(p.metrics_path, metrics_payload)

    logger.info("Training done: best_iteration=%s valid_%s=%.6f", str(best_iter), str(metric_key), float(valid_ndcg))

    return metrics_payload
