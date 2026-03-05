from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainValidSplitConfig:
    """Configuration for a single time-based train/validation split."""

    # Data / schema
    target_col: Optional[str] = None  # resolved from models/features.json if None
    group_col: str = "date"  # query groups for ranking

    # Split + leakage controls
    valid_days: int = 121
    forward_gap_days: int = 5

    train_days: Optional[int] = 1 # if None, use all available before the validation period

    # Panel hygiene
    min_cross_section: int = 200
    target_clip: float = 30.0

    # Optional: split each date's cross-section into smaller ranking queries.
    # If > 0, each date is chunked into groups of at most this many tickers.
    # This can make learning easier by limiting query size (and aligns with max eval_at=50 by default).
    ticker_chunk_size: int = 200

    # Relevance labels (LambdaRank expects integer relevance)
    n_relevance_bins: int = 10
    force_negatives_to_zero: bool = False

    # Training
    seed: int = 42
    num_threads: int = 16 # deterministic by default
    recency_lambda: float = 0.0

    # Training progress logging. 0 disables periodic logs.
    log_evaluation_period: int = 50

    # LightGBM LambdaRank params (overrides defaults)
    lgb_params: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_lgb_params(*, seed: int, num_threads: int, eval_at: Optional[list[int]] = None) -> Dict[str, Any]:
    """Repo-default LightGBM params for LambdaRank.

    Kept intentionally fixed and centralized for fast iteration.
    """

    # Include the repo-standard NDCG cutoffs.
    # Note: ordering matters for early stopping (it uses the first cutoff).
    eval_at_list = eval_at if isinstance(eval_at, list) and eval_at else [10, 20, 50]

    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 300,
        "min_sum_hessian_in_leaf": 10.0,
        "min_gain_to_split": 0.05,
        "lambda_l1": 0.2,
        "lambda_l2": 10.0,
        "feature_fraction": 0.6,
        "feature_fraction_seed": int(seed),
        "bagging_fraction": 0.6,
        "bagging_freq": 1,
        "bagging_seed": int(seed),
        "max_bin": 255,

        "boosting": "gbdt",
        # Put the primary cutoff first (pipeline can reorder to match primary_k).
        "eval_at": eval_at_list,
        "first_metric_only": True,
        "seed": int(seed),
        "num_threads": int(num_threads),
        "force_col_wise": True,
        "verbose": -1,
    }


def build_lgb_params(cfg: TrainValidSplitConfig, *, primary_k: Optional[int] = None) -> Dict[str, Any]:
    """Build final LightGBM parameter dict (single, fixed block).

    No per-fold overrides.
    """

    # Default eval cutoffs should not exceed the query size when chunking is enabled.
    chunk_size = int(getattr(cfg, "ticker_chunk_size", 0) or 0)
    if chunk_size > 0:
        base_eval_at = [min(chunk_size, 10), min(chunk_size, 20), min(chunk_size, 50)]
    else:
        base_eval_at = [10, 20, 50]

    params: Dict[str, Any] = default_lgb_params(seed=cfg.seed, num_threads=cfg.num_threads, eval_at=base_eval_at)
    params.update(cfg.lgb_params or {})

    params["objective"] = "lambdarank"
    params["metric"] = "ndcg"

    # Ensure eval_at is a list of positive ints.
    eval_at = params.get("eval_at")
    if isinstance(eval_at, (list, tuple)):
        eval_at_list = [int(x) for x in eval_at if int(x) > 0]
    else:
        eval_at_list = [10, 20, 50]

    # Early stopping uses the first cutoff; keep primary_k first if provided.
    if primary_k is not None:
        pk = int(primary_k)
        eval_at_list = [pk] + [k for k in eval_at_list if int(k) != pk]

    # Unique, preserve order
    seen: set[int] = set()
    eval_at_list_uniq: list[int] = []
    for k in eval_at_list:
        if k in seen:
            continue
        seen.add(int(k))
        eval_at_list_uniq.append(int(k))

    params["eval_at"] = eval_at_list_uniq
    params["first_metric_only"] = True

    # Align truncation level with the monitored cutoff unless explicitly set.
    if params.get("lambdarank_truncation_level") is None:
        if eval_at_list_uniq:
            params["lambdarank_truncation_level"] = int(eval_at_list_uniq[0])

    # Ensure label_gain is long enough for the chosen relevance bins.
    n_bins = int(cfg.n_relevance_bins or 0)
    if n_bins > 0:
        lg = params.get("label_gain")
        if not (isinstance(lg, (list, tuple)) and len(lg) >= n_bins):
            params["label_gain"] = list(range(n_bins))

    return params
