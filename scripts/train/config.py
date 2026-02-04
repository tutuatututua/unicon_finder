from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

ScalingMode = Literal["none"]


@dataclass
class TrainValidSplitConfig:
    """Configuration for a single time-based train/validation split."""

    # Data / schema
    target_col: Optional[str] = None  # resolved from models/features.json if None
    group_col: str = "date"  # query groups for ranking

    # Split + leakage controls
    valid_days: int = 365
    forward_gap_days: int = 21

    # Optional rolling training window (count of unique dates).
    # If set, training uses only the last `train_days` dates ending at `train_end`.
    # (Note: "days" here means unique panel dates, not calendar days.)
    train_days: Optional[int] = None

    # Panel hygiene
    min_cross_section: int = 200
    target_clip: float = 30.0

    # Relevance labels (LambdaRank expects integer relevance)
    n_relevance_bins: int = 20
    force_negatives_to_zero: bool = False

    # Training
    seed: int = 42
    num_threads: int = 1  # deterministic by default
    recency_lambda: float = 0.0

    # Preprocessing
    feature_scaling: ScalingMode = "none"

    # Feature hygiene
    drop_constant_features: bool = True
    max_missing_frac: float = 0.98
    min_unique_values: int = 2

    # Optional categorical metadata features
    include_sector_industry: bool = True

    # Training progress logging. 0 disables periodic logs.
    log_evaluation_period: int = 50

    # LightGBM LambdaRank params (overrides defaults)
    lgb_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.feature_scaling).strip().lower() != "none":
            raise ValueError(
                f"Unsupported feature_scaling={self.feature_scaling!r}. "
                "Only 'none' is supported (zscore has been removed)."
            )

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_lgb_params(*, seed: int, num_threads: int) -> Dict[str, Any]:
    """Repo-default LightGBM params for LambdaRank.

    Kept intentionally fixed and centralized for fast iteration.
    """

    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 200,
        "min_sum_hessian_in_leaf": 5.0,
        "min_gain_to_split": 0.01,
        "lambda_l1": 0.1,
        "lambda_l2": 5.0,
        "feature_fraction": 0.7,
        "feature_fraction_seed": int(seed),
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "bagging_seed": int(seed),
        "max_bin": 255,
        "boosting": "gbdt",
        # Put the primary cutoff first (pipeline can reorder to match primary_k).
        "eval_at": [200, 100, 50],
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

    params: Dict[str, Any] = default_lgb_params(seed=cfg.seed, num_threads=cfg.num_threads)
    params.update(cfg.lgb_params or {})

    params["objective"] = "lambdarank"
    params["metric"] = "ndcg"

    # Ensure eval_at is a list of positive ints.
    eval_at = params.get("eval_at")
    if isinstance(eval_at, (list, tuple)):
        eval_at_list = [int(x) for x in eval_at if int(x) > 0]
    else:
        eval_at_list = [200, 100, 50]

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
