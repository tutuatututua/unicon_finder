"""Training entrypoint (LightGBM LambdaRank).

The project now uses a single chronological train/validation split.

`train_model_learn_to_rank()` keeps a stable import path for the orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.config.logging import get_logger

from .config import TrainValidSplitConfig
from .pipeline import train_single_split
from .types import TrainingPaths

logger = get_logger(__name__)


def train_model_learn_to_rank(
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    primary_k: Optional[int] = None,
    cfg: Optional[TrainValidSplitConfig] = None,
    paths: Optional["TrainingPaths"] = None,
) -> Dict[str, Any]:
    """Train LambdaRank once using a single train/validation split."""

    cfg = cfg or TrainValidSplitConfig()

    return train_single_split(
        cfg=cfg,
        paths=paths,
        num_boost_round=int(num_boost_round),
        early_stopping_rounds=int(early_stopping_rounds),
        primary_k=(int(primary_k) if primary_k is not None else None),
    )


__all__ = ["train_model_learn_to_rank", "TrainValidSplitConfig"]
