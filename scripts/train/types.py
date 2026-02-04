from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingPaths:
    raw_panel_path: Path = Path("data/processed/raw_training.parquet")
    feature_meta_path: Path = Path("models/features.json")

    model_path: Path = Path("models/lightgbm_model.txt")
    metrics_path: Path = Path("models/metrics.json")
    config_path: Path = Path("models/config.json")

    fi_csv: Path = Path("models/feature_importance.csv")
    fi_png: Path = Path("models/feature_importance.png")

    learning_curve_csv: Path = Path("models/learning_curve_ndcg.csv")
    learning_curve_png: Path = Path("models/learning_curve_ndcg.png")


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_rel_train: pd.Series
    y_rel_valid: pd.Series
    y_cont_valid: pd.Series
    date_valid: pd.Series
    group_train: List[int]
    group_valid: List[int]
    train_weights: Optional[np.ndarray]
    meta: Dict[str, Any]
