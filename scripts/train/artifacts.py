from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config.logging import get_logger

from .types import TrainingPaths

logger = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with reasonable conversions for numpy/pandas types."""

    _ensure_dir(path)
    path.write_text(json.dumps(_to_jsonable(data), indent=2), encoding="utf-8")


def save_model(paths: TrainingPaths, booster: lgb.Booster) -> None:
    _ensure_dir(paths.model_path)
    booster.save_model(str(paths.model_path))


def export_feature_importance(paths: TrainingPaths, booster: lgb.Booster) -> None:
    fi_gain = booster.feature_importance("gain")
    fi_split = booster.feature_importance("split")
    fi_df = (
        pd.DataFrame({"feature": booster.feature_name(), "gain": fi_gain, "split": fi_split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    _ensure_dir(paths.fi_csv)
    fi_df.to_csv(paths.fi_csv, index=False)

    try:
        top = fi_df.head(30)
        plt.figure(figsize=(8, max(4, len(top) * 0.25)))
        plt.barh(top["feature"][::-1], top["gain"][::-1], color="#2166ac")
        plt.xlabel("Gain")
        plt.title("Feature Importance (LambdaRank)")
        plt.tight_layout()
        _ensure_dir(paths.fi_png)
        plt.savefig(paths.fi_png, dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("FI plot failed: %s", e)


def export_learning_curve(
    paths: TrainingPaths,
    evals: Dict[str, Dict[str, list[float]]],
    *,
    best_iteration: int | None = None,
) -> None:
    train = evals.get("train") or {}
    if not train:
        return

    metric_keys = [m for m in train.keys() if m.startswith("ndcg")]
    if not metric_keys:
        return

    n = len(next(iter(train.values())))
    stopped_iteration = int(n)
    best_iter_int = int(best_iteration) if best_iteration is not None else -1
    rows: list[dict[str, Any]] = []
    for i in range(n):
        iteration = int(i + 1)
        row: dict[str, Any] = {
            "iteration": iteration,
            "best_iteration": (best_iter_int if best_iter_int > 0 else None),
            "stopped_iteration": stopped_iteration,
            "is_best_iteration": bool(best_iter_int > 0 and iteration == best_iter_int),
            "is_stopped_iteration": bool(iteration == stopped_iteration),
        }
        for mk in metric_keys:
            row[f"train_{mk}"] = float(evals["train"][mk][i])
            if mk in (evals.get("valid") or {}):
                row[f"valid_{mk}"] = float((evals.get("valid") or {})[mk][i])
        rows.append(row)

    df = pd.DataFrame(rows)
    _ensure_dir(paths.learning_curve_csv)
    df.to_csv(paths.learning_curve_csv, index=False)

    try:
        train_cols = [c for c in df.columns if c.startswith("train_ndcg")]
        valid_cols = [c for c in df.columns if c.startswith("valid_ndcg")]
        metric_cols = sorted(valid_cols) + [c for c in sorted(train_cols) if c not in set(valid_cols)]
        if not metric_cols:
            return

        plt.figure(figsize=(10, 5))
        x = df["iteration"].astype(int)
        for c in metric_cols:
            plt.plot(x, df[c].astype(float), linewidth=1.6, label=c)

        if best_iter_int > 0:
            plt.axvline(best_iter_int, color="#d95f02", linestyle="-", linewidth=1.4, label="best_iteration")
        if stopped_iteration > 0 and stopped_iteration != best_iter_int:
            plt.axvline(
                stopped_iteration,
                color="#7570b3",
                linestyle=":",
                linewidth=1.4,
                label="stopped_iteration",
            )

        plt.xlabel("Iteration")
        plt.ylabel("NDCG")
        plt.title("Learning curve")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="best", ncols=2)
        plt.tight_layout()
        _ensure_dir(paths.learning_curve_png)
        plt.savefig(paths.learning_curve_png, dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Learning curve plot failed: %s", e)
