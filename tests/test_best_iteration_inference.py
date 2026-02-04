from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def test_load_best_iteration_helper(tmp_path: Path) -> None:
    from scripts import predict as predict_mod

    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"best_iteration": 7}), encoding="utf-8")

    assert predict_mod._load_best_iteration(metrics) == 7

    metrics.write_text(json.dumps({"best_iteration": -1}), encoding="utf-8")
    assert predict_mod._load_best_iteration(metrics) is None

    metrics.write_text(json.dumps({"best_iteration": "not-an-int"}), encoding="utf-8")
    assert predict_mod._load_best_iteration(metrics) is None


def test_backtest_uses_metrics_best_iteration(monkeypatch, tmp_path: Path) -> None:
    from scripts.backtest import backtest as bt

    # Arrange: minimal features + model artifacts
    model_path = tmp_path / "lightgbm_model.txt"
    model_path.write_text("dummy", encoding="utf-8")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"best_iteration": 7, "score_sign": 1.0}), encoding="utf-8")

    features_meta = tmp_path / "features.json"
    features_meta.write_text(
        json.dumps(
            {
                "categorical_levels": {},
            }
        ),
        encoding="utf-8",
    )

    # Patch module-level constants to point at our temp artifacts
    monkeypatch.setattr(bt, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(bt, "FEATURE_META", features_meta)

    class FakeBooster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._calls: list[Optional[int]] = []

        def feature_name(self) -> list[str]:
            return ["f1"]

        def predict(self, X: pd.DataFrame, num_iteration: Optional[int] = None):
            self._calls.append(num_iteration)
            # Ensure we honor best_iteration from metrics.json
            assert num_iteration == 7
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(bt.lgb, "Booster", lambda model_file=None: FakeBooster())

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "ticker": ["A", "B", "C"],
            "f1": [1.0, 2.0, 3.0],
        }
    )

    out, feats = bt._score_with_model(df, model_path, chunk_rows=2)

    assert feats == ["f1"]
    assert "score" in out.columns
    assert len(out) == len(df)


def test_predict_passes_best_iteration_to_lightgbm(monkeypatch, tmp_path: Path) -> None:
    # This is an integration-level test for scripts.predict.predict(), but it uses
    # a fake booster so we don't depend on LightGBM training here.
    from scripts import predict as predict_mod

    # Minimal feature snapshot
    snap = tmp_path / "latest.parquet"
    feats = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "f1": [1.0, 2.0],
        }
    )
    feats.to_parquet(snap, index=False)

    dummy_model = tmp_path / "lightgbm_model.txt"
    dummy_model.write_text("dummy", encoding="utf-8")

    # Make predict write outputs to temp
    monkeypatch.setattr(
        predict_mod,
        "PATHS",
        replace(predict_mod.PATHS, scoring_features=snap, rank_model=dummy_model, predictions=tmp_path / "preds.csv"),
    )

    # Ensure predict uses our best-iteration value
    monkeypatch.setattr(predict_mod, "_load_best_iteration", lambda *args, **kwargs: 7)

    class FakeBooster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def feature_name(self) -> list[str]:
            return ["f1"]

        def predict(self, X: pd.DataFrame, num_iteration: Optional[int] = None):
            assert num_iteration == 7
            return np.array([0.1] * len(X), dtype=float)

    monkeypatch.setattr(predict_mod.lgb, "Booster", lambda model_file=None: FakeBooster())

    # Avoid depending on any real models/features.json on disk
    from scripts.feature.metadata import FeaturesMeta

    monkeypatch.setattr(predict_mod, "load_features_meta", lambda *args, **kwargs: FeaturesMeta(raw={}))

    out = predict_mod.predict(top_n=1, snapshot="latest")
    assert len(out) == 1
    assert "rank_score" in out.columns
