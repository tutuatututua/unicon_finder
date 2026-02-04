import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_panel_split_has_gap(tmp_path, monkeypatch):
    """Panel split should enforce a forward-gap between train and validation."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    # Ensure TARGET_COL resolves
    (tmp_path / "models" / "features.json").write_text(
        json.dumps({"target": "target_fwd_5d"}), encoding="utf-8"
    )

    import scripts.train.train as train_mod

    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    tickers = ["AAA", "BBB", "CCC", "DDD"]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "target_fwd_5d": float(rng.normal(0, 0.02)),
                    "f1": float(rng.normal()),
                }
            )

    panel = pd.DataFrame(rows)
    panel_path = tmp_path / "data" / "processed" / "raw_training.parquet"
    panel.to_parquet(panel_path, index=False)

    from scripts.train.io import load_panel, resolve_target
    from scripts.train.pipeline import time_train_valid_split
    from scripts.train.types import TrainingPaths

    cfg = train_mod.TrainValidSplitConfig(
        valid_days=30,
        forward_gap_days=5,
        min_cross_section=2,
    )
    paths = TrainingPaths(raw_panel_path=panel_path, feature_meta_path=(tmp_path / "models" / "features.json"))
    target_col = resolve_target(paths)
    df = load_panel(paths, target_col=target_col, min_cross_section=cfg.min_cross_section, target_clip=cfg.target_clip)

    train_df, valid_df, meta = time_train_valid_split(df, valid_days=cfg.valid_days, forward_gap_days=cfg.forward_gap_days)
    assert not train_df.empty
    assert not valid_df.empty

    train_end = pd.Timestamp(meta["train_end"]).normalize()
    valid_start = pd.Timestamp(meta["valid_start"]).normalize()
    assert train_end < valid_start
    assert int((valid_start - train_end).days) >= int(cfg.forward_gap_days)
