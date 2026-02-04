import json

import numpy as np
import pandas as pd


def test_training_persists_categorical_levels(tmp_path, monkeypatch):
    """Training should persist categorical levels for stable prediction encoding."""
    monkeypatch.chdir(tmp_path)

    # Minimal project layout expected by the code
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    # Make target resolution deterministic at import time
    (tmp_path / "models" / "features.json").write_text(
        json.dumps({"target": "target_fwd_5d"}), encoding="utf-8"
    )

    import scripts.train.train as train_mod

    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    n_per_date = 12
    rows = []
    for d in dates:
        for i in range(n_per_date):
            rows.append(
                {
                    "ticker": f"T{i%20:02d}",
                    "date": d,
                    "target_fwd_5d": float(rng.normal(0, 0.05)),
                    "f1": float(rng.normal()),
                    "sector": "Tech" if (i % 2 == 0) else "Finance",
                    "industry": "Software" if (i % 3 == 0) else "Banks",
                }
            )
    df = pd.DataFrame(rows)

    panel_path = tmp_path / "data" / "processed" / "raw_training.parquet"
    df.to_parquet(panel_path, index=False)

    # Train quickly (small num_boost_round) - we only care about artifacts.
    cfg = train_mod.TrainValidSplitConfig(
        valid_days=30,
        forward_gap_days=5,
        min_cross_section=2,
        n_relevance_bins=5,
        lgb_params={
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_data_in_leaf": 20,
            "max_depth": 5,
            "verbose": -1,
            "eval_at": [20],
        },
    )
    train_mod.train_model_learn_to_rank(cfg=cfg, num_boost_round=25, early_stopping_rounds=5, primary_k=20)

    meta = json.loads((tmp_path / "models" / "features.json").read_text(encoding="utf-8"))
    levels = meta.get("categorical_levels")
    assert isinstance(levels, dict)
    assert "sector" in levels and "industry" in levels
    assert "Tech" in levels["sector"] or "Finance" in levels["sector"]
