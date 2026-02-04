import pandas as pd

from scripts.backtest.walkforward_backtest import generate_walkforward_splits


def test_walkforward_splits_respect_purge_and_order():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    idx = pd.Index(dates)

    splits = generate_walkforward_splits(
        dates,
        train_days=200,
        valid_days=50,
        test_days=30,
        step_days=30,
        purge_days=10,
        max_splits=3,
    )
    assert len(splits) == 3

    for sp in splits:
        train_end = pd.Timestamp(sp["train_end"]).normalize()
        valid_start = pd.Timestamp(sp["valid_start"]).normalize()
        valid_end = pd.Timestamp(sp["valid_end"]).normalize()
        test_start = pd.Timestamp(sp["test_start"]).normalize()

        te = int(idx.get_indexer([train_end])[0])
        vs = int(idx.get_indexer([valid_start])[0])
        ve = int(idx.get_indexer([valid_end])[0])
        ts = int(idx.get_indexer([test_start])[0])

        assert train_end < valid_start
        assert (vs - te - 1) >= 10
        assert valid_end < test_start
        assert (ts - ve - 1) >= 10

        assert sp["train_start"] <= sp["train_end"]
        assert sp["valid_start"] <= sp["valid_end"]
        assert sp["test_start"] <= sp["test_end"]
