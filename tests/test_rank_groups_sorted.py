import numpy as np
import pandas as pd

from scripts.train.config import TrainValidSplitConfig
from scripts.train.dataset import add_date_rank_relevance, build_split_bundle


def test_build_fold_bundle_sorts_by_date_and_groups_sum() -> None:
    rng = np.random.default_rng(0)

    dates = pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
    ])

    rows = []
    for d in dates:
        for i in range(5):
            rows.append(
                {
                    "ticker": f"T{i:02d}",
                    "date": d,
                    "target_fwd_5d": float(rng.normal(0, 0.05)),
                    "f1": float(rng.normal()),
                    # A feature that should become monotonic after sorting.
                    "date_ord": int(pd.Timestamp(d).value),
                }
            )

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=123).reset_index(drop=True)

    cfg = TrainValidSplitConfig(
        n_relevance_bins=5,
        force_negatives_to_zero=False,
        min_cross_section=2,
    )

    df = add_date_rank_relevance(
        df,
        target_col="target_fwd_5d",
        n_bins=cfg.n_relevance_bins,
        force_negatives_to_zero=cfg.force_negatives_to_zero,
        out_col="relevance",
    )

    train_df = df[df["date"] <= pd.Timestamp("2024-01-03")].copy()
    valid_df = df[df["date"] >= pd.Timestamp("2024-01-04")].copy()

    bundle = build_split_bundle(
        train_df,
        valid_df,
        cfg=cfg,
        target_col="target_fwd_5d",
        feature_cols=["f1", "date_ord"],
        train_end_date=pd.Timestamp("2024-01-03"),
    )

    # Groups should cover all rows.
    assert sum(bundle.group_train) == len(bundle.X_train)
    assert sum(bundle.group_valid) == len(bundle.X_valid)

    # The bundle should be sorted by date (then ticker if present), which implies
    # date_ord is non-decreasing.
    assert bundle.X_train["date_ord"].is_monotonic_increasing
    assert bundle.X_valid["date_ord"].is_monotonic_increasing
