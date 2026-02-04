from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

# Allow running as either:
# - python -m scripts.experiments.ab_feature_scaling
# - python scripts/experiments/ab_feature_scaling.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.backtest.backtest import BacktestConfig, BacktestPaths, run_backtest
from scripts.backtest.compare_to_sp500 import CompareConfig, compare_model_vs_sp500
from scripts.config.config import load_yaml_config
from scripts.config.logging import get_logger
from scripts.config.paths import ProjectPaths
from scripts.feature.feature_engineering import build_features
from scripts.predict import Paths as PredictPaths
from scripts.predict import predict
from scripts.train.config import TrainValidSplitConfig
from scripts.train.train import train_model_learn_to_rank
from scripts.train.types import TrainingPaths

logger = get_logger(__name__)


@dataclass(frozen=True)
class VariantResult:
    variant: str
    train_metrics: dict[str, Any]
    backtest_summary: dict[str, Any]
    sp500_compare: dict[str, Any]
    models_dir: str
    backtest_dir: str


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_train_cfg(raw_cfg: Mapping[str, Any], *, feature_scaling: str) -> tuple[TrainValidSplitConfig, dict[str, int]]:
    model_cfg = _as_dict(raw_cfg.get("model"))
    split_cfg_raw = _as_dict(model_cfg.get("split"))

    td_raw = split_cfg_raw.get("train_days")
    train_days = None if td_raw is None else int(td_raw)

    cfg = TrainValidSplitConfig(
        train_days=train_days,
        valid_days=int(split_cfg_raw.get("valid_days", 365)),
        forward_gap_days=int(split_cfg_raw.get("forward_gap_days", 5)),
        min_cross_section=int(split_cfg_raw.get("min_cross_section", 200)),
        n_relevance_bins=int(split_cfg_raw.get("n_relevance_bins", 20)),
        include_sector_industry=bool(split_cfg_raw.get("include_sector_industry", True)),
        feature_scaling="none",
        num_threads=int(split_cfg_raw.get("num_threads", 1)),
        log_evaluation_period=int(split_cfg_raw.get("log_evaluation_period", 50)),
        lgb_params=_as_dict(model_cfg.get("params")),
    )

    train_loop = {
        "num_boost_round": int(model_cfg.get("num_boost_round", 1500)),
        "early_stopping_rounds": int(model_cfg.get("early_stopping_rounds", 40)),
        "primary_k": int(model_cfg.get("primary_k", 200)),
    }

    return cfg, train_loop


def _build_backtest_cfg(raw_cfg: Mapping[str, Any]) -> BacktestConfig:
    bt_raw = _as_dict(raw_cfg.get("backtest"))
    return BacktestConfig(
        use_model_score=bool(bt_raw.get("use_model_score", True)),
        score_col=bt_raw.get("score_col"),
        top_n=int(bt_raw.get("top_n", 20)),
        bottom_n=int(bt_raw.get("bottom_n", 20)),
        require_min_cross_section=int(bt_raw.get("require_min_cross_section", 30)),
        date_step=int(bt_raw.get("date_step", 1)),
        max_dates=bt_raw.get("max_dates"),
        save_picks=bool(bt_raw.get("save_picks", True)),
    )


def _copy_seed_features_meta(*, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return
    # It's OK if missing: training will fall back to reading all panel columns and
    # will still write a complete runtime metadata file.
    logger.warning("Seed features meta not found at %s; proceeding without copy.", src)


def run_ab(
    *,
    config_path: Path,
    run_name: str,
    variants: list[str],
    force_rebuild_features: bool,
    skip_if_model_exists: bool,
    run_predict: bool,
    run_backtest_stage: bool,
) -> list[VariantResult]:
    raw_cfg = load_yaml_config(config_path)
    paths = ProjectPaths.from_config(raw_cfg)
    paths.ensure_dirs()

    # Ensure feature artifacts exist (shared across variants)
    if force_rebuild_features or not (
        paths.raw_training_parquet.exists()
        and paths.latest_snapshot_parquet.exists()
        and paths.last_labeled_snapshot_parquet.exists()
    ):
        logger.info("Building features (force=%s)...", force_rebuild_features)
        build_features(raw_dir=paths.raw_dir, processed_dir=paths.processed_dir, models_dir=paths.models_dir)

    ab_models_root = paths.models_dir / "ab_feature_scaling"
    ab_backtest_root = paths.backtest_dir / "ab_feature_scaling"
    ab_models_root.mkdir(parents=True, exist_ok=True)
    ab_backtest_root.mkdir(parents=True, exist_ok=True)

    results: list[VariantResult] = []

    for variant in variants:
        variant_key = str(variant).strip().lower()
        if variant_key != "none":
            raise ValueError("Only variant 'none' is supported (zscore has been removed).")

        variant_models_dir = ab_models_root / variant_key / run_name
        variant_backtest_dir = ab_backtest_root / variant_key / run_name
        variant_models_dir.mkdir(parents=True, exist_ok=True)
        variant_backtest_dir.mkdir(parents=True, exist_ok=True)

        model_path = variant_models_dir / "lightgbm_model.txt"
        if skip_if_model_exists and model_path.exists():
            logger.info("Skipping training for %s; model exists at %s", variant_key, model_path)
        else:
            # Seed variant models dir with base feature list/target (if available)
            _copy_seed_features_meta(src=paths.features_meta_json, dst=variant_models_dir / "features.json")

            cfg, loop = _build_train_cfg(raw_cfg, feature_scaling=variant_key)

            tpaths = TrainingPaths(
                raw_panel_path=paths.raw_training_parquet,
                feature_meta_path=variant_models_dir / "features.json",
                model_path=model_path,
                metrics_path=variant_models_dir / "metrics.json",
                config_path=variant_models_dir / "config.json",
                fi_csv=variant_models_dir / "feature_importance.csv",
                fi_png=variant_models_dir / "feature_importance.png",
                learning_curve_csv=variant_models_dir / "learning_curve_ndcg.csv",
                learning_curve_png=variant_models_dir / "learning_curve_ndcg.png",
            )

            logger.info("Training variant=%s into %s", variant_key, variant_models_dir)
            train_model_learn_to_rank(
                num_boost_round=int(loop["num_boost_round"]),
                early_stopping_rounds=int(loop["early_stopping_rounds"]),
                primary_k=int(loop["primary_k"]),
                cfg=cfg,
                paths=tpaths,
            )

        # Prediction
        if run_predict:
            ppaths = PredictPaths(
                scoring_features=paths.latest_snapshot_parquet,
                last_labeled_features=paths.last_labeled_snapshot_parquet,
                all_features=paths.raw_training_parquet,
                rank_model=variant_models_dir / "lightgbm_model.txt",
                predictions=variant_models_dir / "predictions.csv",
                feature_meta=variant_models_dir / "features.json",
                metrics=variant_models_dir / "metrics.json",
            )

            top_n = int(_as_dict(raw_cfg.get("prediction")).get("top_n", 20))
            logger.info("Predicting variant=%s (top_n=%d)", variant_key, top_n)
            predict(top_n=top_n, snapshot="latest", paths=ppaths)

        # Backtest + compare to S&P500
        bt_summary: dict[str, Any] = {}
        sp500_summary: dict[str, Any] = {}
        if run_backtest_stage:
            bt_cfg = _build_backtest_cfg(raw_cfg)
            bt_paths = BacktestPaths(
                model_path=variant_models_dir / "lightgbm_model.txt",
                metrics_path=variant_models_dir / "metrics.json",
                feature_meta_path=variant_models_dir / "features.json",
                panel_path=paths.raw_training_parquet,
                output_dir=variant_backtest_dir,
            )

            logger.info("Backtesting variant=%s into %s", variant_key, variant_backtest_dir)
            bt_summary = run_backtest(bt_cfg, paths=bt_paths)

            cmp_cfg = CompareConfig(
                benchmark_path=CompareConfig.benchmark_path,
                backtest_timeseries_csv=variant_backtest_dir / "backtest_timeseries.csv",
                backtest_summary_json=variant_backtest_dir / "backtest_summary.json",
                output_csv=variant_backtest_dir / "merged_sp500_model.csv",
                output_summary_json=variant_backtest_dir / "summary.json",
            )
            sp500_summary = compare_model_vs_sp500(cmp_cfg)
            sp500_summary["variant"] = variant_key

        # Load train metrics payload from disk (authoritative)
        train_metrics_path = variant_models_dir / "metrics.json"
        train_metrics: dict[str, Any] = {}
        if train_metrics_path.exists():
            try:
                train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                train_metrics = {}

        results.append(
            VariantResult(
                variant=variant_key,
                train_metrics=train_metrics,
                backtest_summary=bt_summary,
                sp500_compare=sp500_summary,
                models_dir=str(variant_models_dir.as_posix()),
                backtest_dir=str(variant_backtest_dir.as_posix()),
            )
        )

    # Write a compact comparison table
    rows: list[dict[str, Any]] = []
    for r in results:
        # Find the primary valid NDCG key if present.
        valid_ndcg_key = next((k for k in r.train_metrics.keys() if str(k).startswith("valid_ndcg@")), None)
        rows.append(
            {
                "variant": r.variant,
                "best_iteration": r.train_metrics.get("best_iteration"),
                "primary_k": r.train_metrics.get("primary_k"),
                "valid_ndcg": (r.train_metrics.get(valid_ndcg_key) if valid_ndcg_key else None),
                "mean_long_short": r.backtest_summary.get("mean_long_short"),
                "mean_top": r.backtest_summary.get("mean_top"),
                "mean_alpha_top": r.sp500_compare.get("mean_alpha_top"),
                "alpha_win_rate": r.sp500_compare.get("alpha_win_rate"),
                "n_rows_with_benchmark": r.sp500_compare.get("n_rows_with_benchmark"),
                "models_dir": r.models_dir,
                "backtest_dir": r.backtest_dir,
            }
        )

    compare_csv = ab_backtest_root / f"compare_{run_name}.csv"
    compare_json = ab_backtest_root / f"compare_{run_name}.json"

    df = pd.DataFrame(rows).sort_values("variant")
    df.to_csv(compare_csv, index=False)
    compare_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    logger.info("Wrote comparison table: %s", compare_csv)

    return results


def _cli() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run feature scaling experiment (zscore removed; only none supported)")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--run-name", type=str, default="quick", help="Output subfolder name")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["none"],
        help="Variants to run (default: none).",
    )
    ap.add_argument("--force-rebuild-features", action="store_true", help="Rebuild features before running.")
    ap.add_argument("--skip-if-model-exists", action="store_true", help="Skip training if model file exists.")
    ap.add_argument("--no-predict", action="store_true", help="Skip prediction stage.")
    ap.add_argument("--no-backtest", action="store_true", help="Skip backtest + S&P500 compare stage.")

    args = ap.parse_args()

    run_ab(
        config_path=Path(args.config),
        run_name=str(args.run_name),
        variants=[str(v) for v in (args.variants or [])],
        force_rebuild_features=bool(args.force_rebuild_features),
        skip_if_model_exists=bool(args.skip_if_model_exists),
        run_predict=not bool(args.no_predict),
        run_backtest_stage=not bool(args.no_backtest),
    )


if __name__ == "__main__":  # pragma: no cover
    _cli()
