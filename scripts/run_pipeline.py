# run_pipeline.py

"""End-to-end ML pipeline orchestrator (ranking-only).

Stages (controlled via `config.yaml`):
 1. Configuration loading.
 2. Historical data download (skipped if not forced and cache present).
 3. Feature engineering (optionally skipped via checkpoint flags).
 4. Model training (ranking head only).
 5. Prediction generation on latest or last labeled snapshot.

Model mode: lambdarank (ranking head).

This script is intentionally thin: heavy logic lives in dedicated modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as either:
# - python -m scripts.run_pipeline
# - python scripts/run_pipeline.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.backtest.backtest import BacktestConfig, run_backtest
from scripts.data.data_download import download_full_history
from scripts.data.download_all_us_tickers import get_all_tickers
from scripts.feature.feature_engineering import build_features
from scripts.config.logging import get_logger
from scripts.predict import predict
from scripts.train.train import train_model_learn_to_rank
from scripts.train.config import TrainValidSplitConfig
from scripts.config.config import CheckpointConfig, PredictionConfig, load_yaml_config
from scripts.config.paths import ProjectPaths
logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _has_raw_history(raw_dir: Path) -> bool:
    return raw_dir.exists() and any(raw_dir.glob("*.parquet"))


def _has_feature_artifacts(paths: ProjectPaths) -> bool:
    return (
        paths.raw_training_parquet.exists()
        and paths.latest_snapshot_parquet.exists()
        and paths.last_labeled_snapshot_parquet.exists()
    )


def _has_trained_model(model_path: Path) -> bool:
    return model_path.exists()


def _download_history(*, universe_csv: Path, raw_dir: Path) -> None:
    """
    Downloads historical data for tickers defined in the universe file.
    Skips if a valid checkpoint exists and data files are present, unless forced.
    """
    logger.info("Starting full-history download for tickers in %s...", universe_csv)
    if not universe_csv.exists():
        logger.error("Ticker universe file not found at %s. Cannot proceed.", universe_csv)
        raise FileNotFoundError(f"Missing ticker universe file: {universe_csv}")

    try:
        tickers = download_full_history(universe_csv=universe_csv, raw_dir=raw_dir)
        successful_downloads = len(tickers)
        logger.info("Download complete: %d tickers processed.", successful_downloads)
    except Exception as e:
        logger.error("An error occurred during data download: %s", e, exc_info=True)
        raise


def main() -> None:
    """Runs the full ML pipeline from data ingestion to prediction."""

    # --- Stage 1: Configuration ---
    logger.info("Loading configuration from %s...", CONFIG_PATH)
    raw_cfg = load_yaml_config(CONFIG_PATH)
    paths = ProjectPaths.from_config(raw_cfg)
    paths.ensure_dirs()

    checkpoint_cfg = CheckpointConfig.from_config(raw_cfg)
    prediction_cfg = PredictionConfig.from_config(raw_cfg)

    # --- Stage 2: Ticker Universe and Sector Download ---
    if not paths.universe_csv.exists():
        logger.info("Fetching all US stock tickers (NASDAQ, NYSE, AMEX)...")
        syms = get_all_tickers(include_etfs=False)
        paths.universe_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": syms}).to_csv(paths.universe_csv, index=False)
        logger.info("Saved %d tickers to %s", len(syms), paths.universe_csv)
    else:
        logger.info("Universe file found at %s. Skipping ticker download.", paths.universe_csv)

    if not paths.sector_map_csv.exists():
        from scripts.data.download_sectors import main as download_sectors_main
        logger.info("Downloading sector/industry mapping for universe tickers...")
        download_sectors_main()
    else:
        logger.info("Sector map file found at %s. Skipping sector download.", paths.sector_map_csv)

    # --- Stage 3: Data Download ---
    force_fetch = checkpoint_cfg.force_fetch
    if force_fetch or not _has_raw_history(paths.raw_dir):
        logger.info("Downloading/updating historical data (force=%s)...", force_fetch)
        _download_history(
            universe_csv=paths.universe_csv,
            raw_dir=paths.raw_dir,
        )
    else:
        logger.info("Raw history already present; skipping download.")

    # --- Stage 4: Feature Engineering ---
    logger.info("Starting feature engineering stage...")
    force_rebuild = checkpoint_cfg.force_rebuild
    if force_rebuild or not _has_feature_artifacts(paths):
        logger.info("Building features (force=%s)...", force_rebuild)
        build_features(
            raw_dir=paths.raw_dir,
            processed_dir=paths.processed_dir,
            models_dir=paths.models_dir,
        )
    else:
        logger.info("Feature artifacts already present; skipping build.")
    logger.info("Feature engineering complete.")

    # --- Stage 5: Model Training ---
    logger.info("Starting model training stage...")
    model_cfg = raw_cfg.get('model', {})

    metrics: dict[str, Any] = {}
    force_retrain = checkpoint_cfg.force_retrain
    if force_retrain or not _has_trained_model(paths.model_file):
        logger.info("Training ranking head (LambdaRank) (force=%s)...", force_retrain)

        split_cfg_raw = model_cfg.get("split", {}) if isinstance(model_cfg.get("split", {}), dict) else {}

        fs_raw = split_cfg_raw.get("feature_scaling", "none")
        if str(fs_raw).strip().lower() == "zscore":
            logger.warning("feature_scaling='zscore' is no longer supported; using 'none'.")
        feature_scaling = "none"

        train_days_parsed: int | None
        td_raw = split_cfg_raw.get("train_days")
        if td_raw is None:
            train_days_parsed = None
        else:
            train_days_parsed = int(td_raw)

        cfg = TrainValidSplitConfig(
            train_days=train_days_parsed,
            valid_days=int(split_cfg_raw.get("valid_days", 365)),
            forward_gap_days=int(split_cfg_raw.get("forward_gap_days", 5)),
            min_cross_section=int(split_cfg_raw.get("min_cross_section", 200)),
            n_relevance_bins=int(split_cfg_raw.get("n_relevance_bins", 20)),
            include_sector_industry=bool(split_cfg_raw.get("include_sector_industry", True)),
            feature_scaling=feature_scaling,
            num_threads=int(split_cfg_raw.get("num_threads", 1)),
            log_evaluation_period=int(split_cfg_raw.get("log_evaluation_period", 50)),
            lgb_params=(model_cfg.get("params", {}) if isinstance(model_cfg.get("params", {}), dict) else {}),
        )

        metrics['ranking'] = train_model_learn_to_rank(
            num_boost_round=int(model_cfg.get('num_boost_round', 1500)),
            early_stopping_rounds=int(model_cfg.get('early_stopping_rounds', 40)),
            primary_k=int(model_cfg.get('primary_k', 200)),
            cfg=cfg,
        )
    else:
        logger.info("Model already trained; skipping retrain.")
    logger.info("Model training complete. Heads present: %s", ", ".join(metrics.keys()) or "none")

    # --- Stage 6: Prediction ---
    logger.info("Generating predictions on latest data...")
    top_n = prediction_cfg.top_n
    # Determine which snapshot to use for prediction based on config
    snapshot_mode = 'last_labeled' if prediction_cfg.use_last_labeled else 'latest'
    
    predictions = predict(top_n=top_n, snapshot=snapshot_mode)
    logger.info("Top %d predictions for 252-day forward return:\n%s", top_n, predictions.head(top_n).to_string(index=False))

    # --- Stage 7: Backtest ---
    bt_cfg_raw = raw_cfg.get('backtest', {})
    if bt_cfg_raw.get('run', True):
        logger.info("Running cross-sectional backtest...")
        bt_cfg = BacktestConfig(
            use_model_score=bt_cfg_raw.get('use_model_score', True),
            score_col=bt_cfg_raw.get('score_col'),
            top_n=int(bt_cfg_raw.get('top_n', 20)),
            bottom_n=int(bt_cfg_raw.get('bottom_n', 20)),
            require_min_cross_section=int(bt_cfg_raw.get('require_min_cross_section', 30)),
            date_step=int(bt_cfg_raw.get('date_step', 1)),
            max_dates=bt_cfg_raw.get('max_dates'),
            save_picks=bool(bt_cfg_raw.get('save_picks', True)),
        )
        bt_summary = run_backtest(bt_cfg)
        logger.info("Backtest summary: %s", json.dumps(bt_summary, indent=2))

    logger.info("Pipeline run finished successfully.")


if __name__ == "__main__":
    main()