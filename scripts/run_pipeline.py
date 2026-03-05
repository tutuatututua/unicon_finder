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
import os
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

from scripts.data.data_download import download_full_history
from scripts.data.download_all_us_tickers import get_all_tickers
from scripts.feature.feature_engineering import build_features
from scripts.config.logging import get_logger
from scripts.predict import predict
from scripts.train.config import TrainValidSplitConfig
from scripts.train.pipeline import train_single_split
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
    metrics: dict[str, Any] = {}
    force_retrain = checkpoint_cfg.force_retrain
    if force_retrain or not _has_trained_model(paths.model_file):
        logger.info("Training ranking head (LambdaRank) (force=%s)...", force_retrain)

        # Model config from YAML (optional).
        model_cfg = raw_cfg.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}

        # Threads: default to all CPU cores unless explicitly configured.
        num_threads_raw = model_cfg.get("num_threads")
        if num_threads_raw is None:
            n_threads = int(os.cpu_count() or 1)
        else:
            n_threads = int(num_threads_raw)
        if n_threads < 1:
            n_threads = 1

        primary_k = model_cfg.get("primary_k", 20)
        primary_k_int = int(primary_k) if primary_k is not None else None


        eval_at = model_cfg.get("eval_at")
        lgb_params: dict[str, Any] = {}
        if isinstance(eval_at, (list, tuple)) and eval_at:
            lgb_params["eval_at"] = [int(x) for x in eval_at if int(x) > 0]

        num_boost_round = int(model_cfg.get("num_boost_round", 2000))
        early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))

        cfg = TrainValidSplitConfig(
            num_threads=n_threads,
            lgb_params=lgb_params,
        )

        metrics["ranking"] = train_single_split(
            cfg=cfg,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            primary_k=primary_k_int,
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



if __name__ == "__main__":
    main()