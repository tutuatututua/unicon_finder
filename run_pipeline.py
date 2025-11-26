# run_pipeline.py

"""End-to-end ML pipeline orchestrator (ranking-only).

Stages (controlled via `config.yaml`):
 1. Configuration loading.
 2. Historical data download (skipped if not forced and cache present).
 3. Feature engineering (optionally skipped via checkpoint flags).
 4. Model training (ranking head only).
 5. Prediction generation on latest or last labeled snapshot.

Model mode: lambdarank (ranking head).

Checkpoint JSON markers with timestamps are written under `.checkpoints/`.
This script is intentionally thin: heavy logic lives in dedicated modules.
"""

from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd

from data_download import download_full_history
from download_all_us_tickers import get_all_tickers
from features import build_features
from train import train_model_learn_to_rank, _json_default  # reuse serializer
from predict import predict
from logging_config import get_logger
from backtest import run_backtest, BacktestConfig
logger = get_logger(__name__)

CONFIG_PATH = Path("config.yaml")

TICKERS_CACHE = Path("data/universe.csv")
CHECKPOINT_DIR = Path(".checkpoints")
FETCH_MARKER = Path(".checkpoints/fetch.json")
FEATURES_MARKER = Path(".checkpoints/features.json")
TRAIN_MARKER = Path(".checkpoints/train.json")
RAW_FEATURES_PATH = Path("data/processed/latest_data.parquet")
UNIVERSE_PATH = Path("data/universe.csv")
SECTORS_PATH = Path(".sectors.csv")

def _ensure_dir(p: Path) -> None:
    """Create directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _write_marker(path: Path, payload: dict) -> None:
    """Write a JSON checkpoint marker with merged timestamp (best-effort).

    Uses the shared _json_default serializer from ranking trainer to avoid
    crashes when metrics contain numpy scalar types (e.g., int64).
    """
    try:
        _ensure_dir(path.parent)
        payload_with_ts = {"timestamp": datetime.utcnow().isoformat(timespec='seconds') + 'Z', **payload}
        path.write_text(json.dumps(payload_with_ts, indent=2, default=_json_default), encoding="utf-8")
    except OSError as e:  # pragma: no cover (non-critical)
        logger.warning("Failed writing checkpoint %s: %s", path, e)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _download_history() -> None:
    """
    Downloads historical data for tickers defined in the universe file.
    Skips if a valid checkpoint exists and data files are present, unless forced.
    """
    logger.info("Starting full-history download for tickers in %s...", TICKERS_CACHE)
    if not TICKERS_CACHE.exists():
        logger.error("Ticker universe file not found at %s. Cannot proceed.", TICKERS_CACHE)
        raise FileNotFoundError(f"Missing ticker universe file: {TICKERS_CACHE}")

    try:
        tickers = download_full_history(str(TICKERS_CACHE))
        successful_downloads = len(tickers)
        logger.info("Download complete: %d tickers processed.", successful_downloads)
        _write_marker(FETCH_MARKER, {"tickers_processed": successful_downloads})
    except Exception as e:
        logger.error("An error occurred during data download: %s", e, exc_info=True)
        raise


def main() -> None:
    """Runs the full ML pipeline from data ingestion to prediction."""

    # --- Stage 1: Configuration ---
    logger.info("Loading configuration from %s...", CONFIG_PATH)
    raw_cfg = load_config(CONFIG_PATH)
    
    # Create structured config objects for type safety and clarity
    checkpoint_cfg = raw_cfg.get('checkpoint', {})
    prediction_cfg = raw_cfg.get('prediction', {})


    # Ensure checkpoint directory exists (may differ from default now)
    _ensure_dir(CHECKPOINT_DIR)

    # --- Stage 2: Ticker Universe and Sector Download ---
    if not UNIVERSE_PATH.exists():
        logger.info("Fetching all US stock tickers (NASDAQ, NYSE, AMEX)...")
        syms = get_all_tickers(include_etfs=False)
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"ticker": syms}).to_csv("data/universe.csv", index=False)
        logger.info(f"Saved {len(syms)} tickers to data/universe.csv")
    else:
        logger.info("Universe file found at %s. Skipping ticker download.", UNIVERSE_PATH)

    if not SECTORS_PATH.exists():
        from download_sectors import main as download_sectors_main
        logger.info("Downloading sector/industry mapping for universe tickers...")
        download_sectors_main()
    else:
        logger.info("Sector map file found at %s. Skipping sector download.", SECTORS_PATH)

    # --- Stage 3: Data Download ---
    if checkpoint_cfg.get("force_fetch", True):
        logger.info("Downloading new historical data...")
        _download_history()

    # --- Stage 4: Feature Engineering ---
    logger.info("Starting feature engineering stage...")
    if checkpoint_cfg.get("force_rebuild", True):
        logger.info("Building new features...")
        features_df = build_features()
        _write_marker(FEATURES_MARKER, {"rows": len(features_df), "features": len(features_df.columns) - 3})
    else:
        logger.info(
            "Feature checkpoint found. Skipping feature build. Set checkpoint.force_rebuild to true to override."
        )
        # Reuse existing processed training features parquet (full panel) when not rebuilding.
        features_df = pd.read_parquet(RAW_FEATURES_PATH)
    logger.info("Feature engineering complete.")

    # --- Stage 5: Model Training ---
    logger.info("Starting model training stage...")
    model_cfg = raw_cfg.get('model', {})

    metrics: Dict[str, Any] = {}
    if checkpoint_cfg.get("force_retrain", False):
        logger.info("Training ranking head (LambdaRank)...")
        metrics['ranking'] = train_model_learn_to_rank(
            tune=model_cfg.get('tune_regularization', False),
            tune_param_grid=model_cfg.get('tune_param_grid'),
            max_combinations=int(model_cfg.get('tune_max_evals', 200)),
            params=model_cfg.get('params'),
            early_stopping_rounds=int(model_cfg.get('early_stopping_rounds', 20)),
            primary_k=int(model_cfg.get('primary_k', 20)),
        )
        # Persist metrics to marker
        _write_marker(TRAIN_MARKER, metrics)
    logger.info("Model training complete. Heads present: %s", ", ".join(metrics.keys()) or "none")

    # --- Stage 6: Prediction ---
    logger.info("Generating predictions on latest data...")
    top_n = prediction_cfg.get('top_n', 20)
    # Determine which snapshot to use for prediction based on config
    snapshot_mode = 'last_labeled' if prediction_cfg.get('use_last_labeled') else 'latest'
    
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
        )
        bt_summary = run_backtest(bt_cfg)
        logger.info("Backtest summary: %s", json.dumps(bt_summary, indent=2))

    logger.info("Pipeline run finished successfully.")


if __name__ == "__main__":
    main()