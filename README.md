# Unicon Finder

Cross-sectional stock ranking pipeline using LightGBM LambdaRank. It downloads data, builds features, trains a ranking model, generates predictions, and backtests performance.

## Quick Start

- Python: 3.11+
- Install deps:
  ```sh
  pip install -r requirements.txt
  ```

- End-to-end pipeline:
  ```sh
  python run_pipeline.py
  ```
  See [run_pipeline.py](run_pipeline.py). It writes checkpoints under [.checkpoints/](.checkpoints/) and artifacts in [models/](models/) and [data/processed/](data/processed/).

## Main Steps

1) Download historical data
- Universe auto-fetch or provide your own CSV to [data/universe.csv](data/universe.csv).
- Sector map auto-fetch to [data/sector_map.csv](data/sector_map.csv).
- Daily OHLCV saved under [data/raw/](data/raw/).
- Entry points:
  - [`data_download.download_full_history`](data_download.py)
  - [`download_all_us_tickers.get_all_tickers`](download_all_us_tickers.py)
  - [`download_sectors.main`](download_sectors.py)

2) Build features
- Full panel + snapshots written by [`features.build_features`](features.py):
  - [data/processed/raw_training.parquet](data/processed/raw_training.parquet) (full panel)
  - [data/processed/latest_data.parquet](data/processed/latest_data.parquet) (latest unlabeled)
  - [data/processed/extract_training.parquet](data/processed/extract_training.parquet) (last labeled)
  - Feature engineering per ticker via [`features.compute_features_for_ticker`](features.py). Sector merge in [`features.assemble_dataset`](features.py).

3) Train model (LambdaRank)
- Train and persist model artifacts:
  ```sh
  python -c "from train import train_model_learn_to_rank; print(train_model_learn_to_rank())"
  ```
- Entry points and artifacts:
  - [`train.train_model_learn_to_rank`](train.py)
  - Model: [models/lightgbm_model.txt](models/lightgbm_model.txt)
  - Metrics: [models/metrics.json](models/metrics.json)
  - Feature meta: [models/features.json](models/features.json)

4) Predict
- Generate ranked predictions on latest or last labeled snapshot:
  ```sh
  python predict.py --top-n 20 --snapshot latest
  ```
- Function: [`predict.predict`](predict.py)
- Output CSV: [models/predictions.csv](models/predictions.csv)

5) Backtest
- Strict cross-sectional backtest using model score or a single feature:
  ```sh
  python backtest.py --use-model-score --top-n 20 --bottom-n 20
  ```
- Entrypoint: [`backtest.run_backtest`](backtest.py)
- Outputs under [backtest/](backtest):
  - backtest_timeseries.csv, backtest_deciles.csv, backtest_ndcg.csv, backtest_summary.json

## Aggregation Utility

Aggregate daily Top-N model scores across dates and optionally compute NDCG:
- Script: [`scripts.aggregate_topN`](scripts/aggregate_top500_scores.py)
- Example:
  ```sh
  python scripts/aggregate_top500_scores.py --top-n 500 --last-days 10 --generate-ndcg
  ```

## Configuration

- Global config: [config.yaml](config.yaml)
  - Training hyperparameter search bounds
  - Prediction defaults
  - Backtest settings

## Project Structure

- Pipeline: [run_pipeline.py](run_pipeline.py)
- Data ingest: [data_download.py](data_download.py), [download_all_us_tickers.py](download_all_us_tickers.py), [download_sectors.py](download_sectors.py)
- Features: [features.py](features.py)
- Training: [train.py](train.py)
- Prediction: [predict.py](predict.py)
- Backtesting: [backtest.py](backtest.py)
- EDA utilities: [eda/](eda)

## Notes

- Model objective: LambdaRank with NDCG metrics ($\text{objective}=\text{lambdarank}$).
- Target column is dynamically tracked in [models/features.json](models/features.json).
- Sector/industry are categorical and aligned with model’s pandas_categorical levels.
- For manual date scoring:
  ```sh
  python predict.py --date YYYY-MM-DD
  ```

## Troubleshooting

- Missing features parquet: rebuild via [`features.build_features`](features.py).
- Missing model: train via [`train.train_model_learn_to_rank`](train.py).
- Benchmark S&P 500 cache: use [scripts/fetch_sp500_cache.py](scripts/fetch_sp500_cache.py).
