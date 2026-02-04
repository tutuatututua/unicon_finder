# Copilot instructions for `unicon_finder`

## Big picture (what this repo is)
- End-to-end learn-to-rank pipeline for US equities: raw price history → feature panel → LightGBM LambdaRank (single time-based train/validation split) → predictions + optional backtest.
- Orchestrator is intentionally thin; heavy logic lives under `scripts/` (see `scripts/run_pipeline.py`).

## Key entrypoints (use these first)
- Full pipeline: `python scripts/run_pipeline.py` (stages + caching controlled by `config.yaml`).
- Training API: `scripts/train/train.py::train_model_learn_to_rank()`.
- Training core: `scripts/train/pipeline.py::train_single_split()`.
- Feature build: `scripts/feature/feature_engineering/artifacts.py::build_features()`.
- Predict: `scripts/predict.py::predict()` (scores `data/processed/latest_data.parquet` or `extract_training.parquet`).
- Backtest (strict, no fallbacks): `scripts/backtest/backtest.py`.

## Data flow + artifacts (important invariants)
- Raw history: per-ticker parquet in `data/raw/<TICKER>.parquet` downloaded via yfinance (`scripts/data/data_download.py`). Dates are normalized to UTC-naive daily (`.dt.tz_localize(None).dt.normalize()`).
- Feature panel: `data/processed/raw_training.parquet` (must include at least `ticker`, `date`, target like `target_fwd_252d`, plus features).
- Prediction snapshots written by feature build:
  - `data/processed/latest_data.parquet` (latest unlabeled rows)
  - `data/processed/extract_training.parquet` (last labeled snapshot)
- Training overwrites `models/features.json` with **runtime-critical metadata** (not just feature list):
  - `features`, `target`, `categorical_levels`, `preprocessing` (see `scripts/train/pipeline.py`).
  Prediction and backtest rely on this to align categorical dtypes and preprocessing.
- Model: `models/lightgbm_model.txt`; metrics: `models/metrics.json`; predictions: `models/predictions.csv`.

## Repo-specific ML conventions (don’t break these)
- Group/query for ranking is **per date**: each `date` is a query; all tickers that day are items.
- Leakage control is enforced via `forward_gap_days` between train end and valid start (tests cover this in `tests/test_panel_split_gap.py`).
- Relevance labels are binned from continuous forward returns using per-date ranking.
- Determinism is preferred: `TrainValidSplitConfig.num_threads = 1` by default.

## Configuration + caching behavior
- `config.yaml` controls:
  - checkpoint flags: `checkpoint.force_fetch|force_rebuild|force_retrain`
  - LightGBM overrides under `model.params`
  - split settings under `model.split.*`
- Pipeline stages skip work based on file existence (see `_has_raw_history`, `_has_feature_artifacts`, `_has_trained_model` in `scripts/run_pipeline.py`).

## Dev workflows (commands that matter here)
- Setup (Windows PowerShell):
  - `python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt`
- Run pipeline: `python .\scripts\run_pipeline.py`
- Run tests: `python -m pytest -q`

## When changing code
- If you change feature columns, targets, categorical handling, or preprocessing, update both:
  - feature builder outputs (`data/processed/*.parquet` via `build_features()`), and
  - training metadata writing (`models/features.json` via `train_single_split()`) so predict/backtest stay consistent.
- Backtest is strict: missing model/features/benchmark files should raise (don’t silently “fallback”).
