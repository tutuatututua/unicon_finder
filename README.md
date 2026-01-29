# unicon_finder

Learn-to-rank pipeline for US equities: download price history, build a feature panel, train a LightGBM **LambdaRank** model with **walk-forward validation**, and generate top-\(N\) ticker predictions.

## Quick start (Windows / PowerShell)

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the full pipeline (download -> features -> train -> predict)
python .\scripts\run_pipeline.py
```

Run tests:

```powershell
python -m pytest -q
```

## What the pipeline does

The main entrypoint is [scripts/run_pipeline.py](scripts/run_pipeline.py). It runs these stages:

1. **Config load** from `config.yaml`
2. **Universe + sectors**
   - creates `data/universe.csv` if missing
   - creates `data/sector_map.csv` if missing
3. **Raw data download** into `data/raw/` (cached; controlled by `checkpoint.force_fetch`)
4. **Feature engineering** into `data/processed/` (cached; controlled by `checkpoint.force_rebuild`)
   - produces a training panel `data/processed/raw_training.parquet`
   - produces prediction snapshots (latest / last-labeled)
5. **Model training** into `models/` (cached; controlled by `checkpoint.force_retrain`)
6. **Prediction** (top-\(N\) tickers)
7. **Backtest** (optional; `backtest.run`)

## Training: how it works

Training is implemented in [scripts/train/wf/pipeline.py](scripts/train/wf/pipeline.py) and uses **walk-forward validation**:

- **Group/query definition:** each *date* is a ranking query; all tickers on that date form the items to rank.
- **Label:** continuous forward return (default `target_fwd_252d`) is converted into discrete **relevance bins** (quantiles).
- **Leakage control:** a `forward_gap_days` gap is enforced between train and validation windows.

### One fold (example)

Assume:
- `train_window_days = 1095`
- `valid_days = 365`
- `forward_gap_days = 252`

If a fold has:
- `valid_start = 2022-01-03`
- `valid_end_exclusive = 2023-01-03`

Then:
- `train_end` is snapped to the nearest available trading date at `valid_start - 252 days` (about `2021-04-26`)
- `train_start` is snapped to `train_end - 1095 days` (about `2018-04-26`)

Within that fold, the trainer:

1. **Selects rows** in TRAIN and VALID windows.
2. **Fits relevance bins on TRAIN only** (quantile edges via `qcut`) and applies them to both TRAIN and VALID.
3. **Builds group sizes per date** for LightGBM LambdaRank.
4. **Optionally scales numeric features** (z-score) fit on TRAIN only.
5. **Trains LightGBM** with early stopping based on VALID NDCG.
6. **Logs metrics** for the fold (NDCG@k and Spearman on VALID).

After all folds, it trains a **final model** on the full eligible training history (up to the latest date), using the average fold `best_iteration` as the final boosting rounds.

## Configuration

All runtime toggles live in `config.yaml`.

- `checkpoint.force_fetch`: re-download raw history
- `checkpoint.force_rebuild`: rebuild features and datasets
- `checkpoint.force_retrain`: retrain the model

Model-related keys (examples):

- `model.params`: LightGBM parameter overrides
- `model.tune_regularization`: enable random-search tuning
- `model.tune_param_grid`, `model.tune_max_evals`
- `model.early_stopping_rounds`, `model.primary_k`

Walk-forward window settings (e.g. `valid_days`, `step_days`, `train_window_days`, `forward_gap_days`, `min_cross_section`) are configured in code via `WalkForwardConfig`.

- Defaults live in [scripts/train/wf/config.py](scripts/train/wf/config.py).
- To change them for pipeline runs, pass an explicit config from [scripts/run_pipeline.py](scripts/run_pipeline.py) into `train_model_learn_to_rank(cfg=WalkForwardConfig(...))`.

## Outputs

Typical generated artifacts:

- `data/raw/`: downloaded price history
- `data/processed/`: feature panel and snapshots
- `models/lightgbm_model.txt`: trained model
- `models/metrics.json`: fold metrics + aggregate metrics
- `models/feature_importance.csv` (+ optional PNG)
- `models/features.json`, `models/preprocessing.json`: metadata used at prediction time
- `backtest/`: backtest CSV/JSON outputs (if enabled)

## Troubleshooting

- **Not enough tickers per date:** decrease `WalkForwardConfig(min_cross_section=...)`.
- **Training is slow:** increase `WalkForwardConfig(train_date_step=...)` / `WalkForwardConfig(valid_date_step=...)`, or reduce `WalkForwardConfig(train_window_days=...)`.
- **LightGBM errors about groups:** ensure your feature panel has many tickers per date and dates aren’t filtered too aggressively.

## Project layout (high level)

- [scripts/run_pipeline.py](scripts/run_pipeline.py): orchestrates the full pipeline
- [scripts/train/train.py](scripts/train/train.py): training entrypoint (calls walk-forward trainer)
- [scripts/train/wf/](scripts/train/wf/): walk-forward training implementation
- `data/`: universe, sector map, raw downloads, processed features
- `models/`: trained model + training artifacts
- `tests/`: unit/regression tests
