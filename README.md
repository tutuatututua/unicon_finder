# unicon_finder

Learn-to-rank pipeline for US equities: download price history, build a feature panel, train a LightGBM **LambdaRank** model with a simple **time-based train/validation split**, and generate top-\(N\) ticker predictions.

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

## Walk-forward backtest (out-of-sample)

To measure **realistic historical performance**, use the purged walk-forward backtest. It repeatedly:

- trains on an earlier window
- validates on a later window
- tests on the next window (out-of-sample)

It also applies a **purge gap** (by default equal to the target horizon, e.g. 252 trading days for `target_fwd_252d`) to reduce label-overlap leakage.

```powershell
python .\scripts\backtest\walkforward_backtest.py --valid-days 252 --test-days 63 --step-days 63 --top-n 20 --rebalance-step 21
```

Outputs are written under `backtest/walkforward/`:

- `summary.json`: high-level summary
- `splits.csv`: metrics per walk-forward split
- `trades.csv`: per-rebalance portfolio forward returns vs benchmark

## Training: how it works

Training is implemented in [scripts/train/pipeline.py](scripts/train/pipeline.py) and uses a **single chronological split**:

- **Group/query definition:** each *date* is a ranking query; all tickers on that date form the items to rank.
- **Label:** continuous forward return (default `target_fwd_252d`) is converted into discrete **relevance bins** using **per-date ranking**.
- **Leakage control:** a `forward_gap_days` gap is enforced between `train_end` and `valid_start`.

The trainer:

1. **Loads the labeled panel** from `data/processed/raw_training.parquet`.
2. **Splits once** into TRAIN and VALID by date (VALID is the last `valid_days` dates).
3. **Builds group sizes per date** for LightGBM LambdaRank.
4. **Optionally scales numeric features** (z-score) fit on TRAIN only.
5. **Trains LightGBM once** with early stopping based on VALID NDCG.
6. **Writes artifacts** (model, metrics, features metadata) into `models/`.

## Configuration

All runtime toggles live in `config.yaml`.

- `checkpoint.force_fetch`: re-download raw history
- `checkpoint.force_rebuild`: rebuild features and datasets
- `checkpoint.force_retrain`: retrain the model

Model-related keys (examples):

- `model.params`: LightGBM parameter overrides
- `model.num_boost_round`, `model.early_stopping_rounds`, `model.primary_k`
- `model.split.*`: train/validation split settings

## Outputs

Typical generated artifacts:

- `data/raw/`: downloaded price history
- `data/processed/`: feature panel and snapshots
- `models/lightgbm_model.txt`: trained model
- `models/metrics.json`: validation metrics for the single split
- `models/feature_importance.csv` (+ optional PNG)
- `models/features.json`: metadata used at prediction time (features, categorical levels, preprocessing)
- `backtest/`: backtest CSV/JSON outputs (if enabled)

## Troubleshooting

- **Not enough tickers per date:** decrease `model.split.min_cross_section`.
- **Not enough history for split:** decrease `model.split.valid_days` or `model.split.forward_gap_days`.
- **LightGBM errors about groups:** ensure your feature panel has many tickers per date.

## Project layout (high level)

- [scripts/run_pipeline.py](scripts/run_pipeline.py): orchestrates the full pipeline
- [scripts/train/train.py](scripts/train/train.py): training entrypoint
- [scripts/train/pipeline.py](scripts/train/pipeline.py): single-split training implementation
- `data/`: universe, sector map, raw downloads, processed features
- `models/`: trained model + training artifacts
- `tests/`: unit/regression tests
