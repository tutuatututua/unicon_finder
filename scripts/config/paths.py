from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import get_paths_overrides


def _as_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    try:
        return Path(str(value))
    except Exception:
        return default


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path = Path(".")

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    checkpoints_dir: Path = Path(".checkpoints")

    models_dir: Path = Path("models")
    backtest_dir: Path = Path("backtest")

    universe_csv: Path = Path("data/universe.csv")
    sector_map_csv: Path = Path("data/sector_map.csv")

    raw_training_parquet: Path = Path("data/processed/raw_training.parquet")
    latest_snapshot_parquet: Path = Path("data/processed/latest_data.parquet")
    last_labeled_snapshot_parquet: Path = Path("data/processed/extract_training.parquet")

    features_meta_json: Path = Path("models/features.json")
    model_file: Path = Path("models/lightgbm_model.txt")
    predictions_csv: Path = Path("models/predictions.csv")

    @staticmethod
    def from_config(cfg: Mapping[str, Any], *, repo_root: str | Path = ".") -> "ProjectPaths":
        root = Path(repo_root)
        overrides = get_paths_overrides(cfg)

        raw_dir = _as_path(overrides.get("raw_dir"), Path("data/raw"))
        processed_dir = _as_path(overrides.get("processed_dir"), Path("data/processed"))
        checkpoints_dir = _as_path(overrides.get("checkpoints_dir"), Path(".checkpoints"))
        models_dir = _as_path(overrides.get("models_dir"), Path("models"))
        backtest_dir = _as_path(overrides.get("backtest_dir"), Path("backtest"))

        universe_csv = _as_path(overrides.get("universe_csv"), Path("data/universe.csv"))
        sector_map_csv = _as_path(overrides.get("sector_map_csv"), Path("data/sector_map.csv"))

        raw_training_parquet = processed_dir / "raw_training.parquet"
        latest_snapshot_parquet = processed_dir / "latest_data.parquet"
        last_labeled_snapshot_parquet = processed_dir / "extract_training.parquet"

        features_meta_json = models_dir / "features.json"
        model_file = models_dir / "lightgbm_model.txt"
        predictions_csv = models_dir / "predictions.csv"

        return ProjectPaths(
            repo_root=root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            checkpoints_dir=checkpoints_dir,
            models_dir=models_dir,
            backtest_dir=backtest_dir,
            universe_csv=universe_csv,
            sector_map_csv=sector_map_csv,
            raw_training_parquet=raw_training_parquet,
            latest_snapshot_parquet=latest_snapshot_parquet,
            last_labeled_snapshot_parquet=last_labeled_snapshot_parquet,
            features_meta_json=features_meta_json,
            model_file=model_file,
            predictions_csv=predictions_csv,
        )

    def ensure_dirs(self) -> None:
        """Create directories used by the pipeline if they do not exist."""
        for p in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.backtest_dir,
            self.universe_csv.parent,
            self.sector_map_csv.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)
