"""Configuration loading and typed access helpers.

The project currently uses a single YAML file (default: `config.yaml`).
This module centralizes YAML parsing so individual scripts don't duplicate
"open -> yaml.safe_load -> dict" logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigDict = dict[str, Any]


def load_yaml_config(path: str | Path = "config.yaml") -> ConfigDict:
    """Load a YAML config file.

    Returns an empty dict if the file does not exist.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def get_section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Safely fetch a nested config section as a mapping."""
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def get_paths_overrides(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the optional `paths:` section from config."""
    return get_section(cfg, "paths")


@dataclass(frozen=True)
class CheckpointConfig:
    force_fetch: bool = False
    force_rebuild: bool = False
    force_retrain: bool = False

    @staticmethod
    def from_config(cfg: Mapping[str, Any]) -> "CheckpointConfig":
        section = get_section(cfg, "checkpoint")
        return CheckpointConfig(
            force_fetch=bool(section.get("force_fetch", False)),
            force_rebuild=bool(section.get("force_rebuild", False)),
            force_retrain=bool(section.get("force_retrain", False)),
        )


@dataclass(frozen=True)
class PredictionConfig:
    top_n: int = 20
    use_last_labeled: bool = False

    @staticmethod
    def from_config(cfg: Mapping[str, Any]) -> "PredictionConfig":
        section = get_section(cfg, "prediction")
        return PredictionConfig(
            top_n=int(section.get("top_n", 20)),
            use_last_labeled=bool(section.get("use_last_labeled", False)),
        )
