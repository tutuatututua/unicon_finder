"""Feature engineering subpackage.

This package contains the implementation for per-ticker feature computation,
panel assembly, and artifact generation.

`features.py` at the repo root remains the public, backwards-compatible API
wrapper for scripts that import it.
"""

from __future__ import annotations

from .feature_config import FeatureConfig
from .dataset import assemble_dataset
from .artifacts import build_features

__all__ = ["FeatureConfig", "assemble_dataset", "build_features"]
