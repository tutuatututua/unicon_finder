"""Shared library code for the Unicon Finder project.

This package intentionally contains no heavy ML dependencies at import time.
Scripts at the repo root (e.g., run_pipeline.py) should stay thin entrypoints
that import and call functions from here.
"""

from __future__ import annotations

__all__ = [
    "metadata",
    "feature_engineering",
]
