"""Artifact metadata helpers.

Currently `models/features.json` stores feature list, categorical levels, and the
selected target column name. Multiple modules were re-implementing the same
"read json -> extract key" logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FeaturesMeta:
    raw: Mapping[str, Any]

    @property
    def target(self) -> str | None:
        value = self.raw.get("target")
        if value is None:
            return None
        s = str(value).strip()
        return s or None

    @property
    def categorical_levels(self) -> Mapping[str, Any]:
        value = self.raw.get("categorical_levels")
        return value if isinstance(value, Mapping) else {}


def load_features_meta(path: str | Path) -> FeaturesMeta:
    p = Path(path)
    if not p.exists():
        return FeaturesMeta(raw={})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return FeaturesMeta(raw={})
        return FeaturesMeta(raw=data)
    except Exception:
        return FeaturesMeta(raw={})


def resolve_target_column(meta_path: str | Path, *, default: str = "target_fwd_252d") -> str:
    meta = load_features_meta(meta_path)
    return meta.target or default
