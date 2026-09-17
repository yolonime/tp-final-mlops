"""Generic YAML config loading shared by both pipelines.

Both `src/churn/config.py` and `src/distilbert/config.py` build on top of
this: it turns a YAML file into a nested, attribute-accessible object
(`cfg.data.csv_path` instead of `cfg["data"]["csv_path"]`).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    """Load a YAML config file into an attribute-accessible namespace."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
