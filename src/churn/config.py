"""Configuration loader for the churn (Adult Income) classifier pipeline."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.common.config import load_config


def get_config(config_path: str | None = None) -> SimpleNamespace:
    """Return the churn pipeline configuration as an attribute-accessible object.

    If no path is given, parses `--config` from `sys.argv` (defaulting to
    `configs/churn.yaml`), so scripts can be run standalone with a CLI flag.
    """
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="configs/churn.yaml")
        args, _ = parser.parse_known_args()
        config_path = args.config
    return load_config(config_path)
