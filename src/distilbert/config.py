"""Configuration loader for the DistilBERT sentiment pipeline."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.common.config import load_config


def get_config(config_path: str | None = None) -> SimpleNamespace:
    """Return the DistilBERT pipeline configuration as an attribute-accessible object."""
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="configs/distilbert.yaml")
        args, _ = parser.parse_known_args()
        config_path = args.config
    return load_config(config_path)
