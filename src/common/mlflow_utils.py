"""MLflow tracking setup shared by both pipelines."""
from __future__ import annotations

import os

import mlflow


def setup_mlflow(default_experiment_name: str) -> None:
    """Point MLflow at the configured tracking server and experiment.

    `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` (see `.env.example`)
    override the config-file defaults when set, matching the Makefile
    pattern `MLFLOW_EXPERIMENT_NAME=$(EXP) python src/.../train.py`.
    """
    # MLflow 3.x deprecated the plain filesystem store; sqlite is the
    # lightweight local default that still supports the Model Registry.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or default_experiment_name
    mlflow.set_experiment(experiment_name)
