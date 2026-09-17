"""Final evaluation: reload the trained model, log ROC/PR/confusion-matrix
plots and a predictions CSV as MLflow artifacts.

Usage:
    python src/churn/evaluate.py --config configs/churn.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python src/churn/evaluate.py` to resolve `src.*` imports without
# requiring PYTHONPATH to be set manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.churn.config import get_config
from src.churn.preprocess import load_dataset, split_data
from src.churn.utils import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.common.mlflow_utils import setup_mlflow


def main(config_path: str, model_path: str) -> None:
    cfg = get_config(config_path)
    setup_mlflow(cfg.mlflow.experiment_name)

    X, y = load_dataset(cfg.data.csv_path, cfg.data.target)
    _, X_test, _, y_test = split_data(X, y, cfg.data.test_size, cfg.data.random_state)

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("artifacts/churn", exist_ok=True)
    roc_path = "artifacts/churn/roc_curve.png"
    pr_path = "artifacts/churn/pr_curve.png"
    cm_path = "artifacts/churn/confusion_matrix.png"
    preds_path = "artifacts/churn/predictions.csv"

    plot_roc_curve(model, X_test, y_test, roc_path)
    plot_pr_curve(model, X_test, y_test, pr_path)
    plot_confusion_matrix(model, X_test, y_test, cm_path)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}).to_csv(
        preds_path, index=False
    )

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metric("test_roc_auc", auc)
        for path in (roc_path, pr_path, cm_path, preds_path):
            mlflow.log_artifact(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/churn.yaml")
    parser.add_argument("--model", default="artifacts/churn/model.joblib")
    args = parser.parse_args()
    main(args.config, args.model)
