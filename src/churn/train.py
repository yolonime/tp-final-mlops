"""Train + tune the churn (Adult Income) classifier with MLflow autologging.

Usage:
    python src/churn/train.py --config configs/churn.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python src/churn/train.py` to resolve `src.*` imports without
# requiring PYTHONPATH to be set manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.churn.config import get_config
from src.churn.pipeline import build_pipeline, param_grid_for_model
from src.churn.preprocess import load_dataset, split_data
from src.common.mlflow_utils import setup_mlflow


def main(config_path: str) -> None:
    cfg = get_config(config_path)
    setup_mlflow(cfg.mlflow.experiment_name)

    X, y = load_dataset(cfg.data.csv_path, cfg.data.target)
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg.data.test_size, cfg.data.random_state
    )

    pipeline = build_pipeline(cfg.features.numeric, cfg.features.categorical, cfg.model.type)
    param_grid = param_grid_for_model(vars(cfg.model.params))

    cv = StratifiedKFold(
        n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.data.random_state
    )

    mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

    with mlflow.start_run(run_name=f"{cfg.model.type}-gridsearch"):
        search = GridSearchCV(pipeline, param_grid, scoring=cfg.cv.scoring, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)

        test_score = search.score(X_test, y_test)
        mlflow.log_metric(f"test_{cfg.cv.scoring}", test_score)
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})

        print(f"Best params: {search.best_params_}")
        print(f"Best CV {cfg.cv.scoring}: {search.best_score_:.4f}")
        print(f"Test {cfg.cv.scoring}: {test_score:.4f}")

        registered_name = getattr(cfg.mlflow, "registered_model_name", None)
        if registered_name:
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                registered_model_name=registered_name,
                # MLflow 3.x defaults to `skops`, which rejects some dtypes
                # used by our ColumnTransformer; cloudpickle is fine for a
                # model we both save and load ourselves.
                serialization_format="cloudpickle",
            )

    os.makedirs("artifacts/churn", exist_ok=True)
    joblib.dump(search.best_estimator_, "artifacts/churn/model.joblib")
    print("Saved best estimator to artifacts/churn/model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/churn.yaml")
    args = parser.parse_args()
    main(args.config)
