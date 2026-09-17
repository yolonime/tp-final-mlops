"""Build the preprocessing ColumnTransformer + model sklearn Pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

DEFAULT_MODEL_KWARGS = {
    "logreg": {"max_iter": 1000},
    "random_forest": {"random_state": 42},
}


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def build_pipeline(
    numeric_features: list[str], categorical_features: list[str], model_type: str
) -> Pipeline:
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Expected one of {list(MODELS)}."
        )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = MODELS[model_type](**DEFAULT_MODEL_KWARGS[model_type])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def param_grid_for_model(params: dict) -> dict:
    """Prefix a flat param grid with `model__` for use with the Pipeline above."""
    return {f"model__{k}": v for k, v in params.items()}
