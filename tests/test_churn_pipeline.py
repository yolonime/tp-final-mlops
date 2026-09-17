"""Sanity checks for the churn (Adult Income) classifier pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.churn.pipeline import build_pipeline
from src.churn.preprocess import encode_target


def _dummy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 33, 50, 29, 61],
            "hours-per-week": [40, 35, 45, 20, 38, 50],
            "workclass": ["Private", "Self-emp", "Private", "Gov", "Private", "Gov"],
            "sex": ["Male", "Female", "Female", "Male", "Male", "Female"],
        }
    )


def test_encode_target_is_binary():
    y = pd.Series(["<=50K", ">50K", "<=50K", ">50K"])
    encoded = encode_target(y)
    assert set(encoded.unique()) <= {0, 1}
    assert encoded.sum() == 2


def test_pipeline_fits_and_predicts():
    X = _dummy_dataframe()
    y = np.array([0, 1, 0, 1, 0, 1])

    pipeline = build_pipeline(
        numeric_features=["age", "hours-per-week"],
        categorical_features=["workclass", "sex"],
        model_type="logreg",
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_pipeline(["age"], ["sex"], "not-a-model")
